# infra/data_manager/downloader_etf.py
# -*- coding: utf-8 -*-

import os
import pandas as pd
import FinanceDataReader as fdr
from tqdm.auto import tqdm
import logging
from datetime import datetime

# 로깅 설정 (UTF-8 호환)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)


class ETFDownloader:
    def __init__(self, base_dir="data/raw_etf", csv_path="etf_list_20260115.csv", start_date="2005-01-03",
                 end_date="2026-01-15"):
        self.base_dir = base_dir
        self.csv_path = csv_path
        self.start_date = start_date
        self.end_date = end_date

        # 저장 디렉토리 생성
        os.makedirs(self.base_dir, exist_ok=True)

    def _load_etf_list(self):
        """ETF 목록 로드 (UTF-8 우선, 실패시 cp949)"""
        if not os.path.exists(self.csv_path):
            logger.warning(f"⚠️ CSV 파일 없음: {self.csv_path}")
            return [], {}

        try:
            # 1. UTF-8 시도
            df = pd.read_csv(self.csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # 2. CP949 시도 (한글 윈도우 파일 대비)
                logger.info("UTF-8 디코딩 실패, CP949로 시도합니다.")
                df = pd.read_csv(self.csv_path, encoding='cp949')
            except Exception as e:
                logger.error(f"❌ 파일 읽기 완전 실패: {e}")
                return [], {}

        # ★ 종목코드 문자열(str) 변환 및 6자리 패딩 보장 ★
        df['종목코드'] = df['종목코드'].astype(str).str.strip().str.zfill(6)

        # 상장좌수 숫자 변환
        if df['상장좌수'].dtype == object:
            df['상장좌수'] = df['상장좌수'].astype(str).str.replace(',', '').astype(float)

        shares_map = df.set_index('종목코드')['상장좌수'].to_dict()
        target_codes = df['종목코드'].tolist()

        return target_codes, shares_map

    def process_etf(self, code, shares_map):
        # 코드 문자열 재확인
        code = str(code).zfill(6)
        save_path = os.path.join(self.base_dir, f"{code}.parquet")

        if os.path.exists(save_path):
            return "Skip"

        try:
            df = fdr.DataReader(code, self.start_date, self.end_date)
            if df.empty: return "Empty"

            df = df.reset_index()

            # 계산 로직
            df['Amount'] = df['Close'] * df['Volume']
            shares = shares_map.get(code, 0)
            df['Marcap'] = df['Close'] * shares

            # ★ 코드 컬럼 문자열 보장 ★
            df['Code'] = str(code)

            # 컬럼 정리
            cols_needed = ['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Marcap']
            for c in cols_needed:
                if c not in df.columns: df[c] = 0
            df = df[cols_needed]

            # 데이터 타입 (Int64)
            int_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Marcap']
            for c in int_cols:
                df[c] = df[c].fillna(0).astype('Int64')

            df.to_parquet(save_path, index=False)
            return "Success"

        except Exception as e:
            return f"Error"

    def run(self):
        logger.info("📂 ETF 리스트 로딩 시작...")
        codes, shares_map = self._load_etf_list()

        if not codes:
            logger.error("대상 종목이 없습니다. 종료합니다.")
            return

        logger.info(f"🚀 총 {len(codes)}개 ETF 다운로드 시작")

        results = []
        for code in tqdm(codes):
            res = self.process_etf(code, shares_map)
            results.append(res)

        logger.info("작업 완료!")
        logger.info(f"성공: {results.count('Success')}, 스킵: {results.count('Skip')}")


if __name__ == "__main__":
    # 실행 경로 설정
    downloader = ETFDownloader(
        base_dir="data/raw_etf",
        csv_path="data/raw/etf_list_20260115.csv"
    )
    downloader.run()
