# infra/data_manager/downloader.py
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import FinanceDataReader as fdr
from tqdm.auto import tqdm
import logging
from datetime import datetime
from marcap import marcap_data

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockDownloader:
    def __init__(self, base_dir="data/raw", start_date="2005-01-03", end_date="2026-01-15"):
        """
        :param base_dir: 데이터 저장 루트 경로 (예: data/raw)
        :param start_date: 수집 시작일
        :param end_date: 수집 종료일
        """
        self.base_dir = base_dir
        self.start_date = start_date
        self.end_date = end_date
        self.marcap_dir = os.path.join(os.path.dirname(self.base_dir), "marcap_repo")  # marcap 저장소 경로

        # 저장 디렉토리 생성
        os.makedirs(self.base_dir, exist_ok=True)

        # Marcap 준비
        self._prepare_marcap()

    def _prepare_marcap(self):
        """Marcap 라이브러리가 없으면 Git Clone 후 경로 추가"""
        if not os.path.exists(self.marcap_dir):
            logger.info("Marcap 라이브러리 다운로드 중...")
            try:
                # git이 설치된 환경이어야 함
                os.system(f"git clone --depth 1 https://github.com/FinanceData/marcap.git {self.marcap_dir}")
            except Exception as e:
                logger.error(f"Marcap Clone 실패: {e}")

        if self.marcap_dir not in sys.path:
            sys.path.append(self.marcap_dir)

    def _load_marcap_memory(self):
        """Marcap 데이터를 메모리에 로드 (2025년까지)"""
        logger.info("🚀 Marcap 전체 데이터 로딩 중 (메모리 적재)...")
        # 2025년 말까지만 로드 (2026 에러 방지)
        df_all = marcap_data("2005-01-03", "2025-12-31")

        logger.info("📦 종목별 그룹핑 중...")
        # Dictionary로 변환 (Code -> DataFrame)
        return dict(tuple(df_all.groupby('Code')))

    def _get_target_list(self):
        """KRX 전체 종목 리스트 및 상장주식수 맵핑 확보"""
        logger.info("KRX 종목 리스트 확보 중 (FDR)...")
        stocks = fdr.StockListing('KRX')
        stocks['Code'] = stocks['Code'].astype(str).str.zfill(6)

        # 상장주식수 컬럼 찾기 (한글/영문 대응)
        stock_col = next((c for c in stocks.columns if 'Stocks' in c or '상장주식수' in c), None)
        shares_map = stocks.set_index('Code')[stock_col].to_dict() if stock_col else {}

        # 종목코드 리스트 저장 (요청사항)
        list_path = os.path.join(self.base_dir, f"krx_codes_{datetime.now().strftime('%Y%m%d')}.csv")
        stocks[['Code', 'Name']].to_csv(list_path, index=False, encoding='utf-8-sig')

        return stocks['Code'].tolist(), shares_map

    def process_stock(self, code, marcap_groups, shares_map):
        """개별 종목 처리 로직 (Marcap + FDR 병합)"""
        save_path = os.path.join(self.base_dir, f"{code}.parquet")

        if os.path.exists(save_path):
            return "Skip"

        try:
            # (A) Marcap 데이터 (메모리에서 조회)
            df_m = marcap_groups.get(code, pd.DataFrame()).reset_index()

            # (B) FDR 데이터 (2026년 1월 ~ 현재)
            # Marcap 데이터가 끝나는 시점 이후부터 긁어오는게 안전하지만, 여기선 고정된 날짜 사용
            df_f = fdr.DataReader(code, "2026-01-01", self.end_date)

            if not df_f.empty:
                df_f = df_f.reset_index()
                # 컬럼 계산
                df_f['Amount'] = df_f['Close'] * df_f['Volume']
                shares = shares_map.get(code, 0)
                df_f['Marcap'] = df_f['Close'] * (shares if pd.notna(shares) else 0)
                df_f['Code'] = code

                # 필수 컬럼 채우기
                cols_needed = ['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Marcap']
                for c in cols_needed:
                    if c not in df_f.columns: df_f[c] = 0
                df_f = df_f[cols_needed]

            # (C) 병합
            cols = ['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Marcap']

            if not df_m.empty:
                # Marcap 컬럼 정리
                for c in cols:
                    if c not in df_m.columns: df_m[c] = pd.NA

                if not df_f.empty:
                    # 겹치는 날짜 제거 (FDR이 최신)
                    last_m_date = df_m['Date'].max()
                    df_f = df_f[df_f['Date'] > last_m_date]
                    df_final = pd.concat([df_m[cols], df_f])
                else:
                    df_final = df_m[cols]
            else:
                df_final = df_f

            if df_final.empty:
                return "Empty"

            # (D) 최종 저장
            df_final = df_final.sort_values('Date').reset_index(drop=True)
            int_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Marcap']
            for c in int_cols:
                if c in df_final.columns:
                    df_final[c] = df_final[c].fillna(0).astype('Int64')

            df_final.to_parquet(save_path, index=False)
            return "Success"

        except Exception as e:
            return f"Error: {str(e)}"

    def run(self):
        """전체 프로세스 실행"""
        # 1. Marcap 로딩
        marcap_groups = self._load_marcap_memory()

        # 2. 타겟 리스트 확보
        target_codes, shares_map = self._get_target_list()

        logger.info(f"🔥 총 {len(target_codes)}개 종목 변환 시작...")

        # 3. 루프 실행
        results = []
        for code in tqdm(target_codes):
            res = self.process_stock(code, marcap_groups, shares_map)
            results.append(res)

        logger.info("작업 완료!")
        logger.info(f"성공: {results.count('Success')}, 스킵: {results.count('Skip')}")
        logger.info(f"데이터없음: {results.count('Empty')}, 에러: {sum(1 for r in results if r.startswith('Error'))}")


if __name__ == "__main__":
    # 이 파일을 직접 실행할 때 동작
    downloader = StockDownloader(base_dir="data/raw")
    downloader.run()
