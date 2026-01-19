# CIO 중심 모듈형 퀀트 시스템

이 프로젝트는 **CIO가 고정 유니버스를 결정**하고, 시스템은 **팩터 스코어링 + 모듈형 리밸런싱**으로 대응형 자산배분을 수행하는 단일 구조입니다.

## 핵심 철학
- 유니버스는 CIO가 직접 입력하며, 시스템은 임의 필터링을 하지 않습니다.
- 예측 대신 대응: 개별 종목의 상태(팩터 점수)에 반응합니다.
- 입력 → 판단 → 실행이 한 흐름으로 연결된 단일 구조입니다.

## 구성 요소

### 1) 팩터 스코어링
- 모멘텀/변동성/밸류 등 검증된 팩터로 점수 산출
- 가중치 합산으로 최종 스코어 생성
- 점수가 낮으면 자동으로 현금/단기채로 이동

### 2) 리밸런싱 모듈화
- Trigger(언제): `time`(월말/분기말 등), `threshold`(드리프트 기반)
- Method(어떻게): `absolute_score`, `rank`, `risk_parity`

## 실행 방법

### 1) 설정 파일
- 유니버스: [config/universe.yaml](config/universe.yaml)
- 백테스트: [config/backtest.yaml](config/backtest.yaml)

### 2) 가격 데이터 주입
CSV 형식: `data/processed/prices.csv`
- 인덱스: 날짜
- 컬럼: 종목 심볼(유니버스와 동일)

### 3) 백테스트 실행
```bash
python research/cli.py --start 2015-01-01 --end 2024-12-31
```

결과는 `data/results/metrics.csv`, `data/results/equity_curve.csv`로 저장됩니다.

## 의존성
`requirements.txt` 참고
