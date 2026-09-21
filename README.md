# agent-channel-cost

KSC 2026 연구 명세의 I/T/O 채널 회계와 실험 실행·분석 파이프라인이다.
실제 API 호출 전 단계는 오프라인 fixture와 scripted provider로 검증한다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[live,dev]'
.venv/bin/python -m pytest
```

실제 API 실행 전 프로젝트 루트의 `.env`에 키를 입력한다. 따옴표는 필요 없다.
셸에 이미 같은 환경변수가 있으면 셸 값이 `.env`보다 우선한다.

```dotenv
OPENAI_API_KEY=sk-여기에_본인_키
```

`.env`와 `.env.*`는 Git에서 제외되며 `.env.example`만 예시로 추적한다.

`configs/model.yaml`은 `gpt-5.6-terra`, `o200k_base`, `tiktoken==0.14.0`을
고정한다. `configs/pricing.yaml`은 2026-09-14에 확인한 공식 Standard·장문
요율을 고정하며, cached read와 cache write를 별도 과금 항목으로 계산한다.
Terra가 `temperature`를 거부하므로 요청에서는 해당 파라미터를 생략한다.

현재 확증 실험은 v2의 도구 관측(T-H) 압축이다. 공통 의미 구조에서 한국어·영어
surface를 함께 만들고, 첫 도구 결과를 읽어야 후속 호출을 완성하는 `long`과
`event` 과제만 사용한다. 도구 환경은 네트워크·현재 시각·전역 난수에 의존하지
않고 run마다 격리된다.

## 현재 준비 상태

- `data/v2/`는 개발 25쌍과 보류 75쌍을 분리하며, 보류 checksum은 고정돼 있다.
- 본 계획은 B0, 균등 추출, head-tail, schema-oracle, query-aware를 각 150회씩
  하나의 순서로 섞어 총 750회를 실행한다.
- `results/v2/heldout-main-2026-09-21.jsonl`은 누락·중복·checksum 불일치 없이
  완료된 본 실험 원자료다.
- `scripts/analyze_v2_main.py`는 의미쌍 군집 bootstrap 성공당 비용 CI, Holm 보정,
  정답 행동 필드 잔존 층화, protocol residual 분포를 재생성한다.
- schema-oracle과 query-aware는 제안 방법이 아니라 각각 상한과 이론적 비용
  하한이다. 초기 I-H는 실제 다중 턴 이력이 아니므로 확증 실험에서 제외했다.
- 개발 B0의 네 라운드와 중단 기록은 `docs/v2-experiment-decisions.md`에 보존한다.

모델 ID, 가격, tokenizer, protocol residual 허용치, realized-ratio 허용 오차는
모두 고정되어 있다. 계정의 모델 접근 권한과 실제 응답 schema는 크레딧을 받은
뒤 첫 smoke 호출에서만 최종 확인할 수 있다.

```bash
.venv/bin/python scripts/audit_v2_candidates.py
.venv/bin/python scripts/run_v2_experiment.py preflight
.venv/bin/python scripts/run_v2_experiment.py heldout-main \
  --output results/v2/heldout-main.jsonl --projected-cost-per-run 0.03
.venv/bin/python scripts/analyze_v2_main.py \
  results/v2/heldout-main-2026-09-21.jsonl \
  --output results/v2/heldout-main-summary-2026-09-21.json \
  --csv results/v2/heldout-method-summary-2026-09-21.csv \
  --retention-csv results/v2/information-retention-2026-09-21.csv \
  --residual-csv results/v2/protocol-residual-summary-2026-09-21.csv
```

유료 실행은 `heldout-main`과 출력·예상 비용을 명시해야 한다. 중단된 실행은
같은 명령에 `--resume`을 붙이면 기존 ordinal과 계획의 일치를 확인한 뒤
이어간다. v1의 I-H/T-H/TCPC 스크립트와 결과는 탐색적 재현 자료로 남아 있다.

가격 출처: <https://developers.openai.com/api/docs/pricing>

모델 출처: <https://developers.openai.com/api/docs/models/gpt-5.6-terra>
