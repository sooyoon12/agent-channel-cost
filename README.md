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

현재 태스크 생성기는 공통 의미 구조에서 한국어·영어 surface를 함께 만들며,
`generate_preview()`는 데이터 동결 전 검토용으로 유형별 3쌍만 반환한다.
도구 환경은 네트워크·현재 시각·전역 난수에 의존하지 않고 run마다 격리된다.

## 현재 준비 상태

- `data/manifest.json`은 40개 의미쌍과 한·영 80개 인스턴스의 checksum을 고정한다.
- `build_main_plan()`은 B0 240회와 나머지 조건 각 80회, 총 480회를 만든다.
- `build_smoke_plan()`은 최대 20회만 만든다.
- 실제 provider는 preflight 통과와 명시적 live 허용 전까지 runtime에서 차단된다.
- provider 장애 재시도는 연구 행동 재시도와 분리되며 기본 최대 1회다.
- `scripts/analyze_results.py`가 processed JSON/CSV, 실행 audit JSON과 필수 SVG 두 개를 만든다.

모델 ID, 가격, tokenizer, protocol residual 허용치, realized-ratio 허용 오차는
모두 고정되어 있다. 계정의 모델 접근 권한과 실제 응답 schema는 크레딧을 받은
뒤 첫 smoke 호출에서만 최종 확인할 수 있다.

```bash
.venv/bin/python scripts/verify_dataset.py
.venv/bin/python scripts/run_experiment.py preflight
.venv/bin/python scripts/check_smoke.py results/smoke/RUN.jsonl
.venv/bin/python scripts/analyze_results.py results/raw/runs.jsonl
```

유료 실행은 `run_experiment.py smoke|main --output ...
--projected-cost-per-run ...`처럼 명시해야 한다. 중단된 실행은 같은 명령에
`--resume`을 붙이면 기존 ordinal과 계획의 일치를 확인한 뒤 이어서 실행한다.

가격 출처: <https://developers.openai.com/api/docs/pricing>

모델 출처: <https://developers.openai.com/api/docs/models/gpt-5.6-terra>
