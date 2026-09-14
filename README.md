# agent-channel-cost

KSC 2026 연구 명세의 I/T/O 채널 회계 구현이다. 현재 범위는 외부 모델을
호출하지 않는 회계 코어와 fixture 기반 검증으로 제한한다.

```bash
python -m unittest discover -s tests -v
```

실제 실행 전에는 `configs/model.yaml`과 `configs/pricing.yaml`의 `null` 값을
공식 provider 문서에 근거해 채워야 한다. 미확정 값은 의도적으로 실행 가능
기본값을 제공하지 않는다.

현재 태스크 생성기는 공통 의미 구조에서 한국어·영어 surface를 함께 만들며,
`generate_preview()`는 데이터 동결 전 검토용으로 유형별 3쌍만 반환한다.
도구 환경은 네트워크·현재 시각·전역 난수에 의존하지 않고 run마다 격리된다.

## 현재 준비 상태

- `data/manifest.json`은 40개 의미쌍과 한·영 80개 인스턴스의 checksum을 고정한다.
- `build_main_plan()`은 B0 240회와 나머지 조건 각 80회, 총 480회를 만든다.
- `build_smoke_plan()`은 최대 20회만 만든다.
- 실제 provider는 preflight 통과와 명시적 live 허용 전까지 runtime에서 차단된다.
- provider 장애 재시도는 연구 행동 재시도와 분리되며 기본 최대 1회다.
- `scripts/analyze_results.py`가 processed JSON/CSV와 필수 SVG 두 개를 만든다.

아직 공식 확인이 필요한 값은 model ID, 가격표와 버전, tokenizer encoding,
protocol residual 허용치, realized-ratio 허용 오차다. 이 값이 비어 있으면 live
preflight가 실패하는 것이 정상이다.

```bash
python3 scripts/verify_dataset.py
python3 scripts/run_experiment.py preflight
python3 scripts/check_smoke.py results/smoke/RUN.jsonl
python3 scripts/analyze_results.py results/raw/runs.jsonl
```

유료 실행은 `run_experiment.py smoke|main --output ...
--projected-cost-per-run ...`처럼 명시해야 한다. 중단된 실행은 같은 명령에
`--resume`을 붙이면 기존 ordinal과 계획의 일치를 확인한 뒤 이어서 실행한다.
