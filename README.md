# agent-channel-cost

KSC 2026 연구 명세의 I/T/O 채널 회계 구현이다. 현재 범위는 외부 모델을
호출하지 않는 회계 코어와 fixture 기반 검증으로 제한한다.

```bash
python -m unittest discover -s tests -v
```

실제 실행 전에는 `configs/model.yaml`과 `configs/pricing.yaml`의 `null` 값을
공식 provider 문서에 근거해 채워야 한다. 미확정 값은 의도적으로 실행 가능
기본값을 제공하지 않는다.

