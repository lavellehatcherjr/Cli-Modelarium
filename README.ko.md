<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cli-modelarium-wordmark-dark.svg">
  <img alt="cli modelarium" src="docs/assets/cli-modelarium-wordmark-light.svg" width="420">
</picture>

다른 언어로 읽기: [English](README.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [中文](README.zh.md) | [Deutsch](README.de.md) | [Português](README.pt.md) | [Italiano](README.it.md)

참고: 이 README는 접근성을 위해 번역되었습니다. Cli Modelarium CLI 도구 자체는 영어로만 출력됩니다. 모든 명령, 오류 메시지 및 출력은 시스템 로케일에 관계없이 영어로 유지됩니다.

> Note: Features added after v0.1.0 (`--runs` in v0.1.1, statistical significance in v0.1.2, confidence intervals/paired tests/McNemar in v0.1.3) are documented in English only — translations pending.

> 터미널에서 LLM 출력을 나란히 비교 - 8개 클라우드 제공자 + 로컬 모델, 병렬 스트리밍, 배치 평가, LLM-as-judge 스코어링, 환각 감지 및 CI/CD 지원 어설션 포함.

[![CI](https://github.com/lavellehatcherjr/Cli-Modelarium/actions/workflows/ci.yml/badge.svg)](https://github.com/lavellehatcherjr/Cli-Modelarium/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cli-modelarium)](https://pypi.org/project/cli-modelarium/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Platforms](https://img.shields.io/badge/platforms-Mac%20%7C%20Windows%20%7C%20Linux-lightgrey)](#)

## 기능 개요

**Cli Modelarium**은 제공자, 모델, 시스템 프롬프트 및 온도 전반에 걸쳐 LLM 출력을 비교하기 위한 세련된 명령줄 도구입니다 - 라이브 병렬 스트리밍, 배치 평가, 결정론적 테스트 및 품질 스코어링이 내장되어 있습니다.

특정 작업에 적합한 모델을 평가하거나, CI/CD에서 프롬프트 회귀 테스트를 실행하거나, 로컬 모델을 클라우드 API와 비교하거나, 평가 데이터셋을 구축하는 데 유용합니다 - 모두 단일 터미널 명령으로 가능합니다.

## 빠른 시작

```bash
pip install cli-modelarium

# API 키 구성 (OS 키체인에 안전하게 저장됨)
cli-modelarium configure

# 첫 번째 비교 실행
cli-modelarium "Explain quantum computing in one sentence" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro \
  --temperatures 0,0.7
```

그게 전부입니다. 세 모델 모두 응답을 병렬로 라이브 스트리밍하며, 지연 시간, 토큰 수 및 비용이 깔끔한 비교 테이블에 표시됩니다.

## 기능

### 🤖 제공자 (8개 클라우드 + 무제한 로컬)

- **클라우드 제공자:** OpenAI, Anthropic, Google (Gemini), xAI (Grok), DeepSeek, Mistral, Groq, OpenRouter
- **로컬 모델:** Ollama, LM Studio, vLLM, llama.cpp - 모든 OpenAI 호환 로컬 서버
- 동일한 비교에서 로컬 및 클라우드 모델 혼합 가능
- 호출별로 구성 가능한 모델 선택 (하드코딩된 목록 없음)

### ⚡ 병렬 스트리밍

- 모든 모델에서 동시에 토큰별 라이브 표시
- 모델당 Time-to-First-Token (TTFT) 추적
- 어떤 모델이 먼저 완료되는지 확인하고 출력이 실시간으로 분기되는 것을 관찰
- 8개 제공자 모두에서 스트림 (내부적으로 SSE)

### 📊 다양한 비교 모드

- **단일 프롬프트 vs. 여러 모델** - 빠른 "어떤 것이 가장 좋은가?" 비교
- **단일 프롬프트 vs. 여러 온도** - 무작위성이 출력에 어떻게 영향을 미치는지 확인
- **여러 시스템 프롬프트 vs. 하나의 사용자 프롬프트** - 프롬프트 엔지니어링 A/B 테스트
- **배치 모드** - 실제 평가 작업을 위한 멀티 프롬프트 × 멀티 모델
- **로컬 vs. 클라우드 비교** - 격차(또는 그 부재)를 정량화

### 🧪 평가 기능

- **결정론적 어설션** - 10가지 어설션 유형 (`contains`, `regex`, `json_valid`, `json_schema`, `max_length_chars`, `latency_under`, `cost_under` 등), 통과/실패 출력 및 CI 종료 코드 포함
- **LLM-as-a-judge 스코어링** - 한 LLM을 사용하여 품질 기준에 따라 다른 LLM의 출력 점수 매김
- **저지 패널** - 여러 저지가 점수를 평균화하여 덜 편향된 평가 제공
- **환각 감지 프리셋** - 사실 정확성 검사를 위해 즉시 사용 가능한 기준
- **사용자 정의 기준** - 자신만의 스코어링 루브릭 정의
- **자기 평가 자동 건너뛰기** - 저지 모델이 평가 대상이기도 할 때 자동으로 건너뜀

### 💾 출력 형식

- **라이브 터미널** - 진행 바 및 스트리밍 표시가 있는 Rich 기반 패널
- **CSV** - 스프레드시트 친화적 (Excel, Google Sheets, pandas에서 열기)
- **JSON** - 스크립트 및 파이프라인을 위한 구조화
- **Markdown** - 블로그 게시물 및 보고서를 위한 깔끔한 테이블
- **종료 코드** - CI/CD를 위해 통과/실패 상태를 반영하는 0/1/2

### 💰 비용 투명성

- 각 제공자의 보고된 사용량으로부터 호출별 비용 표시
- 비교당 총 비용 요약
- LLM-as-judge가 활성화될 때 저지 비용 별도 표시
- 로컬 모델은 "Free"로 표시
- 예상치 못한 청구를 방지하기 위한 `--max-cost` 플래그

### 🔒 보안

- API 키는 `keyring`을 통해 OS 네이티브 키체인에 저장됨 (Mac Keychain, Windows Credential Manager, Linux Secret Service)
- 형식 검증으로 저장 전에 붙여넣기 오류 포착
- 오류 메시지 수정으로 트레이스백에서 키 누출 방지
- 로컬 모델 URL에 대한 localhost 전용 검증
- 책임 있는 공개 정책이 포함된 `SECURITY.md`

### 🛡️ 속도 제한 처리

- 제공자별 동시성 제한 (기본 5)으로 모든 티어 기준선 준수
- 지수 백오프를 사용한 자동 429 재시도
- Anthropic의 529 "overloaded"는 속도 제한과 별도로 처리됨
- 상위 티어의 파워 유저를 위한 `--concurrency` 플래그
- 모델별 우아한 실패 처리 (다른 모델은 계속 진행)

### 🌐 크로스 플랫폼

- macOS, Windows (10+ 및 ARM), Linux에서 동일하게 작동
- 모든 파일 I/O는 `pathlib` + 명시적 UTF-8 인코딩 사용
- CSV 쓰기는 Windows 호환성을 위해 `newline=""` 사용
- Python 3.11+ 필요

### 📋 개발자 경험

- **단일 CLI 바이너리** - `pip install cli-modelarium`으로 완료
- **세련된 Rich 기반 UI** - Claude Code 수준의 터미널 마무리
- **JSON 출력** - 무엇이든 파이프 (`jq`, 스크립트, 모니터링)
- **CI/CD 준비 완료** - 종료 코드, 구조화된 출력, GitHub Actions 예제 포함
- **Apache 2.0 라이선스** - 상업적이든 아니든 모든 프로젝트에서 사용

## 예제

### 코딩 작업에서 3개 모델 비교

```bash
cli-modelarium "Write a Python function to find the longest palindromic substring" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro
```

### 어설션을 사용한 배치 평가

`eval.json` 생성:

```json
[
  {
    "id": "math-1",
    "prompt": "What is 2 + 2?",
    "assertions": [
      {"type": "contains", "value": "4"},
      {"type": "max_length_chars", "value": 100}
    ]
  },
  {
    "id": "json-1",
    "prompt": "List 3 colors in JSON array format",
    "assertions": [
      {"type": "json_valid"}
    ]
  }
]
```

실행:

```bash
cli-modelarium batch eval.json \
  --models gpt-5.5,claude-opus-4-7 \
  --output results.csv
```

### LLM 저지로 출력 점수 매기기

```bash
cli-modelarium "Explain recursion in one paragraph" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro,local/llama-3.3-70b \
  --judge claude-opus-4-7 \
  --judge-criteria "accuracy,clarity,brevity"
```

### 알려진 사실에 대한 환각 감지

```bash
cli-modelarium "Tell me about the Eiffel Tower" \
  --models gpt-5.5,claude-opus-4-7 \
  --judge claude-opus-4-7 \
  --check-hallucination \
  --expected-facts "Built 1887-1889,Located in Paris France,Designed by Gustave Eiffel"
```

### 로컬 모델과 클라우드 API 비교

```bash
# 먼저 Ollama 시작: ollama run llama3.3
cli-modelarium "Summarize the key features of microservices architecture" \
  --models local/llama-3.3-70b,gpt-5.5,claude-opus-4-7
```

### CI/CD에서 실행 (GitHub Actions 예제)

```yaml
- name: Run LLM evaluation
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    cli-modelarium batch ./eval/test_suite.json \
      --models gpt-5.5,claude-opus-4-7 \
      --output eval_results.json \
      --min-pass-rate 0.90
```

통과율이 90% 미만으로 떨어지면 명령이 종료 코드 1로 종료되어 빌드가 실패합니다.

## 구성

### API 키

Cli Modelarium은 API 키를 OS 네이티브 키체인 (Mac Keychain, Windows Credential Manager 또는 `keyring`을 통한 Linux Secret Service)에 저장합니다. 키는 절대로 디스크에 평문으로 저장되지 않습니다.

```bash
# 대화형 설정 (권장)
cli-modelarium configure

# 또는 개별적으로 설정
cli-modelarium keys set openai
cli-modelarium keys set anthropic
cli-modelarium keys set google

# 어떤 키가 구성되었는지 확인
cli-modelarium keys list

# 키 제거
cli-modelarium keys delete openai
```

환경 변수도 사용할 수 있습니다 (CI/CD에 유용):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

환경 변수는 키체인 저장보다 우선합니다.

### 로컬 모델 (Ollama, LM Studio 등)

로컬 모델은 OpenAI 호환 엔드포인트를 통해 작동합니다 - API 키가 필요하지 않습니다. 도구는 Ollama의 기본 포트를 자동으로 감지합니다.

```bash
# 기본값: Ollama가 localhost:11434에 있다고 가정
cli-modelarium "test" --models local/llama-3.3

# 대신 LM Studio 사용
cli-modelarium "test" --models local/qwen-3-32b --local-url http://localhost:1234/v1

# 사용자 정의 로컬 URL을 기본값으로 저장
cli-modelarium keys set local --base-url http://localhost:1234/v1
```

## 지원되는 제공자

| 제공자 | API 키 필요 | 스트리밍 | 비용 추적 |
|----------|-----------------|-----------|---------------|
| OpenAI (GPT-5, GPT-5 mini, o3, o4-mini 등) | ✅ | ✅ | ✅ |
| Anthropic (Claude Opus 4.7, Sonnet 4.6, Haiku 4.5 등) | ✅ | ✅ | ✅ |
| Google (Gemini 3.1 Pro, Gemini 3 Flash 등) | ✅ | ✅ | ✅ |
| xAI (Grok 4.1 등) | ✅ | ✅ | ✅ |
| DeepSeek (V3, R1) | ✅ | ✅ | ✅ |
| Mistral (Large, Medium, Small) | ✅ | ✅ | ✅ |
| Groq (Llama, Mixtral 등) | ✅ | ✅ | ✅ |
| OpenRouter (플랫폼의 모든 모델) | ✅ | ✅ | ✅ |
| **로컬: Ollama** | ❌ | ✅ | 무료 |
| **로컬: LM Studio** | ❌ | ✅ | 무료 |
| **로컬: vLLM** | ❌ | ✅ | 무료 |
| **로컬: llama.cpp server** | ❌ | ✅ | 무료 |

현재 지원되는 모든 모델을 보려면 `cli-modelarium list-models`를 실행하십시오.

## 작동 방식

Cli Modelarium은 OpenAI의 `messages` 배열, Anthropic의 최상위 `system` 매개변수, Google의 `system_instruction` 등 API 간의 차이를 숨기는 모듈식 제공자 추상화 계층을 사용합니다. 모든 제공자가 동일한 비동기 스트리밍 인터페이스를 구현하므로 CLI는 `asyncio.gather()`로 모든 제공자를 병렬로 실행할 수 있습니다.

비용 계산은 각 제공자의 보고된 `usage` 필드(입력 토큰, 출력 토큰, 캐시된 토큰)에 현재 가격 상수를 곱하여 산출됩니다. 가격 데이터는 **2026년 5월 25일**에 공식 제공자 문서에서 검증되었습니다 - 주의사항은 [참고사항 및 제한사항](#참고사항-및-제한사항)을 참조하십시오.

로컬 모델의 경우 Ollama, LM Studio, vLLM 및 llama.cpp 모두 OpenAI 호환 REST 엔드포인트를 노출하므로 사용자 정의 `base_url`과 함께 동일한 OpenAI Python SDK가 사용됩니다.

## 참고사항 및 제한사항

### 가격 데이터

Cli Modelarium에 포함된 모든 가격은 **2026년 5월 25일**에 공식 제공자 문서에서 검증되었습니다. LLM 가격은 자주 변경됩니다(때로는 매월). 도구는 모든 출력에 `pricing_as_of` 날짜를 표시합니다. 예산 책정이나 프로덕션 결정을 위해 비용 계산에 의존하기 전에 항상 각 제공자의 공식 가격 페이지와 대조하여 확인하십시오.

### 속도 제한

속도 제한 처리 및 제공자별 기본 동시성 설정은 **2026년 5월 25일**에 검증된 제공자 속도 제한을 기반으로 합니다. 특정 티어의 제한은 여기에 가정된 기본값과 다를 수 있습니다. 프로덕션 용량 가정을 구축하기 전에 제공자의 공식 대시보드에서 현재 제한을 확인하십시오.

### 모델 가용성

Cli Modelarium에서 지원하는 모델은 **2026년 5월 25일**에 제공자가 제공한 것을 반영합니다. 제공자는 정기적으로 새 모델을 출시하고 오래된 모델을 폐기하며 기능을 조정합니다. 레지스트리의 모델이 더 이상 작동하지 않으면 `cli-modelarium list-models`를 실행하고 제공자의 문서를 확인하십시오.

### 프로덕션 등급 게이트웨이가 아님

Cli Modelarium은 평가 및 비교용으로 설계되었습니다 - 개발자 터미널에서 제공자 간에 임시 나란히 비교 테스트를 실행합니다. 프로덕션 추론 게이트웨이가 아닙니다. 프로덕션 규모의 라우팅, 로드 밸런싱, 폴백 체인 또는 SLA 관리 추론이 필요한 경우 해당 목적을 위해 특별히 구축된 도구를 찾으십시오.

### 제공자 간 토큰 수 비교

결과에 표시되는 토큰 수는 각 제공자의 API에서 보고됩니다. 다른 제공자는 다른 토크나이저를 사용하므로 동일한 텍스트에 대해 "출력 토큰"은 제공자 간에 직접 비교할 수 없습니다. 프로덕션 사용을 위한 비용 효율성을 비교하는 경우 실제 워크로드에서 실제 프롬프트를 실행하십시오 - 제공자 간 토큰당 계산에만 의존하지 마십시오.

### LLM-as-a-Judge 사용

Cli Modelarium에는 `--judge` 플래그로 활성화되는 선택적 LLM-as-a-judge 스코어링이 포함되어 있으며, 이는 한 LLM을 사용하여 다른 LLM의 출력을 평가합니다. 이는 표준 벤치마킹 방법론이며 지원되는 모든 제공자의 서비스 약관에 따라 평가/벤치마킹 활동으로 허용됩니다.

`--judge`를 사용할 때 사용자는 사용하는 각 제공자의 모델의 서비스 약관을 준수할 책임이 있습니다. 각 제공자의 ToS는 평가되는 모델과 저지 모델 자체 모두에 적용됩니다.

**저지 편향 공지:** LLM 저지는 문서화된 편향(자기 선호, 동일 패밀리 선호, 장황함 선호)을 가지고 있습니다. 저지 점수는 유용한 신호이지 절대적 진실이 아닙니다. 편향을 줄이려면 저지 패널(여러 모델을 사용한 `--judges`)을 사용하십시오.

### 환각 감지

환각 감지 프리셋은 모델 간의 유용한 비교 신호이지 절대적 진실 검증이 아닙니다. 감지 정확도는 사용된 저지 모델, 필요한 도메인 지식 및 `--expected-facts`를 통해 참조 사실이 제공되는지 여부에 따라 달라집니다. 절대적 정확성 검증이 아닌 상대적 품질 비교에 사용하십시오.

### 비교 방법론

LLM은 온도 > 0에서 비결정론적입니다 - 동일한 프롬프트를 다시 실행하면 다른 출력이 생성될 수 있습니다. 단일 비교 실행은 각 모델에서 하나의 샘플을 보여줄 뿐이지 결정적인 품질 판정이 아닙니다.

더 신뢰할 수 있는 결론을 도출하려면:
- 더 결정론적인 출력을 위해 `--temperatures 0` 사용 (지원되는 경우)
- 동일한 비교를 3-5회 실행하고 패턴을 찾기
- 하나가 아닌 여러 프롬프트에 걸쳐 비교
- 체계적 분석을 위해 실행 결과를 저장하려면 `--output json` 플래그 사용

## 저자 소개

Cli Modelarium은 **[Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)**가 제작했습니다.

### 연결

- 💼 LinkedIn: [linkedin.com/in/lavellehatcherjr](https://linkedin.com/in/lavellehatcherjr)
- 🐙 GitHub: [github.com/lavellehatcherjr](https://github.com/lavellehatcherjr)
- 💬 이 프로젝트에 대한 질문: [이슈 열기](../../issues)
- 📩 협업/기회: LinkedIn을 통해 연락

## 제작 이유

제공자 간에 LLM 출력을 비교하는 것은 번거롭습니다 - 다른 SDK, 다른 인증 패턴, 다른 응답 형태, 비용 및 지연 시간 데이터와 함께 나란히 볼 수 있는 쉬운 방법이 없습니다. 세련된 클라우드 플레이그라운드는 한 번에 한 제공자만 표시하며, 사용 가능한 오픈 소스 옵션은 프로덕션 라우팅에 집중하거나 팀에 최적화된 본격적인 평가 플랫폼입니다.

Cli Modelarium은 한 가지를 잘 수행하는 작고 집중된 CLI 도구입니다: 품질 스코어링, 어설션, 배치 모드 및 스트리밍을 통한 나란히 비교 - 모두 터미널 우선 개발자 워크플로우를 위해 설계되었습니다.

의도적으로 집중되어 있습니다: 프로덕션 라우팅 없음, 에이전트 오케스트레이션 없음, 파인 튜닝 없음, GUI 없음. 명령줄에서 깔끔하고 빠른 비교만.

모듈식 제공자 추상화, 병렬 실행, 투명한 비용 계산, 로컬 사용자를 위한 OS 키체인 시스템을 통한 안전한 키 저장으로 구축되었습니다.

## 기여

이슈와 PR을 환영합니다. 지침은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하십시오.

보안 문제는 [SECURITY.md](SECURITY.md)를 참조하십시오 - 보안 우려사항에 대해 공개 이슈를 제출하지 마십시오.

## 라이선스

[Apache License, Version 2.0](LICENSE) 하에 라이선스가 부여됩니다.

귀속 요구사항은 [NOTICE](NOTICE) 파일을 참조하십시오.

---

[Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)가 제작

Apache 2.0 라이선스 하에 있습니다. 이슈, PR 및 대화를 환영합니다.
