<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cli-modelarium-wordmark-dark.svg">
  <img alt="cli modelarium" src="docs/assets/cli-modelarium-wordmark-light.svg" width="420">
</picture>

用其他语言阅读: [English](README.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [한국어](README.ko.md) | [Deutsch](README.de.md) | [Português](README.pt.md) | [Italiano](README.it.md)

注意: 此 README 是为了可访问性而翻译的。Cli Modelarium CLI 工具本身仅输出英文。无论系统区域设置如何，所有命令、错误消息和输出均保持英文。

> Note: Features added after v0.1.0 (`--runs` in v0.1.1, statistical significance in v0.1.2, confidence intervals/paired tests/McNemar in v0.1.3) are documented in English only — translations pending.

> 在终端中并排比较 LLM 输出 - 8 个云服务提供商 + 本地模型，支持并行流式传输、批量评估、LLM-as-judge 评分、幻觉检测和 CI/CD 就绪的断言。

[![CI](https://github.com/lavellehatcherjr/Cli-Modelarium/actions/workflows/ci.yml/badge.svg)](https://github.com/lavellehatcherjr/Cli-Modelarium/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cli-modelarium)](https://pypi.org/project/cli-modelarium/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Platforms](https://img.shields.io/badge/platforms-Mac%20%7C%20Windows%20%7C%20Linux-lightgrey)](#)

<p align="center">
  <img src="docs/assets/cli-modelarium-demo.png" alt="Cli Modelarium help output showing the banner and available commands" width="520">
</p>

## 功能简介

**Cli Modelarium** 是一款精心打造的命令行工具，用于跨提供商、模型、系统提示和温度参数比较 LLM 输出 - 内置实时并行流式传输、批量评估、确定性测试和质量评分。

适用于评估哪个模型适合您的特定任务、在 CI/CD 中运行提示回归测试、将本地模型与云 API 进行比较，或构建评估数据集 - 一切都通过单个终端命令完成。

## 快速开始

```bash
pip install cli-modelarium

# 配置 API 密钥（安全保存到您的操作系统密钥链中）
cli-modelarium configure

# 运行您的第一次比较
cli-modelarium "Explain quantum computing in one sentence" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro \
  --temperatures 0,0.7
```

就这么简单。您将看到三个模型并行实时流式传输响应，延迟、令牌数和成本显示在简洁的比较表中。

## 特性

### 🤖 提供商（8 个云端 + 无限本地）

- **云服务提供商:** OpenAI、Anthropic、Google (Gemini)、xAI (Grok)、DeepSeek、Mistral、Groq、OpenRouter
- **本地模型:** Ollama、LM Studio、vLLM、llama.cpp - 任何 OpenAI 兼容的本地服务器
- 在同一比较中混合使用本地和云模型
- 每次调用可配置的模型选择（无硬编码列表）

### ⚡ 并行流式传输

- 同时跨所有模型逐令牌实时显示
- 每个模型的 Time-to-First-Token (TTFT) 跟踪
- 查看哪个模型首先完成，实时观察输出分歧
- 来自所有 8 个提供商的流（底层使用 SSE）

### 📊 多种比较模式

- **单一提示 vs. 多个模型** - 快速"哪个最好？"比较
- **单一提示 vs. 多个温度** - 查看随机性如何影响输出
- **多个系统提示 vs. 一个用户提示** - A/B 测试提示工程
- **批量模式** - 用于实际评估工作的多提示 × 多模型
- **本地 vs. 云比较** - 量化差距（或其缺失）

### 🧪 评估功能

- **确定性断言** - 10 种断言类型（`contains`、`regex`、`json_valid`、`json_schema`、`max_length_chars`、`latency_under`、`cost_under` 等），具有通过/失败输出和 CI 退出代码
- **LLM-as-a-judge 评分** - 使用一个 LLM 根据质量标准对其他 LLM 的输出进行评分
- **评判面板** - 多个评判平均得分以减少偏见的评估
- **幻觉检测预设** - 用于事实准确性检查的开箱即用标准
- **自定义标准** - 定义您自己的评分规则
- **自评自动跳过** - 当评判模型也是被评判对象时自动跳过

### 💾 输出格式

- **实时终端** - 基于 Rich 的面板，带有进度条和流式显示
- **CSV** - 电子表格友好（在 Excel、Google Sheets、pandas 中打开）
- **JSON** - 为脚本和管道结构化
- **Markdown** - 用于博客文章和报告的精美表格
- **退出代码** - 反映 CI/CD 通过/失败状态的 0/1/2

### 💰 成本透明度

- 从每个提供商报告的使用情况显示每次调用成本
- 每次比较的总成本汇总
- 启用 LLM-as-judge 时单独显示评判成本
- 本地模型显示为 "Free"
- 通过 `--max-cost` 标志防止意外账单

### 🔒 安全性

- 通过 `keyring` 将 API 密钥存储在 OS 原生密钥链中（Mac Keychain、Windows Credential Manager、Linux Secret Service）
- 格式验证在存储前捕获粘贴错误
- 错误消息编辑防止密钥在回溯中泄漏
- 仅限 localhost 的本地模型 URL 验证
- 包含负责任披露政策的 `SECURITY.md`

### 🛡️ 速率限制处理

- 每个提供商的并发限制（默认 5）尊重所有层级基线
- 自动 429 重试，带指数退避
- Anthropic 的 529 "overloaded" 与速率限制分开处理
- 为较高层级的高级用户提供 `--concurrency` 标志
- 每个模型的优雅失败（其他模型继续）

### 🌐 跨平台

- 在 macOS、Windows（10+ 和 ARM）和 Linux 上以相同方式工作
- 所有文件 I/O 使用 `pathlib` + 显式 UTF-8 编码
- CSV 写入使用 `newline=""` 以兼容 Windows
- 需要 Python 3.11+

### 📋 开发者体验

- **单一 CLI 二进制文件** - `pip install cli-modelarium` 即可完成
- **精致的基于 Rich 的 UI** - Claude Code 级别的终端打磨
- **JSON 输出** - 可通过管道输入任何工具（`jq`、脚本、监控）
- **CI/CD 就绪** - 退出代码、结构化输出、包含 GitHub Actions 示例
- **Apache 2.0 许可** - 可用于任何项目，商业或其他

## 示例

### 在编码任务上比较 3 个模型

```bash
cli-modelarium "Write a Python function to find the longest palindromic substring" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro
```

### 带断言的批量评估

创建 `eval.json`:

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

运行它:

```bash
cli-modelarium batch eval.json \
  --models gpt-5.5,claude-opus-4-7 \
  --output results.csv
```

### 使用 LLM 评判对输出进行评分

```bash
cli-modelarium "Explain recursion in one paragraph" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro,local/llama-3.3-70b \
  --judge claude-opus-4-7 \
  --judge-criteria "accuracy,clarity,brevity"
```

### 针对已知事实检测幻觉

```bash
cli-modelarium "Tell me about the Eiffel Tower" \
  --models gpt-5.5,claude-opus-4-7 \
  --judge claude-opus-4-7 \
  --check-hallucination \
  --expected-facts "Built 1887-1889,Located in Paris France,Designed by Gustave Eiffel"
```

### 将本地模型与云 API 进行比较

```bash
# 首先启动 Ollama: ollama run llama3.3
cli-modelarium "Summarize the key features of microservices architecture" \
  --models local/llama-3.3-70b,gpt-5.5,claude-opus-4-7
```

### 在 CI/CD 中运行（GitHub Actions 示例）

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

如果通过率低于 90%，命令将以代码 1 退出，从而使构建失败。

## 配置

### API 密钥

Cli Modelarium 将 API 密钥存储在您的 OS 原生密钥链中（Mac Keychain、Windows Credential Manager 或通过 `keyring` 的 Linux Secret Service）。密钥永远不会以明文形式写入磁盘。

```bash
# 交互式设置（推荐）
cli-modelarium configure

# 或单独设置
cli-modelarium keys set openai
cli-modelarium keys set anthropic
cli-modelarium keys set google

# 检查配置了哪些密钥
cli-modelarium keys list

# 删除密钥
cli-modelarium keys delete openai
```

您也可以使用环境变量（对 CI/CD 有用）:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

环境变量优先于密钥链存储。

### 本地模型（Ollama、LM Studio 等）

本地模型通过 OpenAI 兼容端点工作 - 无需 API 密钥。该工具自动检测默认的 Ollama 端口。

```bash
# 默认: 假定 Ollama 在 localhost:11434
cli-modelarium "test" --models local/llama-3.3

# 改用 LM Studio
cli-modelarium "test" --models local/qwen-3-32b --local-url http://localhost:1234/v1

# 将自定义本地 URL 保存为默认值
cli-modelarium keys set local --base-url http://localhost:1234/v1
```

## 支持的提供商

| 提供商 | 需要 API 密钥 | 流式传输 | 成本跟踪 |
|----------|-----------------|-----------|---------------|
| OpenAI (GPT-5, GPT-5 mini, o3, o4-mini 等) | ✅ | ✅ | ✅ |
| Anthropic (Claude Opus 4.7, Sonnet 4.6, Haiku 4.5 等) | ✅ | ✅ | ✅ |
| Google (Gemini 3.1 Pro, Gemini 3 Flash 等) | ✅ | ✅ | ✅ |
| xAI (Grok 4.1 等) | ✅ | ✅ | ✅ |
| DeepSeek (V3, R1) | ✅ | ✅ | ✅ |
| Mistral (Large, Medium, Small) | ✅ | ✅ | ✅ |
| Groq (Llama, Mixtral 等) | ✅ | ✅ | ✅ |
| OpenRouter (平台上的任何模型) | ✅ | ✅ | ✅ |
| **本地: Ollama** | ❌ | ✅ | 免费 |
| **本地: LM Studio** | ❌ | ✅ | 免费 |
| **本地: vLLM** | ❌ | ✅ | 免费 |
| **本地: llama.cpp server** | ❌ | ✅ | 免费 |

运行 `cli-modelarium list-models` 查看所有当前支持的模型。

## 工作原理

Cli Modelarium 使用模块化的提供商抽象层，隐藏了 OpenAI 的 `messages` 数组、Anthropic 的顶级 `system` 参数、Google 的 `system_instruction` 以及其他 API 之间的差异。每个提供商都实现了相同的异步流式接口，因此 CLI 可以使用 `asyncio.gather()` 并行运行它们。

成本计算来自每个提供商报告的 `usage` 字段（输入令牌、输出令牌、缓存令牌）乘以当前定价常数。定价数据于 **2026 年 5 月 25 日** 从官方提供商文档中验证 - 详细注意事项请参阅 [注意事项与限制](#注意事项与限制)。

对于本地模型，使用相同的 OpenAI Python SDK 加上自定义 `base_url`，因为 Ollama、LM Studio、vLLM 和 llama.cpp 都暴露了 OpenAI 兼容的 REST 端点。

## 注意事项与限制

### 定价数据

Cli Modelarium 内置的所有定价均于 **2026 年 5 月 25 日** 从官方提供商文档中验证。LLM 定价经常变化（有时每月一次）。该工具在每个输出中显示 `pricing_as_of` 日期。在依赖成本计算进行预算或生产决策之前，请始终对照每个提供商的官方定价页面进行验证。

### 速率限制

速率限制处理和默认的每个提供商的并发设置基于 **2026 年 5 月 25 日** 验证的提供商速率限制。您的特定层级的限制可能与此处假定的默认值不同。在构建生产容量假设之前，请对照提供商的官方仪表板验证您当前的限制。

### 模型可用性

Cli Modelarium 支持的模型反映了 **2026 年 5 月 25 日** 提供商提供的内容。提供商会定期发布新模型、弃用旧模型并调整能力。如果注册表中的模型不再工作，请运行 `cli-modelarium list-models` 并查看提供商的文档。

### 不是生产级网关

Cli Modelarium 是为评估和比较而设计的 - 从开发者终端跨提供商运行临时并排测试。它不是生产推理网关。如果您需要生产规模的路由、负载均衡、回退链或 SLA 管理的推理，请寻找专门为此目的构建的工具。

### 跨提供商的令牌计数比较

结果中显示的令牌计数由每个提供商的 API 报告。不同的提供商使用不同的分词器，因此"输出令牌"在相同文本下不能直接跨提供商比较。如果您要比较生产使用的成本效率，请在实际工作负载中运行真实提示 - 不要仅依赖跨提供商的每令牌数学计算。

### LLM-as-a-Judge 使用

Cli Modelarium 包含可选的 LLM-as-a-judge 评分（通过 `--judge` 标志启用），它使用一个 LLM 来评估其他 LLM 的输出。这是标准的基准测试方法，并且在所有支持的提供商的服务条款下作为评估/基准测试活动是被允许的。

使用 `--judge` 时，您有责任遵守您使用其模型的每个提供商的服务条款。每个提供商的 ToS 同时适用于被评判的模型和评判模型本身。

**评判偏见提示:** LLM 评判有已记录的偏见（自我偏好、同家族偏好、冗长偏好）。评判分数是有用的信号，而不是基本事实。使用评判面板（带多个模型的 `--judges`）来减少偏见。

### 幻觉检测

幻觉检测预设是模型之间有用的比较信号，而不是基本事实验证。检测准确性取决于使用的评判模型、所需的领域知识以及是否通过 `--expected-facts` 提供参考事实。将其用于相对质量比较，而不是绝对正确性验证。

### 比较方法论

LLM 在温度 > 0 时是非确定性的 - 重新运行相同的提示可能产生不同的输出。单次比较运行向您显示每个模型的一个样本，而不是最终的质量判决。

要得出更可靠的结论:
- 使用 `--temperatures 0` 获得更确定性的输出（在支持的地方）
- 运行相同的比较 3-5 次并查找模式
- 跨多个提示比较，而不仅仅是一个
- 使用 `--output json` 标志保存运行结果以进行系统分析

## 关于作者

Cli Modelarium 由 **[Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)** 构建。

### 联系

- 💼 LinkedIn: [linkedin.com/in/lavellehatcherjr](https://linkedin.com/in/lavellehatcherjr)
- 🐙 GitHub: [github.com/lavellehatcherjr](https://github.com/lavellehatcherjr)
- 💬 关于此项目的问题: [打开 issue](../../issues)
- 📩 合作/机会: 通过 LinkedIn 联系

## 为什么构建它

跨提供商比较 LLM 输出很繁琐 - 不同的 SDK、不同的认证模式、不同的响应形状，没有简单的方法可以并排查看它们以及成本和延迟数据。精致的云游乐场一次只显示一个提供商，可用的开源选项要么专注于生产路由，要么是为团队优化的完整评估平台。

Cli Modelarium 是一个专注的小型 CLI 工具，专门做好一件事: 带有质量评分、断言、批量模式和流式传输的并排比较 - 一切都为终端优先的开发者工作流程设计。

它是有意聚焦的: 没有生产路由、没有代理编排、没有微调、没有 GUI。只有来自命令行的清洁、快速的比较。

通过模块化的提供商抽象、并行执行、透明的成本计算和通过 OS 密钥链系统为本地用户提供的安全密钥存储构建。

## 贡献

欢迎 issues 和 PR。请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

对于安全问题，请参阅 [SECURITY.md](SECURITY.md) - 请勿为安全问题提交公开 issue。

## 许可证

依据 [Apache License, Version 2.0](LICENSE) 授权。

请参阅 [NOTICE](NOTICE) 文件了解归属要求。

---

由 [Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr) 构建

依据 Apache 2.0 授权。欢迎 issues、PR 和对话。
