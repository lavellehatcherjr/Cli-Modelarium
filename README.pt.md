<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cli-modelarium-wordmark-dark.svg">
  <img alt="cli modelarium" src="docs/assets/cli-modelarium-wordmark-light.svg" width="420">
</picture>

Leia isto em outros idiomas: [English](README.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [한국어](README.ko.md) | [中文](README.zh.md) | [Deutsch](README.de.md) | [Italiano](README.it.md)

Nota: Este README é traduzido para acessibilidade. A própria ferramenta CLI Cli Modelarium produz saída apenas em inglês. Todos os comandos, mensagens de erro e saídas permanecem em inglês, independentemente da localização do sistema.

> Note: Features added after v0.1.0 (`--runs` in v0.1.1, statistical significance in v0.1.2, confidence intervals/paired tests/McNemar in v0.1.3) are documented in English only — translations pending.

> Compare saídas de LLM lado a lado do seu terminal - 9 provedores de nuvem + modelos locais, com streaming paralelo, avaliação em lote, pontuação LLM-as-judge, detecção de alucinação e asserções prontas para CI/CD.

[![CI](https://github.com/lavellehatcherjr/Cli-Modelarium/actions/workflows/ci.yml/badge.svg)](https://github.com/lavellehatcherjr/Cli-Modelarium/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cli-modelarium)](https://pypi.org/project/cli-modelarium/)
[![Downloads](https://img.shields.io/pepy/dt/cli-modelarium)](https://pepy.tech/project/cli-modelarium)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Platforms](https://img.shields.io/badge/platforms-Mac%20%7C%20Windows%20%7C%20Linux-lightgrey)](#)

<p align="center">
  <img src="docs/assets/cli-modelarium-demo.png" alt="Cli Modelarium help output showing the banner and available commands" width="520">
</p>

## O que ele faz

**Cli Modelarium** é uma ferramenta de linha de comando refinada para comparar saídas de LLM entre provedores, modelos, prompts de sistema e temperaturas - com streaming paralelo ao vivo, avaliação em lote, testes determinísticos e pontuação de qualidade integrados.

Útil para avaliar qual modelo se adequa à sua tarefa específica, executar testes de regressão de prompts em CI/CD, comparar modelos locais com APIs em nuvem ou construir conjuntos de dados de avaliação - tudo a partir de um único comando de terminal.

## Início rápido

```bash
pip install cli-modelarium

# Configurar chaves de API (salvas com segurança no keychain do seu SO)
cli-modelarium configure

# Execute sua primeira comparação
cli-modelarium "Explain quantum computing in one sentence" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro \
  --temperatures 0,0.7
```

É isso. Você verá os três modelos transmitirem suas respostas ao vivo em paralelo, com latência, contagens de tokens e custo exibidos em uma tabela de comparação limpa.

## Recursos

### 🤖 Provedores (9 na nuvem + locais ilimitados)

- **Provedores de nuvem:** OpenAI, Anthropic, Google (Gemini), xAI (Grok), DeepSeek, Mistral, Groq, OpenRouter, Alibaba (DashScope)
- **Modelos locais:** Ollama, LM Studio, vLLM, llama.cpp - qualquer servidor local compatível com OpenAI
- Combine modelos locais e em nuvem na mesma comparação
- Seleção de modelo configurável por chamada (sem listas codificadas)

### ⚡ Streaming paralelo

- Exibição ao vivo token por token em todos os modelos simultaneamente
- Rastreamento de Time-to-First-Token (TTFT) por modelo
- Veja qual modelo termina primeiro, observe as saídas divergirem em tempo real
- Streams de todos os 9 provedores (SSE por baixo)

### 📊 Múltiplos modos de comparação

- **Prompt único vs. múltiplos modelos** - comparações rápidas de "qual é o melhor?"
- **Prompt único vs. múltiplas temperaturas** - veja como a aleatoriedade afeta a saída
- **Múltiplos prompts de sistema vs. um prompt de usuário** - teste A/B de engenharia de prompts
- **Modo em lote** - multi-prompt × multi-modelo para trabalho de avaliação real
- **Comparações local vs. nuvem** - quantifique a lacuna (ou a ausência dela)

### 🧪 Recursos de avaliação

- **Asserções determinísticas** - 10 tipos de asserção (`contains`, `regex`, `json_valid`, `json_schema`, `max_length_chars`, `latency_under`, `cost_under` e mais) com saída de aprovado/falhou e códigos de saída de CI
- **Pontuação LLM-as-a-judge** - Use um LLM para pontuar as saídas de outros LLMs em critérios de qualidade
- **Painéis de juízes** - Múltiplos juízes calculam a média das pontuações para avaliação menos enviesada
- **Preset de detecção de alucinação** - Critérios prontos para uso para verificação de precisão factual
- **Critérios personalizados** - Defina suas próprias rubricas de pontuação
- **Auto-pular auto-avaliação** - Modelos juízes automaticamente pulados quando também estão sendo julgados

### 💾 Formatos de saída

- **Terminal ao vivo** - Painéis baseados em Rich com barras de progresso e exibição de streaming
- **CSV** - Amigável a planilhas (abra no Excel, Google Sheets, pandas)
- **JSON** - Estruturado para scripts e pipelines
- **Markdown** - Tabelas bonitas para postagens de blog e relatórios
- **Códigos de saída** - 0/1/2 refletindo status de aprovado/falhou para CI/CD

### 💰 Transparência de custos

- Custo por chamada exibido a partir do uso reportado por cada provedor
- Resumo de custo total por comparação
- Custo do juiz mostrado separadamente quando LLM-as-judge está habilitado
- Modelos locais exibidos como "Free"
- Flag `--max-cost` para evitar contas surpresa

### 🔒 Segurança

- Chaves de API armazenadas no keychain nativo do SO via `keyring` (Mac Keychain, Windows Credential Manager, Linux Secret Service)
- Validação de formato captura erros de colagem antes do armazenamento
- Redação de mensagens de erro previne vazamento de chaves em tracebacks
- Validação somente localhost para URLs de modelos locais
- `SECURITY.md` com política de divulgação responsável

### 🛡️ Tratamento de limites de taxa

- Limites de concorrência por provedor (padrão 5) respeitam todas as baselines de tier
- Retentativa automática de 429 com backoff exponencial
- 529 "overloaded" do Anthropic tratado separadamente dos limites de taxa
- Flag `--concurrency` para usuários avançados em tiers superiores
- Falha graciosa por modelo (outros modelos continuam)
- Os limites de taxa do tier gratuito do DashScope e do Qwen carro-chefe (qwen3.7-max) são mais restritos que os da maioria dos provedores; reduza `--concurrency` se você encontrar erros 429.

### 🌐 Multiplataforma

- Funciona de forma idêntica em macOS, Windows (10+ e ARM) e Linux
- Todo I/O de arquivo usa `pathlib` + codificação UTF-8 explícita
- Escrita CSV usa `newline=""` para compatibilidade com Windows
- Python 3.11+ requerido

### 📋 Experiência do desenvolvedor

- **Binário CLI único** - `pip install cli-modelarium` e pronto
- **UI refinada baseada em Rich** - Polimento de terminal no nível Claude Code
- **Saída JSON** - Encaminhe para qualquer coisa (`jq`, scripts, monitoramento)
- **Pronto para CI/CD** - Códigos de saída, saída estruturada, exemplo de GitHub Actions incluído
- **Licenciado sob Apache 2.0** - Use em qualquer projeto, comercial ou não

## Exemplos

### Compare 3 modelos em uma tarefa de programação

```bash
cli-modelarium "Write a Python function to find the longest palindromic substring" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro
```

### Avaliação em lote com asserções

Crie `eval.json`:

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

Execute-o:

```bash
cli-modelarium batch eval.json \
  --models gpt-5.5,claude-opus-4-7 \
  --output results.csv
```

### Pontue saídas com um juiz LLM

```bash
cli-modelarium "Explain recursion in one paragraph" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro,local/llama-3.3-70b \
  --judge claude-opus-4-7 \
  --judge-criteria "accuracy,clarity,brevity"
```

### Detecte alucinações contra fatos conhecidos

```bash
cli-modelarium "Tell me about the Eiffel Tower" \
  --models gpt-5.5,claude-opus-4-7 \
  --judge claude-opus-4-7 \
  --check-hallucination \
  --expected-facts "Built 1887-1889,Located in Paris France,Designed by Gustave Eiffel"
```

### Compare modelo local com APIs em nuvem

```bash
# Inicie o Ollama primeiro: ollama run llama3.3
cli-modelarium "Summarize the key features of microservices architecture" \
  --models local/llama-3.3-70b,gpt-5.5,claude-opus-4-7
```

### Execute em CI/CD (exemplo de GitHub Actions)

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

O comando sai com código 1 se a taxa de aprovação cair abaixo de 90%, fazendo o build falhar.

## Configuração

### Chaves de API

Cli Modelarium armazena chaves de API no keychain nativo do seu SO (Mac Keychain, Windows Credential Manager ou Linux Secret Service via `keyring`). As chaves nunca tocam o disco em texto simples.

```bash
# Configuração interativa (recomendada)
cli-modelarium configure

# Ou defina individualmente
cli-modelarium keys set openai
cli-modelarium keys set anthropic
cli-modelarium keys set google

# Verifique quais chaves estão configuradas
cli-modelarium keys list

# Remover uma chave
cli-modelarium keys delete openai
```

Você também pode usar variáveis de ambiente (útil para CI/CD):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

Variáveis de ambiente têm precedência sobre o armazenamento do keychain.

### Modelos locais (Ollama, LM Studio, etc.)

Modelos locais funcionam via endpoints compatíveis com OpenAI - sem chaves de API necessárias. A ferramenta detecta automaticamente a porta padrão do Ollama.

```bash
# Padrão: assume Ollama em localhost:11434
cli-modelarium "test" --models local/llama-3.3

# Use LM Studio em vez disso
cli-modelarium "test" --models local/qwen-3-32b --local-url http://localhost:1234/v1

# Salve uma URL local personalizada como padrão
cli-modelarium keys set local --base-url http://localhost:1234/v1
```

## Provedores suportados

| Provedor | Chaves de API Necessárias | Streaming | Rastreamento de Custos |
|----------|-----------------|-----------|---------------|
| OpenAI (GPT-5, GPT-5 mini, o3, o4-mini, etc.) | ✅ | ✅ | ✅ |
| Anthropic (Claude Opus 4.7, Sonnet 4.6, Haiku 4.5, etc.) | ✅ | ✅ | ✅ |
| Google (Gemini 3.1 Pro, Gemini 3 Flash, etc.) | ✅ | ✅ | ✅ |
| xAI (Grok 4.3, etc.) | ✅ | ✅ | ✅ |
| DeepSeek (V4 Pro, V4 Flash, etc.) | ✅ | ✅ | ✅ |
| Mistral (Large, Medium, Small) | ✅ | ✅ | ✅ |
| Groq (Llama, Mixtral, etc.) | ✅ | ✅ | ✅ |
| OpenRouter (qualquer modelo na plataforma) | ✅ | ✅ | ✅ |
| Alibaba/DashScope (Qwen3.7 Max, Qwen3.6 Flash, Qwen3 Coder, etc.; modelos Qwen selecionados, Internacional/Singapura) | ✅ | ✅ | ✅ |
| **Local: Ollama** | ❌ | ✅ | Gratuito |
| **Local: LM Studio** | ❌ | ✅ | Gratuito |
| **Local: vLLM** | ❌ | ✅ | Gratuito |
| **Local: llama.cpp server** | ❌ | ✅ | Gratuito |

Execute `cli-modelarium list-models` para ver todos os modelos atualmente suportados.

## Grupos de modelos

Em vez de listar IDs de modelos, `--models` aceita um atalho de grupo. Os grupos são filtrados de acordo com os provedores que você configurou, então um grupo só executa os modelos para os quais você realmente tem chaves.

**Grupos estáticos** (composição fixa):

| Grupo | Modelos |
|-------|---------|
| `all-premium` / `all-flagship` | gpt-5.5, claude-opus-4-7, gemini-3.1-pro, grok-4.3, deepseek-v4-pro, mistral-large-latest |
| `all-budget` | gpt-5.4-nano, claude-haiku-4-5, gemini-3.1-flash-lite, grok-4.1-fast, deepseek-v4-flash, mistral-small-latest |
| `all-reasoning` | o3, o4-mini, deepseek-reasoner, magistral-medium-latest, magistral-small-latest |
| `all-fast` | claude-haiku-4-5, gemini-3-flash, grok-4.1-fast, deepseek-v4-flash, llama-3.3-70b-versatile |
| `all-cheap` | gpt-4o-mini, claude-haiku-4-5, gemini-2.5-flash-lite, grok-4.1-fast, deepseek-v4-flash, mistral-small-latest |
| `all-open-weight` | gpt-oss-120b, gpt-oss-20b, llama-3.3-70b-versatile, meta-llama/llama-4-scout-17b-16e-instruct |

**Grupos dinâmicos** (resolvidos em tempo de execução):

- `all` — todos os modelos em nuvem para os quais você tem uma chave de API configurada (exclui modelos locais e OpenRouter). Isso pode se expandir para muitos modelos, então combine com `--max-cost`.
- `all-local` — todos os modelos reportados pelo seu servidor local em execução (Ollama / LM Studio / vLLM / llama.cpp). Se nenhum servidor estiver acessível, você recebe uma mensagem clara em vez de um erro.

```bash
cli-modelarium "Explique o teorema CAP" --models all-budget
cli-modelarium "Explique o teorema CAP" --models all --max-cost 0.50
cli-modelarium "Explique o teorema CAP" --models all-local
```

## Como funciona

Cli Modelarium usa uma camada de abstração de provedor modular que oculta as diferenças de API entre o array `messages` do OpenAI, o parâmetro `system` de nível superior do Anthropic, o `system_instruction` do Google e outros. Cada provedor implementa a mesma interface de streaming assíncrono, então a CLI pode executá-los todos em paralelo com `asyncio.gather()`.

Os cálculos de custo vêm do campo `usage` reportado por cada provedor (tokens de entrada, tokens de saída, tokens em cache) multiplicado pelas constantes de preço atuais. Os dados de preço foram verificados a partir da documentação oficial do provedor em **21 de junho de 2026** - veja [Notas e Limitações](#notas-e-limitações) para ressalvas.

Para modelos locais, o mesmo SDK Python da OpenAI é usado com uma `base_url` personalizada, já que Ollama, LM Studio, vLLM e llama.cpp expõem endpoints REST compatíveis com OpenAI.

## Notas e Limitações

### Dados de preço

Todos os preços incorporados ao Cli Modelarium foram verificados a partir da documentação oficial do provedor em **21 de junho de 2026**. Os preços de LLM mudam com frequência (às vezes mensalmente). A ferramenta exibe a data `pricing_as_of` em cada saída. Sempre verifique com a página oficial de preços de cada provedor antes de confiar em cálculos de custo para orçamento ou decisões de produção.

Os preços são a tarifa pública padrão/de tabela de cada provedor por 1M de tokens (não preços em lote, prioritários, fora de pico ou promocionais); para modelos com tiers baseados no tamanho da entrada, é exibido o tier inicial/de contexto curto, e o preço em cache é a tarifa de leitura de cache. Os custos do DashScope/Qwen refletem as tarifas sem raciocínio (a ferramenta envia `enable_thinking=false`).

Execute `cli-modelarium pricing` (ou `pricing --all`) para as tarifas atuais por modelo.

### Limites de taxa

O tratamento de limites de taxa e as configurações padrão de concorrência por provedor são baseados nos limites de taxa do provedor verificados em **21 de junho de 2026**. Os limites do seu tier específico podem diferir dos padrões assumidos aqui. Verifique seus limites atuais no painel oficial do provedor antes de construir suposições de capacidade de produção.

### Disponibilidade do modelo

Os modelos suportados pelo Cli Modelarium refletem o que os provedores ofereciam em **21 de junho de 2026**. Os provedores regularmente lançam novos modelos, descontinuam os mais antigos e ajustam capacidades. Se um modelo no registro não funcionar mais, execute `cli-modelarium list-models` e consulte a documentação do provedor.

### Não é um gateway de produção

Cli Modelarium foi projetado para avaliação e comparação - executando testes ad-hoc lado a lado entre provedores a partir de um terminal de desenvolvedor. NÃO é um gateway de inferência de produção. Se você precisa de roteamento em escala de produção, balanceamento de carga, cadeias de fallback ou inferência gerenciada por SLA, procure ferramentas construídas especificamente para esse propósito.

### Comparações de contagem de tokens entre provedores

As contagens de tokens mostradas nos resultados são reportadas pela API de cada provedor. Diferentes provedores usam diferentes tokenizadores, então "tokens de saída" não é diretamente comparável entre provedores para o mesmo texto. Se você está comparando eficiência de custo para uso em produção, execute prompts reais em sua carga de trabalho real - não confie apenas em cálculos por token entre provedores.

### Uso de LLM-as-a-Judge

Cli Modelarium inclui pontuação LLM-as-a-judge opcional (habilitada com a flag `--judge`), que usa um LLM para avaliar saídas de outros LLMs. Esta é uma metodologia padrão de benchmarking e é permitida sob os Termos de Serviço de todos os provedores suportados como atividade de avaliação/benchmarking.

Ao usar `--judge`, você é responsável por seguir os Termos de Serviço de cada provedor cujos modelos você usa. Os ToS de cada provedor se aplicam tanto aos modelos sendo julgados quanto ao próprio modelo juiz.

**Aviso de viés do juiz:** Juízes LLM têm vieses documentados (preferência pelo próprio, preferência pela mesma família, preferência por verbosidade). Pontuações do juiz são sinal útil, não verdade fundamental. Use painéis de juízes (`--judges` com múltiplos modelos) para reduzir viés.

### Detecção de alucinações

O preset de detecção de alucinações é um sinal de comparação útil entre modelos, não uma validação de verdade fundamental. A precisão da detecção varia com base no modelo juiz usado, no conhecimento de domínio necessário e se os fatos de referência são fornecidos via `--expected-facts`. Use-o para comparação de qualidade relativa, não para verificação de correção absoluta.

### Metodologia de comparação

LLMs são não determinísticos em temperatura > 0 - executar novamente o mesmo prompt pode produzir saídas diferentes. Uma única execução de comparação mostra UMA amostra de cada modelo, não um veredicto de qualidade definitivo.

Para tirar conclusões mais confiáveis:
- Use `--temperatures 0` para saídas mais determinísticas (onde suportado)
- Execute a mesma comparação 3-5 vezes e procure padrões
- Compare entre múltiplos prompts, não apenas um
- Use a flag `--output json` para salvar execuções para análise sistemática

## Sobre o autor

Cli Modelarium foi construído por **[Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)**.

### Conectar

- 💼 LinkedIn: [linkedin.com/in/lavellehatcherjr](https://linkedin.com/in/lavellehatcherjr)
- 🐙 GitHub: [github.com/lavellehatcherjr](https://github.com/lavellehatcherjr)
- 💬 Perguntas sobre este projeto: [abra uma issue](../../issues)
- 📩 Colaboração/oportunidades: entre em contato via LinkedIn

## Por que eu construí isso

Comparar saídas de LLM entre provedores é tedioso - diferentes SDKs, diferentes padrões de autenticação, diferentes formas de resposta, nenhuma maneira fácil de vê-los lado a lado com dados de custo e latência. Os refinados playgrounds em nuvem mostram apenas um provedor por vez, e as opções de código aberto disponíveis ou focam em roteamento de produção ou são plataformas de avaliação completas otimizadas para equipes.

Cli Modelarium é a pequena ferramenta CLI focada que faz uma coisa bem: comparação lado a lado com pontuação de qualidade, asserções, modo em lote e streaming - tudo projetado para o fluxo de trabalho de desenvolvedor centrado no terminal.

É intencionalmente focado: sem roteamento de produção, sem orquestração de agente, sem fine-tuning, sem GUI. Apenas comparação limpa e rápida da linha de comando.

Construído com uma abstração de provedor modular, execução paralela, cálculo de custo transparente e armazenamento seguro de chaves via sistemas de keychain do SO para usuários locais.

## Contribuindo

Issues e PRs bem-vindos. Veja [CONTRIBUTING.md](CONTRIBUTING.md) para diretrizes.

Para problemas de segurança, por favor veja [SECURITY.md](SECURITY.md) - não abra issues públicas para preocupações de segurança.

## Licença

Licenciado sob a [Apache License, Version 2.0](LICENSE).

Veja o arquivo [NOTICE](NOTICE) para requisitos de atribuição.

---

Construído por [Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)

Licenciado sob Apache 2.0. Issues, PRs e conversas bem-vindos.
