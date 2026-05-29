<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cli-modelarium-wordmark-dark.svg">
  <img alt="cli modelarium" src="docs/assets/cli-modelarium-wordmark-light.svg" width="420">
</picture>

Lesen Sie dies in anderen Sprachen: [English](README.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [한국어](README.ko.md) | [中文](README.zh.md) | [Português](README.pt.md) | [Italiano](README.it.md)

Hinweis: Diese README ist aus Gründen der Zugänglichkeit übersetzt. Das Cli Modelarium CLI-Tool selbst gibt nur Englisch aus. Alle Befehle, Fehlermeldungen und Ausgaben bleiben unabhängig von Ihrer System-Locale auf Englisch.

> Note: Features added after v0.1.0 (`--runs` in v0.1.1, statistical significance in v0.1.2, confidence intervals/paired tests/McNemar in v0.1.3) are documented in English only — translations pending.

> Vergleichen Sie LLM-Ausgaben nebeneinander von Ihrem Terminal aus - 8 Cloud-Anbieter + lokale Modelle, mit parallelem Streaming, Batch-Evaluierung, LLM-as-Judge-Scoring, Halluzinationserkennung und CI/CD-fähigen Assertions.

[![CI](https://github.com/lavellehatcherjr/Cli-Modelarium/actions/workflows/ci.yml/badge.svg)](https://github.com/lavellehatcherjr/Cli-Modelarium/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cli-modelarium)](https://pypi.org/project/cli-modelarium/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Platforms](https://img.shields.io/badge/platforms-Mac%20%7C%20Windows%20%7C%20Linux-lightgrey)](#)

<p align="center">
  <img src="docs/assets/cli-modelarium-demo.png" alt="Cli Modelarium help output showing the banner and available commands" width="520">
</p>

## Was es tut

**Cli Modelarium** ist ein ausgereiftes Kommandozeilen-Tool zum Vergleichen von LLM-Ausgaben über Anbieter, Modelle, System-Prompts und Temperaturen hinweg - mit eingebautem Live-Parallel-Streaming, Batch-Evaluierung, deterministischen Tests und Qualitäts-Scoring.

Nützlich, um zu bewerten, welches Modell zu Ihrer spezifischen Aufgabe passt, Prompt-Regressionstests in CI/CD auszuführen, lokale Modelle mit Cloud-APIs zu vergleichen oder Evaluierungs-Datasets zu erstellen - alles aus einem einzigen Terminal-Befehl.

## Schnellstart

```bash
pip install cli-modelarium

# API-Schlüssel konfigurieren (sicher im OS-Keychain gespeichert)
cli-modelarium configure

# Führen Sie Ihren ersten Vergleich aus
cli-modelarium "Explain quantum computing in one sentence" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro \
  --temperatures 0,0.7
```

Das ist alles. Sie sehen alle drei Modelle ihre Antworten parallel live streamen, mit Latenz, Token-Anzahl und Kosten, die in einer übersichtlichen Vergleichstabelle angezeigt werden.

## Funktionen

### 🤖 Anbieter (8 Cloud + unbegrenzt lokal)

- **Cloud-Anbieter:** OpenAI, Anthropic, Google (Gemini), xAI (Grok), DeepSeek, Mistral, Groq, OpenRouter
- **Lokale Modelle:** Ollama, LM Studio, vLLM, llama.cpp - jeder OpenAI-kompatible lokale Server
- Mischen Sie lokale und Cloud-Modelle im selben Vergleich
- Konfigurierbare Modellauswahl pro Aufruf (keine fest codierten Listen)

### ⚡ Paralleles Streaming

- Live-Anzeige Token für Token über alle Modelle gleichzeitig
- Time-to-First-Token (TTFT)-Tracking pro Modell
- Sehen Sie, welches Modell zuerst fertig ist, beobachten Sie Ausgaben in Echtzeit divergieren
- Streams von allen 8 Anbietern (SSE im Hintergrund)

### 📊 Mehrere Vergleichsmodi

- **Einzelner Prompt vs. mehrere Modelle** - schnelle "welches ist am besten?"-Vergleiche
- **Einzelner Prompt vs. mehrere Temperaturen** - sehen Sie, wie Zufälligkeit die Ausgabe beeinflusst
- **Mehrere System-Prompts vs. ein User-Prompt** - A/B-Test von Prompt-Engineering
- **Batch-Modus** - Multi-Prompt × Multi-Modell für echte Evaluierungsarbeit
- **Lokale vs. Cloud-Vergleiche** - quantifizieren Sie die Lücke (oder deren Fehlen)

### 🧪 Evaluierungsfunktionen

- **Deterministische Assertions** - 10 Assertion-Typen (`contains`, `regex`, `json_valid`, `json_schema`, `max_length_chars`, `latency_under`, `cost_under` und mehr) mit Pass/Fail-Ausgabe und CI-Exit-Codes
- **LLM-as-a-Judge-Scoring** - Verwenden Sie ein LLM, um Ausgaben anderer nach Qualitätskriterien zu bewerten
- **Judge-Panels** - Mehrere Judges mitteln Punktzahlen für eine weniger voreingenommene Bewertung
- **Halluzinationserkennungs-Preset** - Sofort einsatzbereite Kriterien für die Überprüfung der sachlichen Genauigkeit
- **Benutzerdefinierte Kriterien** - Definieren Sie Ihre eigenen Bewertungsrubriken
- **Auto-Skip bei Selbstbewertung** - Judge-Modelle werden automatisch übersprungen, wenn sie auch bewertet werden

### 💾 Ausgabeformate

- **Live-Terminal** - Rich-basierte Panels mit Fortschrittsbalken und Streaming-Anzeige
- **CSV** - Tabellenkalkulationsfreundlich (in Excel, Google Sheets, pandas öffnen)
- **JSON** - Strukturiert für Skripte und Pipelines
- **Markdown** - Schöne Tabellen für Blogbeiträge und Berichte
- **Exit-Codes** - 0/1/2 reflektieren Pass/Fail-Status für CI/CD

### 💰 Kostentransparenz

- Kosten pro Aufruf basierend auf der von jedem Anbieter gemeldeten Nutzung
- Gesamtkostenübersicht pro Vergleich
- Judge-Kosten separat angezeigt, wenn LLM-as-Judge aktiviert ist
- Lokale Modelle werden als "Free" angezeigt
- `--max-cost`-Flag zur Vermeidung überraschender Rechnungen

### 🔒 Sicherheit

- API-Schlüssel werden über `keyring` im OS-nativen Keychain gespeichert (Mac Keychain, Windows Credential Manager, Linux Secret Service)
- Format-Validierung fängt Einfügefehler vor der Speicherung ab
- Redaktion von Fehlermeldungen verhindert Schlüssellecks in Tracebacks
- Localhost-only-Validierung für lokale Modell-URLs
- `SECURITY.md` mit Responsible-Disclosure-Richtlinie

### 🛡️ Rate-Limit-Handhabung

- Concurrency-Limits pro Anbieter (Standard 5) respektieren alle Tier-Baselines
- Automatischer 429-Retry mit exponentiellem Backoff
- Anthropics 529 "overloaded" wird separat von Rate-Limits behandelt
- `--concurrency`-Flag für Power-User in höheren Tiers
- Graceful Fehlerbehandlung pro Modell (andere Modelle laufen weiter)

### 🌐 Plattformübergreifend

- Funktioniert identisch auf macOS, Windows (10+ und ARM) und Linux
- Alle Datei-I/O verwenden `pathlib` + explizite UTF-8-Codierung
- CSV-Schreiben verwendet `newline=""` für Windows-Kompatibilität
- Python 3.11+ erforderlich

### 📋 Entwicklererfahrung

- **Einzelne CLI-Binary** - `pip install cli-modelarium` und fertig
- **Ausgereifte Rich-basierte UI** - Terminal-Politur auf Claude-Code-Niveau
- **JSON-Ausgabe** - In alles pipen (`jq`, Skripte, Monitoring)
- **CI/CD-bereit** - Exit-Codes, strukturierte Ausgabe, GitHub-Actions-Beispiel enthalten
- **Apache-2.0-lizenziert** - Verwendung in jedem Projekt, kommerziell oder anderweitig

## Beispiele

### Vergleichen Sie 3 Modelle bei einer Coding-Aufgabe

```bash
cli-modelarium "Write a Python function to find the longest palindromic substring" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro
```

### Batch-Evaluierung mit Assertions

Erstellen Sie `eval.json`:

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

Führen Sie es aus:

```bash
cli-modelarium batch eval.json \
  --models gpt-5.5,claude-opus-4-7 \
  --output results.csv
```

### Bewerten Sie Ausgaben mit einem LLM-Judge

```bash
cli-modelarium "Explain recursion in one paragraph" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro,local/llama-3.3-70b \
  --judge claude-opus-4-7 \
  --judge-criteria "accuracy,clarity,brevity"
```

### Halluzinationen gegen bekannte Fakten erkennen

```bash
cli-modelarium "Tell me about the Eiffel Tower" \
  --models gpt-5.5,claude-opus-4-7 \
  --judge claude-opus-4-7 \
  --check-hallucination \
  --expected-facts "Built 1887-1889,Located in Paris France,Designed by Gustave Eiffel"
```

### Lokales Modell mit Cloud-APIs vergleichen

```bash
# Ollama zuerst starten: ollama run llama3.3
cli-modelarium "Summarize the key features of microservices architecture" \
  --models local/llama-3.3-70b,gpt-5.5,claude-opus-4-7
```

### In CI/CD ausführen (GitHub Actions-Beispiel)

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

Der Befehl beendet mit Code 1, wenn die Pass-Rate unter 90% fällt, wodurch der Build fehlschlägt.

## Konfiguration

### API-Schlüssel

Cli Modelarium speichert API-Schlüssel im OS-nativen Keychain Ihres Systems (Mac Keychain, Windows Credential Manager oder Linux Secret Service über `keyring`). Schlüssel berühren niemals die Festplatte im Klartext.

```bash
# Interaktives Setup (empfohlen)
cli-modelarium configure

# Oder einzeln einstellen
cli-modelarium keys set openai
cli-modelarium keys set anthropic
cli-modelarium keys set google

# Überprüfen, welche Schlüssel konfiguriert sind
cli-modelarium keys list

# Einen Schlüssel entfernen
cli-modelarium keys delete openai
```

Sie können auch Umgebungsvariablen verwenden (nützlich für CI/CD):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

Umgebungsvariablen haben Vorrang vor der Keychain-Speicherung.

### Lokale Modelle (Ollama, LM Studio usw.)

Lokale Modelle funktionieren über OpenAI-kompatible Endpunkte - keine API-Schlüssel erforderlich. Das Tool erkennt automatisch den Standard-Ollama-Port.

```bash
# Standard: nimmt Ollama auf localhost:11434 an
cli-modelarium "test" --models local/llama-3.3

# Stattdessen LM Studio verwenden
cli-modelarium "test" --models local/qwen-3-32b --local-url http://localhost:1234/v1

# Eine benutzerdefinierte lokale URL als Standard speichern
cli-modelarium keys set local --base-url http://localhost:1234/v1
```

## Unterstützte Anbieter

| Anbieter | API-Schlüssel erforderlich | Streaming | Kostenverfolgung |
|----------|-----------------|-----------|---------------|
| OpenAI (GPT-5, GPT-5 mini, o3, o4-mini, usw.) | ✅ | ✅ | ✅ |
| Anthropic (Claude Opus 4.7, Sonnet 4.6, Haiku 4.5, usw.) | ✅ | ✅ | ✅ |
| Google (Gemini 3.1 Pro, Gemini 3 Flash, usw.) | ✅ | ✅ | ✅ |
| xAI (Grok 4.1, usw.) | ✅ | ✅ | ✅ |
| DeepSeek (V3, R1) | ✅ | ✅ | ✅ |
| Mistral (Large, Medium, Small) | ✅ | ✅ | ✅ |
| Groq (Llama, Mixtral, usw.) | ✅ | ✅ | ✅ |
| OpenRouter (jedes Modell auf der Plattform) | ✅ | ✅ | ✅ |
| **Lokal: Ollama** | ❌ | ✅ | Kostenlos |
| **Lokal: LM Studio** | ❌ | ✅ | Kostenlos |
| **Lokal: vLLM** | ❌ | ✅ | Kostenlos |
| **Lokal: llama.cpp server** | ❌ | ✅ | Kostenlos |

Führen Sie `cli-modelarium list-models` aus, um alle derzeit unterstützten Modelle zu sehen.

## Wie es funktioniert

Cli Modelarium verwendet eine modulare Anbieter-Abstraktionsschicht, die die API-Unterschiede zwischen OpenAIs `messages`-Array, Anthropics `system`-Parameter auf oberster Ebene, Googles `system_instruction` und anderen verbirgt. Jeder Anbieter implementiert dasselbe asynchrone Streaming-Interface, sodass die CLI sie alle parallel mit `asyncio.gather()` ausführen kann.

Kostenberechnungen stammen aus dem von jedem Anbieter gemeldeten `usage`-Feld (Input-Tokens, Output-Tokens, gecachte Tokens), multipliziert mit aktuellen Preiskonstanten. Preisdaten wurden am **25. Mai 2026** aus der offiziellen Anbieterdokumentation überprüft - siehe [Hinweise und Einschränkungen](#hinweise-und-einschränkungen) für Vorbehalte.

Für lokale Modelle wird dasselbe OpenAI Python SDK mit einer benutzerdefinierten `base_url` verwendet, da Ollama, LM Studio, vLLM und llama.cpp alle OpenAI-kompatible REST-Endpunkte bereitstellen.

## Hinweise und Einschränkungen

### Preisdaten

Alle in Cli Modelarium integrierten Preise wurden am **25. Mai 2026** aus der offiziellen Anbieterdokumentation überprüft. LLM-Preise ändern sich häufig (manchmal monatlich). Das Tool zeigt das `pricing_as_of`-Datum in jeder Ausgabe an. Überprüfen Sie immer die offizielle Preisseite jedes Anbieters, bevor Sie sich für Budgetierung oder Produktionsentscheidungen auf Kostenberechnungen verlassen.

### Rate-Limits

Die Rate-Limit-Handhabung und die Standard-Concurrency-Einstellungen pro Anbieter basieren auf den am **25. Mai 2026** überprüften Anbieter-Rate-Limits. Die Limits Ihres spezifischen Tiers können von den hier angenommenen Standardwerten abweichen. Überprüfen Sie Ihre aktuellen Limits anhand des offiziellen Anbieter-Dashboards, bevor Sie Produktionskapazitätsannahmen treffen.

### Modellverfügbarkeit

Die von Cli Modelarium unterstützten Modelle spiegeln wider, was Anbieter am **25. Mai 2026** angeboten haben. Anbieter veröffentlichen regelmäßig neue Modelle, veralten ältere und passen Fähigkeiten an. Wenn ein Modell in der Registry nicht mehr funktioniert, führen Sie `cli-modelarium list-models` aus und überprüfen Sie die Dokumentation des Anbieters.

### Kein produktionsreifes Gateway

Cli Modelarium ist für Evaluierung und Vergleich konzipiert - Ausführung von Ad-hoc-Tests nebeneinander über Anbieter hinweg von einem Entwickler-Terminal aus. Es ist KEIN Produktions-Inferenz-Gateway. Wenn Sie produktionsskalierbares Routing, Load-Balancing, Fallback-Chains oder SLA-verwaltete Inferenz benötigen, suchen Sie nach Tools, die speziell für diesen Zweck entwickelt wurden.

### Token-Anzahl-Vergleiche zwischen Anbietern

Die in den Ergebnissen angezeigten Token-Anzahlen werden von der API jedes Anbieters gemeldet. Verschiedene Anbieter verwenden verschiedene Tokenizer, sodass "Output-Tokens" zwischen Anbietern für denselben Text nicht direkt vergleichbar sind. Wenn Sie die Kosteneffizienz für den Produktionseinsatz vergleichen, führen Sie echte Prompts in Ihrer tatsächlichen Workload aus - verlassen Sie sich nicht nur auf Pro-Token-Berechnungen über Anbieter hinweg.

### LLM-as-a-Judge-Nutzung

Cli Modelarium beinhaltet optionales LLM-as-a-Judge-Scoring (aktiviert mit dem `--judge`-Flag), das ein LLM verwendet, um Ausgaben anderer LLMs zu bewerten. Dies ist eine Standard-Benchmarking-Methodik und ist unter den Nutzungsbedingungen aller unterstützten Anbieter als Evaluierungs-/Benchmarking-Aktivität erlaubt.

Bei Verwendung von `--judge` sind Sie dafür verantwortlich, die Nutzungsbedingungen jedes Anbieters einzuhalten, dessen Modelle Sie verwenden. Die ToS jedes Anbieters gelten sowohl für die bewerteten Modelle als auch für das Judge-Modell selbst.

**Judge-Bias-Hinweis:** LLM-Judges haben dokumentierte Voreingenommenheiten (Selbstpräferenz, Präferenz für dieselbe Familie, Verbositäts-Präferenz). Judge-Punktzahlen sind nützliches Signal, keine Ground Truth. Verwenden Sie Judge-Panels (`--judges` mit mehreren Modellen), um Bias zu reduzieren.

### Halluzinationserkennung

Das Halluzinationserkennungs-Preset ist ein nützliches Vergleichssignal zwischen Modellen, keine Ground-Truth-Validierung. Die Erkennungsgenauigkeit variiert je nach verwendetem Judge-Modell, erforderlichem Domänenwissen und ob Referenzfakten über `--expected-facts` bereitgestellt werden. Verwenden Sie es für relativen Qualitätsvergleich, nicht für absolute Korrektheitsverifizierung.

### Vergleichsmethodik

LLMs sind bei Temperatur > 0 nicht deterministisch - das erneute Ausführen desselben Prompts kann unterschiedliche Ausgaben erzeugen. Ein einzelner Vergleichsdurchlauf zeigt Ihnen EINE Stichprobe von jedem Modell, kein endgültiges Qualitätsurteil.

Um zuverlässigere Schlussfolgerungen zu ziehen:
- Verwenden Sie `--temperatures 0` für deterministischere Ausgaben (wo unterstützt)
- Führen Sie denselben Vergleich 3-5 Mal aus und suchen Sie nach Mustern
- Vergleichen Sie über mehrere Prompts hinweg, nicht nur einen
- Verwenden Sie das `--output json`-Flag, um Durchläufe für systematische Analyse zu speichern

## Über den Autor

Cli Modelarium wurde von **[Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)** entwickelt.

### Verbinden

- 💼 LinkedIn: [linkedin.com/in/lavellehatcherjr](https://linkedin.com/in/lavellehatcherjr)
- 🐙 GitHub: [github.com/lavellehatcherjr](https://github.com/lavellehatcherjr)
- 💬 Fragen zu diesem Projekt: [Issue öffnen](../../issues)
- 📩 Kollaboration/Möglichkeiten: über LinkedIn kontaktieren

## Warum ich das gebaut habe

Das Vergleichen von LLM-Ausgaben über Anbieter hinweg ist mühsam - verschiedene SDKs, verschiedene Auth-Patterns, verschiedene Antwortformen, keine einfache Möglichkeit, sie nebeneinander mit Kosten- und Latenzdaten zu sehen. Die ausgereiften Cloud-Playgrounds zeigen jeweils nur einen Anbieter, und die verfügbaren Open-Source-Optionen konzentrieren sich entweder auf Produktions-Routing oder sind vollständige Evaluierungsplattformen, die für Teams optimiert sind.

Cli Modelarium ist das kleine, fokussierte CLI-Tool, das eine Sache gut macht: Nebeneinandervergleich mit Qualitäts-Scoring, Assertions, Batch-Modus und Streaming - alles für den terminal-orientierten Entwickler-Workflow konzipiert.

Es ist bewusst fokussiert: kein Produktions-Routing, keine Agent-Orchestrierung, kein Fine-Tuning, keine GUI. Nur sauberer, schneller Vergleich aus der Kommandozeile.

Gebaut mit einer modularen Anbieter-Abstraktion, paralleler Ausführung, transparenter Kostenberechnung und sicherer Schlüsselspeicherung über OS-Keychain-Systeme für lokale Nutzer.

## Mitwirken

Issues und PRs willkommen. Siehe [CONTRIBUTING.md](CONTRIBUTING.md) für Richtlinien.

Für Sicherheitsprobleme siehe bitte [SECURITY.md](SECURITY.md) - reichen Sie keine öffentlichen Issues für Sicherheitsbedenken ein.

## Lizenz

Lizenziert unter der [Apache License, Version 2.0](LICENSE).

Siehe die [NOTICE](NOTICE)-Datei für Attribution-Anforderungen.

---

Gebaut von [Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)

Lizenziert unter Apache 2.0. Issues, PRs und Gespräche willkommen.
