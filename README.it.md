<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cli-modelarium-wordmark-dark.svg">
  <img alt="cli modelarium" src="docs/assets/cli-modelarium-wordmark-light.svg" width="420">
</picture>

Leggi questo in altre lingue: [English](README.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [한국어](README.ko.md) | [中文](README.zh.md) | [Deutsch](README.de.md) | [Português](README.pt.md)

Nota: Questo README è tradotto per accessibilità. Lo strumento CLI Cli Modelarium stesso produce output solo in inglese. Tutti i comandi, i messaggi di errore e gli output rimangono in inglese indipendentemente dalle impostazioni locali del sistema.

> Note: Features added after v0.1.0 (`--runs` in v0.1.1, statistical significance in v0.1.2, confidence intervals/paired tests/McNemar in v0.1.3) are documented in English only — translations pending.

> Confronta gli output degli LLM affiancati dal tuo terminale - 8 provider cloud + modelli locali, con streaming parallelo, valutazione batch, scoring LLM-as-judge, rilevamento delle allucinazioni e asserzioni pronte per CI/CD.

[![PyPI](https://img.shields.io/pypi/v/cli-modelarium)](https://pypi.org/project/cli-modelarium/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Platforms](https://img.shields.io/badge/platforms-Mac%20%7C%20Windows%20%7C%20Linux-lightgrey)](#)

## Cosa fa

**Cli Modelarium** è uno strumento da riga di comando curato per confrontare gli output degli LLM tra provider, modelli, system prompt e temperature - con streaming parallelo live, valutazione batch, test deterministici e scoring di qualità integrati.

Utile per valutare quale modello si adatta al tuo compito specifico, eseguire test di regressione dei prompt in CI/CD, confrontare modelli locali con API cloud o costruire dataset di valutazione - tutto da un singolo comando del terminale.

## Avvio rapido

```bash
pip install cli-modelarium

# Configura le chiavi API (salvate in modo sicuro nel portachiavi del tuo SO)
cli-modelarium configure

# Esegui il tuo primo confronto
cli-modelarium "Explain quantum computing in one sentence" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro \
  --temperatures 0,0.7
```

Ecco fatto. Si vedranno tutti e tre i modelli trasmettere le loro risposte in parallelo dal vivo, con latenza, conteggi di token e costo mostrati in una tabella di confronto pulita.

## Funzionalità

### 🤖 Provider (8 cloud + locali illimitati)

- **Provider cloud:** OpenAI, Anthropic, Google (Gemini), xAI (Grok), DeepSeek, Mistral, Groq, OpenRouter
- **Modelli locali:** Ollama, LM Studio, vLLM, llama.cpp - qualsiasi server locale compatibile con OpenAI
- Combina modelli locali e cloud nello stesso confronto
- Selezione del modello configurabile per chiamata (nessun elenco hardcoded)

### ⚡ Streaming parallelo

- Visualizzazione live token per token su tutti i modelli contemporaneamente
- Tracciamento Time-to-First-Token (TTFT) per modello
- Vedere quale modello finisce per primo, osservare gli output divergere in tempo reale
- Streams da tutti gli 8 provider (SSE sotto il cofano)

### 📊 Modalità di confronto multiple

- **Prompt singolo vs. modelli multipli** - confronti rapidi "qual è il migliore?"
- **Prompt singolo vs. temperature multiple** - vedere come la casualità influisce sull'output
- **System prompt multipli vs. un prompt utente** - test A/B di prompt engineering
- **Modalità batch** - multi-prompt × multi-modello per il vero lavoro di valutazione
- **Confronti locale vs. cloud** - quantificare il divario (o la sua assenza)

### 🧪 Funzionalità di valutazione

- **Asserzioni deterministiche** - 10 tipi di asserzione (`contains`, `regex`, `json_valid`, `json_schema`, `max_length_chars`, `latency_under`, `cost_under` e altri) con output pass/fail e codici di uscita CI
- **Scoring LLM-as-a-judge** - Usare un LLM per assegnare punteggi agli output di altri LLM su criteri di qualità
- **Pannelli di giudici** - Più giudici calcolano la media dei punteggi per una valutazione meno distorta
- **Preset di rilevamento allucinazioni** - Criteri pronti all'uso per il controllo dell'accuratezza fattuale
- **Criteri personalizzati** - Definire le proprie rubriche di scoring
- **Auto-skip dell'autovalutazione** - I modelli giudici vengono automaticamente saltati quando sono anche giudicati

### 💾 Formati di output

- **Terminal live** - Pannelli basati su Rich con barre di avanzamento e visualizzazione streaming
- **CSV** - Adatto ai fogli di calcolo (apri in Excel, Google Sheets, pandas)
- **JSON** - Strutturato per script e pipeline
- **Markdown** - Tabelle eleganti per post di blog e report
- **Codici di uscita** - 0/1/2 che riflettono lo stato pass/fail per CI/CD

### 💰 Trasparenza dei costi

- Costo per chiamata mostrato dall'utilizzo riportato da ciascun provider
- Riepilogo del costo totale per confronto
- Costo del giudice mostrato separatamente quando LLM-as-judge è abilitato
- I modelli locali vengono visualizzati come "Free"
- Flag `--max-cost` per prevenire fatture a sorpresa

### 🔒 Sicurezza

- Le chiavi API sono archiviate nel portachiavi nativo del SO tramite `keyring` (Mac Keychain, Windows Credential Manager, Linux Secret Service)
- La validazione del formato rileva errori di incollaggio prima dell'archiviazione
- La redazione dei messaggi di errore previene la fuga di chiavi nei traceback
- Validazione solo localhost per gli URL dei modelli locali
- `SECURITY.md` con politica di divulgazione responsabile

### 🛡️ Gestione dei limiti di velocità

- Limiti di concorrenza per provider (default 5) rispettano tutte le baseline di livello
- Riprova 429 automatica con backoff esponenziale
- Il 529 "overloaded" di Anthropic è gestito separatamente dai limiti di velocità
- Flag `--concurrency` per utenti avanzati su livelli superiori
- Fallimento elegante per modello (gli altri modelli continuano)

### 🌐 Multipiattaforma

- Funziona in modo identico su macOS, Windows (10+ e ARM) e Linux
- Tutti gli I/O di file usano `pathlib` + codifica UTF-8 esplicita
- La scrittura CSV usa `newline=""` per la compatibilità con Windows
- Python 3.11+ richiesto

### 📋 Esperienza dello sviluppatore

- **Binary CLI singolo** - `pip install cli-modelarium` e hai finito
- **UI curata basata su Rich** - Rifinitura del terminale di livello Claude Code
- **Output JSON** - Pipe in qualsiasi cosa (`jq`, script, monitoraggio)
- **Pronto per CI/CD** - Codici di uscita, output strutturato, esempio GitHub Actions incluso
- **Licenza Apache 2.0** - Usare in qualsiasi progetto, commerciale o meno

## Esempi

### Confronta 3 modelli su un compito di programmazione

```bash
cli-modelarium "Write a Python function to find the longest palindromic substring" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro
```

### Valutazione batch con asserzioni

Creare `eval.json`:

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

Eseguirlo:

```bash
cli-modelarium batch eval.json \
  --models gpt-5.5,claude-opus-4-7 \
  --output results.csv
```

### Valutare gli output con un giudice LLM

```bash
cli-modelarium "Explain recursion in one paragraph" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro,local/llama-3.3-70b \
  --judge claude-opus-4-7 \
  --judge-criteria "accuracy,clarity,brevity"
```

### Rilevare allucinazioni rispetto a fatti noti

```bash
cli-modelarium "Tell me about the Eiffel Tower" \
  --models gpt-5.5,claude-opus-4-7 \
  --judge claude-opus-4-7 \
  --check-hallucination \
  --expected-facts "Built 1887-1889,Located in Paris France,Designed by Gustave Eiffel"
```

### Confrontare un modello locale con API cloud

```bash
# Avviare prima Ollama: ollama run llama3.3
cli-modelarium "Summarize the key features of microservices architecture" \
  --models local/llama-3.3-70b,gpt-5.5,claude-opus-4-7
```

### Eseguire in CI/CD (esempio GitHub Actions)

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

Il comando esce con codice 1 se il tasso di superamento scende sotto il 90%, facendo fallire la build.

## Configurazione

### Chiavi API

Cli Modelarium archivia le chiavi API nel portachiavi nativo del tuo SO (Mac Keychain, Windows Credential Manager o Linux Secret Service tramite `keyring`). Le chiavi non toccano mai il disco in chiaro.

```bash
# Configurazione interattiva (consigliata)
cli-modelarium configure

# Oppure impostare individualmente
cli-modelarium keys set openai
cli-modelarium keys set anthropic
cli-modelarium keys set google

# Verificare quali chiavi sono configurate
cli-modelarium keys list

# Rimuovere una chiave
cli-modelarium keys delete openai
```

Si possono anche usare variabili d'ambiente (utile per CI/CD):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

Le variabili d'ambiente hanno la precedenza sull'archiviazione del portachiavi.

### Modelli locali (Ollama, LM Studio, ecc.)

I modelli locali funzionano tramite endpoint compatibili con OpenAI - nessuna chiave API necessaria. Lo strumento rileva automaticamente la porta predefinita di Ollama.

```bash
# Default: presuppone Ollama su localhost:11434
cli-modelarium "test" --models local/llama-3.3

# Usare LM Studio invece
cli-modelarium "test" --models local/qwen-3-32b --local-url http://localhost:1234/v1

# Salvare un URL locale personalizzato come predefinito
cli-modelarium keys set local --base-url http://localhost:1234/v1
```

## Provider supportati

| Provider | Chiavi API Necessarie | Streaming | Tracciamento Costi |
|----------|-----------------|-----------|---------------|
| OpenAI (GPT-5, GPT-5 mini, o3, o4-mini, ecc.) | ✅ | ✅ | ✅ |
| Anthropic (Claude Opus 4.7, Sonnet 4.6, Haiku 4.5, ecc.) | ✅ | ✅ | ✅ |
| Google (Gemini 3.1 Pro, Gemini 3 Flash, ecc.) | ✅ | ✅ | ✅ |
| xAI (Grok 4.1, ecc.) | ✅ | ✅ | ✅ |
| DeepSeek (V3, R1) | ✅ | ✅ | ✅ |
| Mistral (Large, Medium, Small) | ✅ | ✅ | ✅ |
| Groq (Llama, Mixtral, ecc.) | ✅ | ✅ | ✅ |
| OpenRouter (qualsiasi modello sulla piattaforma) | ✅ | ✅ | ✅ |
| **Locale: Ollama** | ❌ | ✅ | Gratuito |
| **Locale: LM Studio** | ❌ | ✅ | Gratuito |
| **Locale: vLLM** | ❌ | ✅ | Gratuito |
| **Locale: llama.cpp server** | ❌ | ✅ | Gratuito |

Eseguire `cli-modelarium list-models` per vedere tutti i modelli attualmente supportati.

## Come funziona

Cli Modelarium usa un livello di astrazione del provider modulare che nasconde le differenze API tra l'array `messages` di OpenAI, il parametro `system` di livello superiore di Anthropic, il `system_instruction` di Google e altri. Ogni provider implementa la stessa interfaccia di streaming asincrono, quindi la CLI può eseguirli tutti in parallelo con `asyncio.gather()`.

I calcoli dei costi provengono dal campo `usage` riportato da ciascun provider (token di input, token di output, token in cache) moltiplicato per le costanti di prezzo correnti. I dati sui prezzi sono stati verificati dalla documentazione ufficiale del provider il **25 maggio 2026** - vedere [Note e limitazioni](#note-e-limitazioni) per gli avvertimenti.

Per i modelli locali, viene usato lo stesso SDK Python OpenAI con una `base_url` personalizzata, poiché Ollama, LM Studio, vLLM e llama.cpp espongono tutti endpoint REST compatibili con OpenAI.

## Note e limitazioni

### Dati sui prezzi

Tutti i prezzi integrati in Cli Modelarium sono stati verificati dalla documentazione ufficiale del provider il **25 maggio 2026**. I prezzi degli LLM cambiano frequentemente (a volte mensilmente). Lo strumento visualizza la data `pricing_as_of` in ogni output. Verificare sempre con la pagina dei prezzi ufficiale di ciascun provider prima di fare affidamento sui calcoli dei costi per il budgeting o le decisioni di produzione.

### Limiti di velocità

La gestione dei limiti di velocità e le impostazioni di concorrenza predefinite per provider si basano sui limiti di velocità del provider verificati il **25 maggio 2026**. I limiti del livello specifico possono differire dai default assunti qui. Verificare i limiti correnti rispetto alla dashboard ufficiale del provider prima di costruire ipotesi di capacità di produzione.

### Disponibilità del modello

I modelli supportati da Cli Modelarium riflettono ciò che i provider offrivano il **25 maggio 2026**. I provider rilasciano regolarmente nuovi modelli, depreca quelli più vecchi e ne adegua le capacità. Se un modello nel registro non funziona più, eseguire `cli-modelarium list-models` e controllare la documentazione del provider.

### Non è un gateway di produzione

Cli Modelarium è progettato per la valutazione e il confronto - eseguendo test ad-hoc affiancati tra provider da un terminale dello sviluppatore. NON è un gateway di inferenza di produzione. Se serve routing su scala di produzione, bilanciamento del carico, catene di fallback o inferenza gestita da SLA, cercare strumenti specificamente costruiti per quello scopo.

### Confronti del conteggio dei token tra provider

I conteggi dei token mostrati nei risultati sono riportati dall'API di ciascun provider. Provider diversi usano tokenizer diversi, quindi i "token di output" non sono direttamente comparabili tra provider per lo stesso testo. Se si confronta l'efficienza dei costi per l'uso in produzione, eseguire prompt reali nel proprio carico di lavoro effettivo - non fare affidamento esclusivamente sui calcoli per token tra provider.

### Utilizzo di LLM-as-a-Judge

Cli Modelarium include scoring opzionale LLM-as-a-judge (abilitato con il flag `--judge`), che usa un LLM per valutare gli output di altri LLM. Questa è una metodologia di benchmarking standard ed è consentita ai sensi dei Termini di Servizio di tutti i provider supportati come attività di valutazione/benchmarking.

Quando si usa `--judge`, l'utente è responsabile di seguire i Termini di Servizio di ciascun provider di cui usa i modelli. I ToS di ciascun provider si applicano sia ai modelli giudicati che al modello giudice stesso.

**Avviso di pregiudizio del giudice:** I giudici LLM hanno pregiudizi documentati (auto-preferenza, preferenza per la stessa famiglia, preferenza per la verbosità). I punteggi del giudice sono un segnale utile, non verità assoluta. Usare pannelli di giudici (`--judges` con modelli multipli) per ridurre i pregiudizi.

### Rilevamento delle allucinazioni

Il preset di rilevamento delle allucinazioni è un segnale di confronto utile tra modelli, non una convalida di verità assoluta. L'accuratezza del rilevamento varia in base al modello giudice utilizzato, alla conoscenza del dominio richiesta e se i fatti di riferimento sono forniti tramite `--expected-facts`. Usarlo per il confronto della qualità relativa, non per la verifica della correttezza assoluta.

### Metodologia di confronto

Gli LLM non sono deterministici a temperatura > 0 - rieseguire lo stesso prompt può produrre output diversi. Una singola esecuzione di confronto mostra UN campione da ciascun modello, non un verdetto di qualità definitivo.

Per trarre conclusioni più affidabili:
- Usare `--temperatures 0` per output più deterministici (dove supportato)
- Eseguire lo stesso confronto 3-5 volte e cercare schemi
- Confrontare tra più prompt, non solo uno
- Usare il flag `--output json` per salvare le esecuzioni per l'analisi sistematica

## Sull'autore

Cli Modelarium è stato costruito da **[Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)**.

### Connettersi

- 💼 LinkedIn: [linkedin.com/in/lavellehatcherjr](https://linkedin.com/in/lavellehatcherjr)
- 🐙 GitHub: [github.com/lavellehatcherjr](https://github.com/lavellehatcherjr)
- 💬 Domande su questo progetto: [apri una issue](../../issues)
- 📩 Collaborazione/opportunità: contattare tramite LinkedIn

## Perché l'ho costruito

Confrontare gli output degli LLM tra provider è noioso - SDK diversi, pattern di autenticazione diversi, forme di risposta diverse, nessun modo facile per vederli affiancati con dati di costo e latenza. I rifiniti playground cloud mostrano solo un provider alla volta, e le opzioni open source disponibili o si concentrano sul routing di produzione o sono piattaforme di valutazione complete ottimizzate per i team.

Cli Modelarium è il piccolo strumento CLI focalizzato che fa una cosa bene: confronto affiancato con scoring di qualità, asserzioni, modalità batch e streaming - tutto progettato per il flusso di lavoro dello sviluppatore terminal-first.

È intenzionalmente focalizzato: nessun routing di produzione, nessuna orchestrazione di agenti, nessun fine-tuning, nessuna GUI. Solo confronto pulito e veloce dalla riga di comando.

Costruito con un'astrazione del provider modulare, esecuzione parallela, calcolo trasparente dei costi e archiviazione sicura delle chiavi tramite sistemi di portachiavi del SO per utenti locali.

## Contribuire

Issues e PR benvenuti. Vedere [CONTRIBUTING.md](CONTRIBUTING.md) per le linee guida.

Per problemi di sicurezza, vedere [SECURITY.md](SECURITY.md) - non aprire issue pubbliche per preoccupazioni di sicurezza.

## Licenza

Concesso in licenza ai sensi della [Apache License, Version 2.0](LICENSE).

Vedere il file [NOTICE](NOTICE) per i requisiti di attribuzione.

---

Costruito da [Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)

Concesso in licenza ai sensi di Apache 2.0. Issues, PR e conversazioni benvenuti.
