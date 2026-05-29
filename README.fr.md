<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cli-modelarium-wordmark-dark.svg">
  <img alt="cli modelarium" src="docs/assets/cli-modelarium-wordmark-light.svg" width="420">
</picture>

Lire ceci dans d'autres langues : [English](README.md) | [日本語](README.ja.md) | [Español](README.es.md) | [한국어](README.ko.md) | [中文](README.zh.md) | [Deutsch](README.de.md) | [Português](README.pt.md) | [Italiano](README.it.md)

Note : Ce README est traduit à des fins d'accessibilité. L'outil CLI Cli Modelarium lui-même ne produit que des sorties en anglais. Toutes les commandes, messages d'erreur et sorties restent en anglais quelle que soit la locale du système.

> Note: Features added after v0.1.0 (`--runs` in v0.1.1, statistical significance in v0.1.2, confidence intervals/paired tests/McNemar in v0.1.3) are documented in English only — translations pending.

> Comparez les sorties de LLM côte à côte depuis votre terminal - 8 fournisseurs cloud + modèles locaux, avec streaming parallèle, évaluation par lots, scoring LLM-as-judge, détection d'hallucinations et assertions prêtes pour CI/CD.

[![CI](https://github.com/lavellehatcherjr/Cli-Modelarium/actions/workflows/ci.yml/badge.svg)](https://github.com/lavellehatcherjr/Cli-Modelarium/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cli-modelarium)](https://pypi.org/project/cli-modelarium/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Platforms](https://img.shields.io/badge/platforms-Mac%20%7C%20Windows%20%7C%20Linux-lightgrey)](#)

<p align="center">
  <img src="docs/assets/cli-modelarium-demo.png" alt="Cli Modelarium help output showing the banner and available commands" width="520">
</p>

## Ce que ça fait

**Cli Modelarium** est un outil en ligne de commande soigné pour comparer les sorties de LLM entre fournisseurs, modèles, prompts système et températures - avec streaming parallèle en direct, évaluation par lots, tests déterministes et scoring de qualité intégrés.

Utile pour évaluer quel modèle convient à votre tâche spécifique, exécuter des tests de régression de prompts en CI/CD, comparer des modèles locaux aux APIs cloud ou construire des jeux de données d'évaluation - le tout depuis une seule commande de terminal.

## Démarrage rapide

```bash
pip install cli-modelarium

# Configurer les clés API (enregistrées de manière sécurisée dans le trousseau de votre OS)
cli-modelarium configure

# Exécuter votre première comparaison
cli-modelarium "Explain quantum computing in one sentence" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro \
  --temperatures 0,0.7
```

C'est tout. Vous verrez les trois modèles diffuser leurs réponses en direct en parallèle, avec la latence, les nombres de tokens et le coût affichés dans un tableau de comparaison clair.

## Fonctionnalités

### 🤖 Fournisseurs (8 cloud + locaux illimités)

- **Fournisseurs cloud :** OpenAI, Anthropic, Google (Gemini), xAI (Grok), DeepSeek, Mistral, Groq, OpenRouter
- **Modèles locaux :** Ollama, LM Studio, vLLM, llama.cpp - tout serveur local compatible OpenAI
- Mélangez les modèles locaux et cloud dans la même comparaison
- Sélection de modèle configurable par appel (pas de listes codées en dur)

### ⚡ Streaming parallèle

- Affichage en direct token par token sur tous les modèles simultanément
- Suivi du Time-to-First-Token (TTFT) par modèle
- Voyez quel modèle termine en premier, observez les sorties diverger en temps réel
- Streams depuis les 8 fournisseurs (SSE en interne)

### 📊 Modes de comparaison multiples

- **Un prompt vs. plusieurs modèles** - comparaisons rapides « lequel est le meilleur ? »
- **Un prompt vs. plusieurs températures** - voyez comment l'aléatoire affecte la sortie
- **Plusieurs prompts système vs. un prompt utilisateur** - tests A/B de prompt engineering
- **Mode par lots** - multi-prompt × multi-modèle pour un vrai travail d'évaluation
- **Comparaisons local vs. cloud** - quantifiez l'écart (ou son absence)

### 🧪 Fonctionnalités d'évaluation

- **Assertions déterministes** - 10 types d'assertions (`contains`, `regex`, `json_valid`, `json_schema`, `max_length_chars`, `latency_under`, `cost_under` et plus) avec sortie pass/fail et codes de sortie CI
- **Scoring LLM-as-a-judge** - Utilisez un LLM pour scorer les sorties d'autres LLMs sur des critères de qualité
- **Panels de juges** - Plusieurs juges moyennent les scores pour une évaluation moins biaisée
- **Preset de détection d'hallucinations** - Critères prêts à l'emploi pour la vérification de la précision factuelle
- **Critères personnalisés** - Définissez vos propres grilles de scoring
- **Auto-omission de l'auto-évaluation** - Les modèles juges sont automatiquement omis quand ils sont aussi jugés

### 💾 Formats de sortie

- **Terminal en direct** - Panneaux basés sur Rich avec barres de progression et affichage streaming
- **CSV** - Compatible tableurs (ouvrir dans Excel, Google Sheets, pandas)
- **JSON** - Structuré pour scripts et pipelines
- **Markdown** - Beaux tableaux pour articles de blog et rapports
- **Codes de sortie** - 0/1/2 reflétant le statut pass/fail pour CI/CD

### 💰 Transparence des coûts

- Coût par appel affiché à partir de l'usage rapporté par chaque fournisseur
- Résumé du coût total par comparaison
- Coût du juge affiché séparément quand LLM-as-judge est activé
- Les modèles locaux affichés comme « Free »
- Flag `--max-cost` pour éviter les factures surprises

### 🔒 Sécurité

- Clés API stockées dans le trousseau natif de l'OS via `keyring` (Mac Keychain, Windows Credential Manager, Linux Secret Service)
- La validation du format détecte les erreurs de collage avant le stockage
- La rédaction des messages d'erreur empêche la fuite de clés dans les tracebacks
- Validation localhost uniquement pour les URLs de modèles locaux
- `SECURITY.md` avec politique de divulgation responsable

### 🛡️ Gestion des limites de débit

- Limites de concurrence par fournisseur (défaut 5) respectant toutes les références de tiers
- Réessai automatique 429 avec backoff exponentiel
- Le 529 « overloaded » d'Anthropic est géré séparément des limites de débit
- Flag `--concurrency` pour les utilisateurs avancés sur des tiers supérieurs
- Échec gracieux par modèle (les autres modèles continuent)

### 🌐 Multiplateforme

- Fonctionne de manière identique sur macOS, Windows (10+ et ARM) et Linux
- Toutes les E/S de fichiers utilisent `pathlib` + encodage UTF-8 explicite
- L'écriture CSV utilise `newline=""` pour la compatibilité Windows
- Python 3.11+ requis

### 📋 Expérience développeur

- **Binaire CLI unique** - `pip install cli-modelarium` et c'est tout
- **UI soignée basée sur Rich** - Polissage de terminal au niveau Claude Code
- **Sortie JSON** - Pipez dans n'importe quoi (`jq`, scripts, monitoring)
- **Prêt pour CI/CD** - Codes de sortie, sortie structurée, exemple GitHub Actions inclus
- **Licence Apache 2.0** - Utilisez dans n'importe quel projet, commercial ou autre

## Exemples

### Comparer 3 modèles sur une tâche de programmation

```bash
cli-modelarium "Write a Python function to find the longest palindromic substring" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro
```

### Évaluation par lots avec assertions

Créez `eval.json` :

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

Exécutez-le :

```bash
cli-modelarium batch eval.json \
  --models gpt-5.5,claude-opus-4-7 \
  --output results.csv
```

### Scorer les sorties avec un juge LLM

```bash
cli-modelarium "Explain recursion in one paragraph" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro,local/llama-3.3-70b \
  --judge claude-opus-4-7 \
  --judge-criteria "accuracy,clarity,brevity"
```

### Détecter les hallucinations contre des faits connus

```bash
cli-modelarium "Tell me about the Eiffel Tower" \
  --models gpt-5.5,claude-opus-4-7 \
  --judge claude-opus-4-7 \
  --check-hallucination \
  --expected-facts "Built 1887-1889,Located in Paris France,Designed by Gustave Eiffel"
```

### Comparer un modèle local aux APIs cloud

```bash
# Démarrer Ollama d'abord : ollama run llama3.3
cli-modelarium "Summarize the key features of microservices architecture" \
  --models local/llama-3.3-70b,gpt-5.5,claude-opus-4-7
```

### Exécuter en CI/CD (exemple GitHub Actions)

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

La commande se termine avec le code 1 si le taux de réussite tombe sous 90 %, faisant échouer le build.

## Configuration

### Clés API

Cli Modelarium stocke les clés API dans le trousseau natif de votre OS (Mac Keychain, Windows Credential Manager ou Linux Secret Service via `keyring`). Les clés ne touchent jamais le disque en clair.

```bash
# Configuration interactive (recommandée)
cli-modelarium configure

# Ou définir individuellement
cli-modelarium keys set openai
cli-modelarium keys set anthropic
cli-modelarium keys set google

# Vérifier quelles clés sont configurées
cli-modelarium keys list

# Supprimer une clé
cli-modelarium keys delete openai
```

Vous pouvez aussi utiliser les variables d'environnement (utile pour CI/CD) :

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

Les variables d'environnement ont priorité sur le stockage du trousseau.

### Modèles locaux (Ollama, LM Studio, etc.)

Les modèles locaux fonctionnent via des endpoints compatibles OpenAI - pas de clés API nécessaires. L'outil détecte automatiquement le port Ollama par défaut.

```bash
# Défaut : suppose Ollama sur localhost:11434
cli-modelarium "test" --models local/llama-3.3

# Utiliser LM Studio à la place
cli-modelarium "test" --models local/qwen-3-32b --local-url http://localhost:1234/v1

# Enregistrer une URL locale personnalisée par défaut
cli-modelarium keys set local --base-url http://localhost:1234/v1
```

## Fournisseurs supportés

| Fournisseur | Clés API Requises | Streaming | Suivi des Coûts |
|----------|-----------------|-----------|---------------|
| OpenAI (GPT-5, GPT-5 mini, o3, o4-mini, etc.) | ✅ | ✅ | ✅ |
| Anthropic (Claude Opus 4.7, Sonnet 4.6, Haiku 4.5, etc.) | ✅ | ✅ | ✅ |
| Google (Gemini 3.1 Pro, Gemini 3 Flash, etc.) | ✅ | ✅ | ✅ |
| xAI (Grok 4.1, etc.) | ✅ | ✅ | ✅ |
| DeepSeek (V3, R1) | ✅ | ✅ | ✅ |
| Mistral (Large, Medium, Small) | ✅ | ✅ | ✅ |
| Groq (Llama, Mixtral, etc.) | ✅ | ✅ | ✅ |
| OpenRouter (n'importe quel modèle sur la plateforme) | ✅ | ✅ | ✅ |
| **Local : Ollama** | ❌ | ✅ | Gratuit |
| **Local : LM Studio** | ❌ | ✅ | Gratuit |
| **Local : vLLM** | ❌ | ✅ | Gratuit |
| **Local : llama.cpp server** | ❌ | ✅ | Gratuit |

Exécutez `cli-modelarium list-models` pour voir tous les modèles actuellement supportés.

## Comment ça marche

Cli Modelarium utilise une couche d'abstraction de fournisseur modulaire qui masque les différences d'API entre le tableau `messages` d'OpenAI, le paramètre `system` de niveau supérieur d'Anthropic, le `system_instruction` de Google et d'autres. Chaque fournisseur implémente la même interface de streaming asynchrone, donc la CLI peut tous les exécuter en parallèle avec `asyncio.gather()`.

Les calculs de coût proviennent du champ `usage` rapporté par chaque fournisseur (tokens d'entrée, tokens de sortie, tokens en cache) multiplié par les constantes de tarification actuelles. Les données de tarification ont été vérifiées depuis la documentation officielle des fournisseurs le **25 mai 2026** - voir [Notes et limitations](#notes-et-limitations) pour les mises en garde.

Pour les modèles locaux, le même SDK Python OpenAI est utilisé avec une `base_url` personnalisée, puisque Ollama, LM Studio, vLLM et llama.cpp exposent tous des endpoints REST compatibles OpenAI.

## Notes et limitations

### Données de tarification

Toutes les tarifications intégrées dans Cli Modelarium ont été vérifiées depuis la documentation officielle des fournisseurs le **25 mai 2026**. Les tarifs des LLM changent fréquemment (parfois mensuellement). L'outil affiche la date `pricing_as_of` dans chaque sortie. Vérifiez toujours par rapport à la page officielle de tarification de chaque fournisseur avant de vous fier aux calculs de coûts pour la budgétisation ou les décisions de production.

### Limites de débit

La gestion des limites de débit et les paramètres de concurrence par défaut par fournisseur sont basés sur les limites de débit des fournisseurs vérifiées le **25 mai 2026**. Les limites de votre tier spécifique peuvent différer des valeurs par défaut supposées ici. Vérifiez vos limites actuelles par rapport au tableau de bord officiel du fournisseur avant de bâtir des hypothèses de capacité de production.

### Disponibilité des modèles

Les modèles supportés par Cli Modelarium reflètent ce que les fournisseurs proposaient le **25 mai 2026**. Les fournisseurs publient régulièrement de nouveaux modèles, déprécient les anciens et ajustent les capacités. Si un modèle dans le registre ne fonctionne plus, exécutez `cli-modelarium list-models` et consultez la documentation du fournisseur.

### Pas une passerelle de qualité production

Cli Modelarium est conçu pour l'évaluation et la comparaison - exécution de tests ad-hoc côte à côte entre fournisseurs depuis un terminal de développeur. Ce n'est PAS une passerelle d'inférence de production. Si vous avez besoin de routage à l'échelle de production, d'équilibrage de charge, de chaînes de fallback ou d'inférence gérée par SLA, cherchez des outils construits spécifiquement à cet effet.

### Comparaisons de nombre de tokens entre fournisseurs

Les nombres de tokens affichés dans les résultats sont rapportés par l'API de chaque fournisseur. Différents fournisseurs utilisent différents tokenizers, donc les « tokens de sortie » ne sont pas directement comparables entre fournisseurs pour le même texte. Si vous comparez l'efficacité des coûts pour un usage en production, exécutez de vrais prompts dans votre charge de travail réelle - ne vous fiez pas uniquement aux calculs par token entre fournisseurs.

### Utilisation de LLM-as-a-Judge

Cli Modelarium inclut un scoring LLM-as-a-judge optionnel (activé avec le flag `--judge`), qui utilise un LLM pour évaluer les sorties d'autres LLMs. C'est une méthodologie de benchmarking standard et c'est autorisé par les Conditions d'utilisation de tous les fournisseurs supportés en tant qu'activité d'évaluation/benchmarking.

En utilisant `--judge`, vous êtes responsable de respecter les Conditions d'utilisation de chaque fournisseur dont vous utilisez les modèles. Les ToS de chaque fournisseur s'appliquent à la fois aux modèles évalués et au modèle juge lui-même.

**Avis sur le biais du juge :** Les juges LLM ont des biais documentés (préférence pour soi, préférence pour la même famille, préférence pour la verbosité). Les scores du juge sont un signal utile, pas une vérité absolue. Utilisez des panels de juges (`--judges` avec plusieurs modèles) pour réduire les biais.

### Détection d'hallucinations

Le preset de détection d'hallucinations est un signal de comparaison utile entre modèles, pas une validation de vérité absolue. La précision de la détection varie selon le modèle juge utilisé, les connaissances du domaine requises et si des faits de référence sont fournis via `--expected-facts`. Utilisez-le pour la comparaison de qualité relative, pas pour la vérification d'exactitude absolue.

### Méthodologie de comparaison

Les LLMs sont non déterministes à température > 0 - réexécuter le même prompt peut produire des sorties différentes. Une seule exécution de comparaison vous montre UN échantillon de chaque modèle, pas un verdict de qualité définitif.

Pour tirer des conclusions plus fiables :
- Utilisez `--temperatures 0` pour des sorties plus déterministes (où c'est supporté)
- Exécutez la même comparaison 3-5 fois et cherchez des motifs
- Comparez sur plusieurs prompts, pas seulement un
- Utilisez le flag `--output json` pour sauvegarder les exécutions pour analyse systématique

## À propos de l'auteur

Cli Modelarium a été construit par **[Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)**.

### Se connecter

- 💼 LinkedIn : [linkedin.com/in/lavellehatcherjr](https://linkedin.com/in/lavellehatcherjr)
- 🐙 GitHub : [github.com/lavellehatcherjr](https://github.com/lavellehatcherjr)
- 💬 Questions sur ce projet : [ouvrir une issue](../../issues)
- 📩 Collaboration/opportunités : contactez via LinkedIn

## Pourquoi je l'ai construit

Comparer les sorties de LLM entre fournisseurs est fastidieux - différents SDKs, différents patterns d'authentification, différentes formes de réponses, aucun moyen facile de les voir côte à côte avec des données de coût et de latence. Les playgrounds cloud soignés ne montrent qu'un seul fournisseur à la fois, et les options open source disponibles se concentrent soit sur le routage de production, soit sont des plateformes d'évaluation complètes optimisées pour les équipes.

Cli Modelarium est le petit outil CLI focalisé qui fait bien une chose : comparaison côte à côte avec scoring de qualité, assertions, mode par lots et streaming - le tout conçu pour le workflow développeur centré sur le terminal.

C'est intentionnellement focalisé : pas de routage de production, pas d'orchestration d'agent, pas de fine-tuning, pas de GUI. Juste une comparaison propre et rapide depuis la ligne de commande.

Construit avec une abstraction de fournisseur modulaire, exécution parallèle, calcul de coût transparent et stockage sécurisé des clés via les systèmes de trousseau OS pour les utilisateurs locaux.

## Contribuer

Issues et PRs bienvenus. Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour les directives.

Pour les problèmes de sécurité, veuillez consulter [SECURITY.md](SECURITY.md) - ne déposez pas d'issues publiques pour des préoccupations de sécurité.

## Licence

Sous licence [Apache License, Version 2.0](LICENSE).

Voir le fichier [NOTICE](NOTICE) pour les exigences d'attribution.

---

Construit par [Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)

Sous licence Apache 2.0. Issues, PRs et conversations bienvenus.
