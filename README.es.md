<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cli-modelarium-wordmark-dark.svg">
  <img alt="cli modelarium" src="docs/assets/cli-modelarium-wordmark-light.svg" width="420">
</picture>

Leer esto en otros idiomas: [English](README.md) | [日本語](README.ja.md) | [Français](README.fr.md) | [한국어](README.ko.md) | [中文](README.zh.md) | [Deutsch](README.de.md) | [Português](README.pt.md) | [Italiano](README.it.md)

Nota: Este README está traducido por accesibilidad. La herramienta CLI Cli Modelarium en sí solo produce salida en inglés. Todos los comandos, mensajes de error y salidas permanecen en inglés independientemente de la configuración regional del sistema.

> Note: Features added after v0.1.0 (`--runs` in v0.1.1, statistical significance in v0.1.2, confidence intervals/paired tests/McNemar in v0.1.3) are documented in English only — translations pending.

> Compare salidas de LLM lado a lado desde su terminal - 10 proveedores en la nube + modelos locales, con streaming paralelo, evaluación por lotes, puntuación LLM-as-judge, detección de alucinaciones y aserciones listas para CI/CD.

[![CI](https://github.com/lavellehatcherjr/Cli-Modelarium/actions/workflows/ci.yml/badge.svg)](https://github.com/lavellehatcherjr/Cli-Modelarium/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cli-modelarium)](https://pypi.org/project/cli-modelarium/)
[![Downloads](https://img.shields.io/pepy/dt/cli-modelarium)](https://pepy.tech/project/cli-modelarium)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Platforms](https://img.shields.io/badge/platforms-Mac%20%7C%20Windows%20%7C%20Linux-lightgrey)](#)

<p align="center">
  <img src="docs/assets/cli-modelarium-demo.png" alt="Cli Modelarium help output showing the banner and available commands" width="520">
</p>

## Qué hace

**Cli Modelarium** es una herramienta de línea de comandos pulida para comparar salidas de LLM entre proveedores, modelos, prompts de sistema y temperaturas - con streaming paralelo en vivo, evaluación por lotes, pruebas deterministas y puntuación de calidad integrados.

Útil para evaluar qué modelo se adapta a su tarea específica, ejecutar pruebas de regresión de prompts en CI/CD, comparar modelos locales contra APIs en la nube o construir conjuntos de datos de evaluación - todo desde un solo comando de terminal.

## Inicio rápido

```bash
pip install cli-modelarium

# Configurar las claves de API (se guardan de forma segura en el llavero del SO)
cli-modelarium configure

# Ejecutar su primera comparación
cli-modelarium "Explain quantum computing in one sentence" \
  --models gpt-5.5,claude-opus-4-8,gemini-3.1-pro-preview \
  --temperatures 0,0.7
```

Eso es todo. Verá los tres modelos transmitir sus respuestas en vivo en paralelo, con la latencia, los conteos de tokens y el costo mostrados en una tabla de comparación limpia.

## Características

### 🤖 Proveedores (10 en la nube + locales ilimitados)

- **Proveedores en la nube:** OpenAI, Anthropic, Google (Gemini), xAI (Grok), DeepSeek, Mistral, Groq, OpenRouter, Alibaba (DashScope), Z.AI (GLM)
- **Modelos locales:** Ollama, LM Studio, vLLM, llama.cpp - cualquier servidor compatible con OpenAI que se ejecute en localhost
- Combine modelos locales y en la nube en la misma comparación
- Elija cualquier ID de modelo registrado por llamada - sin limitarse a los atajos de grupo integrados

### ⚡ Streaming paralelo

- Visualización token por token en vivo en todos los modelos simultáneamente
- Seguimiento del Time-to-First-Token (TTFT) por modelo
- Vea qué modelo termina primero, observe cómo divergen las salidas en tiempo real
- Streams desde los 10 proveedores (SSE por debajo)

### 📊 Múltiples modos de comparación

- **Un solo prompt vs. múltiples modelos** - comparaciones rápidas de "¿cuál es mejor?"
- **Un solo prompt vs. múltiples temperaturas** - vea cómo la aleatoriedad afecta la salida
- **Múltiples prompts de sistema vs. un prompt de usuario** - pruebas A/B de ingeniería de prompts
- **Modo por lotes** - multi-prompt × multi-modelo para trabajo de evaluación real
- **Comparaciones local vs. nube** - cuantifique la brecha (o la falta de ella)

### 🧪 Características de evaluación

- **Aserciones deterministas** - 10 tipos de aserciones (`contains`, `regex`, `json_valid`, `json_schema`, `max_length_chars`, `latency_under`, `cost_under` y más) con salida de aprobado/fallado y códigos de salida para CI
- **Puntuación LLM-as-a-judge** - Use un LLM para puntuar las salidas de otros según criterios de calidad
- **Paneles de jueces** - Múltiples jueces promedian puntuaciones para una evaluación menos sesgada
- **Preset de detección de alucinaciones** - Criterios listos para usar para la verificación de precisión factual
- **Criterios personalizados** - Defina sus propias rúbricas de puntuación
- **Auto-omisión de autoevaluación** - Los modelos jueces se omiten automáticamente cuando también están siendo juzgados

### 💾 Formatos de salida

- **Terminal en vivo** - Paneles basados en Rich con barras de progreso y visualización de streaming
- **CSV** - Compatible con hojas de cálculo (abrir en Excel, Google Sheets, pandas)
- **JSON** - Estructurado para scripts y pipelines
- **Markdown** - Tablas elegantes para publicaciones de blog e informes
- **Códigos de salida** - 0/1/2 que reflejan el estado de aprobado/fallado para CI/CD

### 💰 Transparencia de costos

- Costo por llamada mostrado desde el uso reportado por cada proveedor
- Resumen del costo total por comparación
- Costo del juez mostrado por separado cuando LLM-as-judge está habilitado
- Los modelos locales se muestran como "Free"
- Flag `--max-cost` para prevenir facturas sorpresa

### 🔒 Seguridad

- Las claves de API se almacenan en el llavero nativo del SO a través de `keyring` (Mac Keychain, Windows Credential Manager, Linux Secret Service)
- La validación de formato detecta errores de pegado antes del almacenamiento
- La redacción de mensajes de error previene la fuga de claves en los tracebacks
- Validación exclusiva de localhost para URLs de modelos locales
- `SECURITY.md` con política de divulgación responsable

### 🛡️ Manejo de límites de velocidad

- Límites de concurrencia por proveedor (predeterminado 5) respetan todas las líneas base de niveles
- Reintento automático de 429 con retroceso exponencial
- El 529 "overloaded" de Anthropic se maneja por separado de los límites de velocidad
- Flag `--concurrency` para usuarios avanzados en niveles superiores
- Fallo elegante por modelo (otros modelos continúan)
- Los límites de velocidad del nivel gratuito de DashScope y del Qwen insignia (qwen3.7-max) son más estrictos que los de la mayoría de los proveedores; reduzca `--concurrency` si encuentra errores 429.

### 🌐 Multiplataforma

- Funciona de forma idéntica en macOS, Windows (10+ y ARM) y Linux
- Toda la E/S de archivos usa `pathlib` + codificación UTF-8 explícita
- La escritura de CSV usa `newline=""` para compatibilidad con Windows
- Se requiere Python 3.11+

### 📋 Experiencia del desarrollador

- **Binario CLI único** - `pip install cli-modelarium` y listo
- **UI pulida basada en Rich** - Pulido de terminal al nivel de Claude Code
- **Salida JSON** - Conecte a cualquier cosa (`jq`, scripts, monitoreo)
- **Listo para CI/CD** - Códigos de salida, salida estructurada, ejemplo de GitHub Actions incluido
- **Licenciado bajo Apache 2.0** - Úselo en cualquier proyecto, comercial o de otro tipo

## Ejemplos

### Comparar 3 modelos en una tarea de codificación

```bash
cli-modelarium "Write a Python function to find the longest palindromic substring" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro-preview
```

### Evaluación por lotes con aserciones

Cree `eval.json`:

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

Ejecútelo:

```bash
cli-modelarium batch eval.json \
  --models gpt-5.5,claude-opus-4-7 \
  --output results.csv
```

### Puntuar salidas con un juez LLM

```bash
cli-modelarium "Explain recursion in one paragraph" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro-preview,local/llama-3.3-70b \
  --judge claude-opus-4-7 \
  --judge-criteria "accuracy,clarity,brevity"
```

### Detectar alucinaciones contra hechos conocidos

```bash
cli-modelarium "Tell me about the Eiffel Tower" \
  --models gpt-5.5,claude-opus-4-7 \
  --judge claude-opus-4-7 \
  --check-hallucination \
  --expected-facts "Built 1887-1889,Located in Paris France,Designed by Gustave Eiffel"
```

### Comparar un modelo local contra APIs en la nube

```bash
# Iniciar Ollama primero: ollama run llama3.3
cli-modelarium "Summarize the key features of microservices architecture" \
  --models local/llama-3.3-70b,gpt-5.5,claude-opus-4-7
```

### Ejecutar en CI/CD (ejemplo de GitHub Actions)

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

El comando termina con código 1 si la tasa de aprobación cae por debajo del 90%, haciendo fallar la build.

## Configuración

### Claves de API

Cli Modelarium almacena las claves de API en el llavero nativo de su SO (Mac Keychain, Windows Credential Manager o Linux Secret Service a través de `keyring`). Las claves nunca tocan el disco en texto plano.

```bash
# Configuración interactiva (recomendada)
cli-modelarium configure

# O establezca individualmente
cli-modelarium keys set openai
cli-modelarium keys set anthropic
cli-modelarium keys set google

# Comprobar qué claves están configuradas
cli-modelarium keys list

# Eliminar una clave
cli-modelarium keys delete openai
```

También puede usar variables de entorno (útil para CI/CD):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

Las variables de entorno tienen prioridad sobre el almacenamiento del llavero.

### Modelos locales (Ollama, LM Studio, etc.)

Los modelos locales funcionan a través de endpoints compatibles con OpenAI - no se necesitan claves de API. La herramienta detecta automáticamente el puerto predeterminado de Ollama.

```bash
# Predeterminado: asume Ollama en localhost:11434
cli-modelarium "test" --models local/llama-3.3

# Usar LM Studio en su lugar
cli-modelarium "test" --models local/qwen-3-32b --local-url http://localhost:1234/v1

# Guardar una URL local personalizada como predeterminada
cli-modelarium keys set local --base-url http://localhost:1234/v1
```

## Proveedores compatibles

| Proveedor | Claves de API Necesarias | Streaming | Seguimiento de Costos |
|----------|-----------------|-----------|---------------|
| OpenAI (GPT-5, GPT-5 mini, o3, o4-mini, etc.) | ✅ | ✅ | ✅ |
| Anthropic (Claude Opus 4.8, Sonnet 4.6, Haiku 4.5, etc.) | ✅ | ✅ | ✅ |
| Google (Gemini 3.5 Flash, Gemini 3.1 Pro, etc.) | ✅ | ✅ | ✅ |
| xAI (Grok 4.3, etc.) | ✅ | ✅ | ✅ |
| DeepSeek (V4 Pro, V4 Flash, etc.) | ✅ | ✅ | ✅ |
| Mistral (Large, Medium, Small) | ✅ | ✅ | ✅ |
| Groq (Llama, Mixtral, etc.) | ✅ | ✅ | ✅ |
| OpenRouter (cualquier modelo en la plataforma) | ✅ | ✅ | ✅ |
| Alibaba/DashScope (Qwen3.7 Max, Qwen3.6 Flash, Qwen3 Coder, etc.; modelos Qwen seleccionados, Internacional/Singapur) | ✅ | ✅ | ✅ |
| Z.AI/GLM (GLM-5.2, GLM-4.7, GLM-4.5 Air, etc.; compatible con OpenAI, endpoint internacional) | ✅ | ✅ | ✅ |
| **Local: Ollama** | ❌ | ✅ | Gratis |
| **Local: LM Studio** | ❌ | ✅ | Gratis |
| **Local: vLLM** | ❌ | ✅ | Gratis |
| **Local: llama.cpp server** | ❌ | ✅ | Gratis |

Ejecute `cli-modelarium list-models` para ver todos los modelos actualmente soportados.

## Grupos de modelos

En lugar de listar IDs de modelos, `--models` acepta un atajo de grupo. Los grupos se filtran según los proveedores que tenga configurados, por lo que un grupo solo ejecuta los modelos para los que realmente tiene claves.

**Grupos estáticos** (membresía fija):

| Grupo | Modelos |
|-------|---------|
| `all-premium` / `all-flagship` | gpt-5.5, claude-opus-4-8, gemini-3.1-pro-preview, grok-4.3, deepseek-v4-pro, mistral-large-latest, qwen3.7-max, glm-5.2 |
| `all-budget` | gpt-5.4-nano, claude-haiku-4-5, gemini-3.1-flash-lite, grok-4.20-0309-non-reasoning, deepseek-v4-flash, mistral-small-latest, qwen3.7-plus, glm-4.5-air |
| `all-reasoning` | o3, o4-mini, deepseek-reasoner, magistral-medium-latest, magistral-small-latest, glm-5.2 |
| `all-fast` | claude-haiku-4-5, gemini-3.5-flash, grok-4.20-0309-non-reasoning, deepseek-v4-flash, llama-3.3-70b-versatile, qwen3.6-flash, glm-5-turbo |
| `all-cheap` | gpt-4o-mini, claude-haiku-4-5, gemini-2.5-flash-lite, deepseek-v4-flash, mistral-small-latest, qwen-flash, glm-4.7-flashx |
| `all-open-weight` | gpt-oss-120b, gpt-oss-20b, llama-3.3-70b-versatile, meta-llama/llama-4-scout-17b-16e-instruct |

**Grupos dinámicos** (resueltos en tiempo de ejecución):

- `all` — todos los modelos en la nube para los que tenga una clave de API configurada (excluye los modelos locales y OpenRouter). Esto puede expandirse a muchos modelos, así que combínelo con `--max-cost`.
- `all-local` — todos los modelos reportados por su servidor local en ejecución (Ollama / LM Studio / vLLM / llama.cpp). Si no se puede acceder a ningún servidor, obtendrá un mensaje claro en lugar de un error.

```bash
cli-modelarium "Explica el teorema CAP" --models all-budget
cli-modelarium "Explica el teorema CAP" --models all --max-cost 0.50
cli-modelarium "Explica el teorema CAP" --models all-local
```

## Cómo funciona

Cli Modelarium usa una capa de abstracción de proveedor modular que oculta las diferencias de API entre el array `messages` de OpenAI, el parámetro `system` de nivel superior de Anthropic, el `system_instruction` de Google y otros. Cada proveedor implementa la misma interfaz de streaming asíncrono, por lo que la CLI puede ejecutarlos todos en paralelo con `asyncio.gather()`.

Los cálculos de costos provienen del campo `usage` reportado por cada proveedor (tokens de entrada, tokens de salida, tokens en caché) multiplicado por las constantes de precios actuales. Los datos de precios fueron verificados desde la documentación oficial del proveedor el **22 de junio de 2026** - vea [Notas y Limitaciones](#notas-y-limitaciones) para las advertencias.

Para los modelos locales, se usa el mismo SDK de Python de OpenAI con una `base_url` personalizada, ya que Ollama, LM Studio, vLLM y llama.cpp exponen endpoints REST compatibles con OpenAI.

## Notas y Limitaciones

### Datos de precios

Todos los precios incorporados en Cli Modelarium fueron verificados desde la documentación oficial del proveedor el **22 de junio de 2026**. Los precios de los LLM cambian con frecuencia (a veces mensualmente). La herramienta muestra la fecha `pricing_as_of` en cada salida. Verifique siempre con la página oficial de precios de cada proveedor antes de confiar en los cálculos de costos para presupuestos o decisiones de producción.

Los precios son la tarifa pública estándar/de lista de cada proveedor por cada 1M de tokens (no precios por lotes, prioritarios, de horas valle ni promocionales); para los modelos con niveles según el tamaño de entrada se muestra el nivel de entrada/contexto corto, y el precio en caché es la tarifa de lectura de caché. Los costos de DashScope/Qwen reflejan las tarifas sin razonamiento (la herramienta envía `enable_thinking=false`).

Ejecute `cli-modelarium pricing` (o `pricing --all`) para obtener las tarifas actuales por modelo.

### Límites de velocidad

El manejo de los límites de velocidad y las configuraciones de concurrencia predeterminadas por proveedor se basan en los límites de velocidad del proveedor verificados el **21 de junio de 2026**. Los límites de su nivel específico pueden diferir de los predeterminados asumidos aquí. Verifique sus límites actuales con el panel oficial del proveedor antes de construir suposiciones de capacidad de producción.

### Disponibilidad del modelo

Los modelos soportados por Cli Modelarium reflejan lo que los proveedores ofrecían el **21 de junio de 2026**. Los proveedores lanzan regularmente nuevos modelos, descontinúan los antiguos y ajustan las capacidades. Si un modelo en el registro ya no funciona, ejecute `cli-modelarium list-models` y consulte la documentación del proveedor.

### No es una pasarela de grado de producción

Cli Modelarium está diseñado para evaluación y comparación - ejecutando pruebas ad-hoc lado a lado entre proveedores desde una terminal de desarrollador. NO es una pasarela de inferencia de producción. Si necesita enrutamiento a escala de producción, balanceo de carga, cadenas de fallback o inferencia administrada por SLA, busque herramientas construidas específicamente para ese propósito.

### Comparaciones de conteo de tokens entre proveedores

Los conteos de tokens mostrados en los resultados son reportados por la API de cada proveedor. Diferentes proveedores usan diferentes tokenizadores, por lo que "tokens de salida" no es directamente comparable entre proveedores para el mismo texto. Si está comparando la eficiencia de costos para uso en producción, ejecute prompts reales en su carga de trabajo real - no confíe únicamente en cálculos por token entre proveedores.

### Uso de LLM-as-a-Judge

Cli Modelarium incluye una puntuación LLM-as-a-judge opcional (habilitada con el flag `--judge`), que usa un LLM para evaluar las salidas de otros LLMs. Esta es una metodología de benchmarking estándar y está permitida bajo los Términos de Servicio de todos los proveedores soportados como actividad de evaluación/benchmarking.

Al usar `--judge`, usted es responsable de cumplir con los Términos de Servicio de cada proveedor cuyos modelos use. Los ToS de cada proveedor se aplican tanto a los modelos siendo juzgados como al modelo juez en sí.

**Aviso de sesgo del juez:** Los jueces LLM tienen sesgos documentados (preferencia propia, preferencia por la misma familia, preferencia por la verbosidad). Las puntuaciones del juez son una señal útil, no una verdad fundamental. Use paneles de jueces (`--judges` con múltiples modelos) para reducir el sesgo.

### Detección de alucinaciones

El preset de detección de alucinaciones es una señal de comparación útil entre modelos, no una validación de verdad fundamental. La precisión de la detección varía según el modelo juez utilizado, el conocimiento del dominio requerido y si se proporcionan hechos de referencia a través de `--expected-facts`. Úselo para comparación de calidad relativa, no para verificación de corrección absoluta.

### Metodología de comparación

Los LLMs son no deterministas a temperatura > 0 - volver a ejecutar el mismo prompt puede producir salidas diferentes. Una sola ejecución de comparación le muestra UNA muestra de cada modelo, no un veredicto de calidad definitivo.

Para sacar conclusiones más confiables:
- Use `--temperatures 0` para salidas más deterministas (donde sea compatible)
- Ejecute la misma comparación 3-5 veces y busque patrones
- Compare entre múltiples prompts, no solo uno
- Use el flag `--output json` para guardar ejecuciones para análisis sistemático

## Sobre el autor

Cli Modelarium fue construido por **[Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)**.

### Conectar

- 💼 LinkedIn: [linkedin.com/in/lavellehatcherjr](https://linkedin.com/in/lavellehatcherjr)
- 🐙 GitHub: [github.com/lavellehatcherjr](https://github.com/lavellehatcherjr)
- 💬 Preguntas sobre este proyecto: [abra un issue](../../issues)
- 📩 Colaboración/oportunidades: contacte vía LinkedIn

## Por qué lo construí

Comparar salidas de LLM entre proveedores es tedioso - diferentes SDKs, diferentes patrones de autenticación, diferentes formas de respuesta, ninguna manera fácil de verlos lado a lado con datos de costo y latencia. Los pulidos playgrounds en la nube solo muestran un proveedor a la vez, y las opciones de código abierto disponibles se enfocan en el enrutamiento de producción o son plataformas de evaluación completas optimizadas para equipos.

Cli Modelarium es la pequeña herramienta CLI enfocada que hace bien una cosa: comparación lado a lado con puntuación de calidad, aserciones, modo por lotes y streaming - todo diseñado para el flujo de trabajo del desarrollador centrado en la terminal.

Está intencionalmente enfocado: sin enrutamiento de producción, sin orquestación de agentes, sin ajuste fino, sin GUI. Solo comparación limpia y rápida desde la línea de comandos.

Construido con una abstracción de proveedor modular, ejecución paralela, cálculo de costos transparente y almacenamiento seguro de claves a través de sistemas de llavero del SO para usuarios locales.

## Contribuyendo

Issues y PRs bienvenidos. Vea [CONTRIBUTING.md](CONTRIBUTING.md) para las pautas.

Para problemas de seguridad, por favor vea [SECURITY.md](SECURITY.md) - no presente issues públicos por preocupaciones de seguridad.

## Licencia

Licenciado bajo la [Apache License, Version 2.0](LICENSE).

Vea el archivo [NOTICE](NOTICE) para los requisitos de atribución.

---

Construido por [Lavelle Hatcher Jr](https://linkedin.com/in/lavellehatcherjr)

Licenciado bajo Apache 2.0. Issues, PRs y conversaciones bienvenidos.
