---
name: zheng-2023-llm-as-judge
description: Resumen de Judging LLM-as-a-Judge (Zheng et al. 2023) — fundamento empírico del paradigma + diagnóstico de sesgos, NeurIPS 2023.
metadata:
  type: summary
  tags: [llm-as-a-judge, evaluacion, sesgos, mt-bench, chatbot-arena]
  last_updated: 2026-05-12
---

# Zheng et al. (2023) — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

## Cita

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica (UC Berkeley / LMSYS). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. arXiv:2306.05685, junio 2023. **NeurIPS 2023 Datasets & Benchmarks Track**.

- **arXiv**: https://arxiv.org/abs/2306.05685

## Resumen técnico

Primer estudio sistemático del uso de un LLM fuerte (GPT-4) como **evaluador** de chats abiertos. Mide acuerdo con humanos y **diagnostica sesgos**. Introduce dos benchmarks que se vuelven estándar:

1. **MT-Bench**: 80 questions multi-turn cubriendo 8 categorías (escritura, juego de rol, razonamiento, matemáticas, código, extracción, STEM, humanidades).
2. **Chatbot Arena**: plataforma open-source de comparación pairwise crowdsourceada.

**Hallazgos principales**:

- GPT-4 como juez acuerda con humanos > 80 % del tiempo — **comparable al acuerdo inter-humano**.
- Cuando GPT-4 prefiere una respuesta sobre otra, los humanos suelen estar de acuerdo.
- Esto **valida** LLM-as-a-Judge como sustituto barato y escalable de evaluación humana.

**Sesgos identificados** (parcialmente cuantificables y mitigables):

| Sesgo | Descripción | Mitigación parcial |
|---|---|---|
| **Posicional** | Preferencia por la primera o última opción presentada. | Swap aleatorio de posiciones + promediar. |
| **Verbosidad** | Preferencia por respuestas más largas. | Length-control normalization (AlpacaEval 2.0 LC). |
| **Self-enhancement** | Preferencia por respuestas del propio modelo o familia. | Usar juez de familia distinta; panel de jueces. |
| **Limitaciones de razonamiento** | El juez no puede evaluar correctamente problemas matemáticos o de razonamiento profundo que él mismo no puede resolver. | Usar juez más capaz; verificación independiente. |

## Aporte clave para la presentación

Tres aportes complementarios:

1. **Fundamento empírico del paradigma LLM-as-a-Judge** que [[summaries/yuan-2024-self-rewarding]] internaliza. Self-Rewarding asume que un LLM puede juzgar de manera útil — Zheng 2023 es el paper que lo demuestra empíricamente.

2. **Catálogo de sesgos del juez**. Para Self-Rewarding, el **self-enhancement bias** es especialmente crítico: el juez **es** el mismo modelo entrenado, así que el sesgo se amplifica al máximo. Ver [[synthesis/diagnostico-fallos]] §2.4.

3. **Benchmarks estándar** (MT-Bench, AlpacaEval implícito) usados por Yuan 2024, Wu 2024 Meta-Rewarding, y la literatura posterior. Conocerlos es necesario para entender los números reportados.

## Sin fórmulas centrales

(Paper empírico — no introduce nueva matemática. Es un estudio de medición.)

## Resultados empíricos clave

- **GPT-4 vs humanos**: > 80 % de acuerdo en preferencia pairwise.
- **Inter-human agreement**: similar al GPT-4-human, lo que implica que GPT-4 está **al nivel del ruido humano**.
- **GPT-3.5 / Claude como juez**: significativamente peor que GPT-4 — el sesgo de auto-preferencia es más fuerte y la habilidad de razonamiento limita su utilidad.

## Limitaciones

1. **El sesgo de self-enhancement no puede eliminarse sin un juez externo independiente**. En Self-Rewarding no hay juez externo — el sesgo se vuelve un problema estructural, no operacional.
2. **GPT-4 es opaco** (modelo cerrado). Replicar la metodología con modelos abiertos da resultados peores.
3. **Multi-turn es difícil**: el juez tiene dificultad evaluando coherencia a través de varios turnos.

## Implicaciones para Self-Rewarding

El paper establece dos cosas simultáneamente:

- **A favor**: un LLM fuerte puede juzgar útilmente → el paradigma Self-Rewarding tiene cimientos.
- **En contra**: los sesgos del juez son sistemáticos → Self-Rewarding hereda esos sesgos y los amplifica al ser juez = generador.

Este doble filo es el corazón de la crítica honesta al paradigma. Combinarlo con [[summaries/wang-2025-temporal-sr]] (saturación) y el paper *Self-Preference Bias in LLM-as-a-Judge* (arXiv:2410.21819, evidencia cuantitativa) da un cuerpo coherente de evidencia.

## Conexiones en el wiki

- [[concepts/llm-as-a-judge]] — paradigma detallado, rúbrica, prompts.
- [[summaries/yuan-2024-self-rewarding]] — usa LLM-as-a-Judge internamente.
- [[summaries/wu-2024-meta-rewarding]] — introduce un meta-judge para mitigar sesgos del juez.
- [[entities/mt-bench]] — benchmark introducido aquí (WIP).
- [[entities/alpaca-eval]] — benchmark relacionado (WIP).
- [[synthesis/diagnostico-fallos]] §2.4 — el self-enhancement bias en el contexto de Self-Rewarding.
