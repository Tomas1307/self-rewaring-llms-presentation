# `wiki/output/` — Explicación didáctica de Self-Rewarding LLMs

> **Audiencia**: experto en NLP (transformers, SFT, sampling, MLE) que no conoce RL formal. Cero conocimiento previo de RLHF, DPO, PPO, IRL.
> **Objetivo**: poder defender la presentación entendiendo cada concepto, fórmula y trade-off.

## Cómo leer este material

**Orden recomendado**: secuencial 00 → 14. Cada capítulo asume los anteriores. Saltarse capítulos solo si ya dominas el tema.

**Estilo**: cada capítulo construye intuición primero, luego fórmula, luego pseudocódigo o ejemplo numérico cuando aplica.

## Mapa de capítulos

### Parte I — Fundamentos
- [00-fundamentos-rl.md](00-fundamentos-rl.md) — Refresh de RL para alguien que viene de NLP. MDP, policy, value function, policy gradient, on-policy vs off-policy, KL divergence, PPO. Larga pero necesaria.
- [00b-informacion-y-entropia.md](00b-informacion-y-entropia.md) — **Teoría de la información**: entropía, cross-entropy, KL (forward vs reverse), principio MaxEnt, distribución de Boltzmann, función de partición. Sin esto, la derivación de DPO y la equivalencia IRL ↔ DPO quedan opacas.

> **Apoyo para dudas puntuales**: si al leer estos fundamentos las confusiones básicas (mapping LLM↔RL, política vs pesos, logits vs política, qué es V(s), por qué normalizar advantage, de dónde sale el reward) no terminan de despegar, consulta [../dudas/01-mapping-llm-rl.md](../dudas/01-mapping-llm-rl.md) — folder separado con explicaciones conversacionales paso a paso.

### Parte II — El pipeline clásico
- [01-rlhf.md](01-rlhf.md) — Reinforcement Learning from Human Feedback. El paradigma de InstructGPT/ChatGPT. SFT → reward model → PPO.
- [02-dpo.md](02-dpo.md) — Direct Preference Optimization. La derivación clave que vuelve a RLHF en un problema supervisado.
- [03-rlaif.md](03-rlaif.md) — RL from AI Feedback + Constitutional AI. Reemplazar humanos por LLM.
- [04-llm-as-judge.md](04-llm-as-judge.md) — El paradigma del juez LLM. Rúbrica, prompts, sesgos.

### Parte III — Marco teórico transversal
- [05-irl.md](05-irl.md) — Inverse RL. De Ng-Russell 2000 a MaxEnt. Por qué DPO es IRL implícito.

### Parte IV — El paper ancla y su descendencia directa
- [06-self-rewarding.md](06-self-rewarding.md) — Yuan et al. 2024. **El paper central**.
- [07-meta-rewarding.md](07-meta-rewarding.md) — Wu et al. 2024. Extensión con meta-juez.

### Parte V — El ecosistema 2024-2025
- [08-variantes-self-play.md](08-variantes-self-play.md) — SPIN, SPPO, DNO, DICE, CREAM, SCIR, ScPO, SER. Refinamientos.
- [09-self-rewarding-razonamiento.md](09-self-rewarding-razonamiento.md) — Process-SRLM, LaTRO, SAR, CoVo, Self-Rew-Correction. Adaptaciones para math.
- [10-self-rewarding-multimodal.md](10-self-rewarding-multimodal.md) — CSR, Vision-SR1. Extensión a vision-language.
- [11-self-rewarding-autonomia.md](11-self-rewarding-autonomia.md) — Self-Taught Evaluators, RLSR, Multi-Agent Evolve. Llevar la autonomía al límite.

### Parte VI — Crítica y comparación
- [12-fallos-y-criticas.md](12-fallos-y-criticas.md) — Temporal SR, Can LRMs Self-Train, mecanismos causales del colapso.
- [13-grpo-deepseek.md](13-grpo-deepseek.md) — El paradigma alternativo: rewards verificables + GRPO + R1.
- [14-comparacion.md](14-comparacion.md) — Tabla canónica, ¿cuándo usar qué?, las dos dimensiones de la evolución.

## Convenciones

- **Fórmulas**: cada símbolo se define al introducirlo. Se explica la intuición antes de la matemática.
- **Pseudocódigo**: en bloques de código sin sintaxis específica.
- **Ejemplos numéricos**: cuando ayudan a fijar una fórmula.
- **Cajitas de "intuición"** y **cajitas de "fórmula"** marcadas con `> 💡` y `> 📐`.
- **Cajas "trampa"** con `> ⚠️` para gotchas frecuentes.
- **Referencias al wiki técnico**: cuando un tema tiene su página densa en `concepts/`, `summaries/` o `synthesis/`, se enlaza al final del capítulo.

## Qué NO encontrarás aquí

- Código ejecutable de PyTorch / TRL / Hugging Face.
- Hiperparámetros listos para reproducir experimentos (esos están en `sources.md` §Hiperparámetros verificados).
- Una historia exhaustiva del campo.

## Sugerencia de lectura por sesión

**Sesión 1 (2-3 h)**: 00 + 01. Fundamentos RL + RLHF. Si vienes solo con NLP, este es el bloque más denso.

**Sesión 2 (1-2 h)**: 02 + 04. DPO + LLM-as-Judge. La maquinaria que usa Self-Rewarding.

**Sesión 3 (1-2 h)**: 03 + 05. RLAIF + IRL. Contexto teórico.

**Sesión 4 (2 h)**: 06 + 07. Self-Rewarding + Meta-Rewarding. El corazón.

**Sesión 5 (1-2 h)**: 12 + 14. Crítica + comparación. Lo necesario para defender.

**Opcional**: 08-11 + 13. Ecosistema completo y paradigma alternativo.
