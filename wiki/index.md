# Índice del Wiki

Catálogo de todas las páginas. Cada entrada: link, resumen de una línea, estado.

**Estado**: `WIP` = en construcción · `REVISAR` = contradicción detectada · sin marca = estable.

## Concepts

Páginas conceptuales — los "bloques" que la presentación reutilizará.

- [[concepts/dpo]] — Direct Preference Optimization: derivación, pérdida, intuición del log-ratio, rol de β.
- [[concepts/self-rewarding-loop]] — Algoritmo completo del loop iterativo (Yuan 2024): seed → generar → juzgar → DPO → repetir.
- [[concepts/llm-as-a-judge]] — Paradigma LLM-as-a-Judge: rúbrica aditiva, prompts, sesgos conocidos.
- [[concepts/irl]] — Inverse Reinforcement Learning: formulación clásica (Ng-Russell) y MaxEnt (Ziebart).
- [[concepts/preference-pair]] — Modelo Bradley-Terry y construcción de pares (y_w, y_l).
- [[concepts/rlhf]] — Pipeline canónico RLHF (Christiano → InstructGPT) — `WIP`.
- [[concepts/rlaif]] — Feedback de IA: Constitutional AI y RLAIF sistemático — `WIP`.
- [[concepts/kl-beta]] — Coeficiente KL β: significado, ajuste, comportamiento iterativo — `WIP`.
- [[concepts/reward-hacking]] — Modos de fallo: sesgo de auto-preferencia, length inflation, collapse — `WIP`.
- [[concepts/cooperative-marl]] — Common-payoff Markov games aplicados a LLMs — `WIP`.
- [[concepts/pbrl]] — Preference-Based RL pre-LLM (Wirth 2017) — `WIP`.

## Entities

Modelos, benchmarks, datasets como ciudadanos de primera clase.

- [[entities/alpaca-eval]] — AlpacaEval 2.0: setup, length-controlled win rate vs GPT-4 Turbo — `WIP`.
- [[entities/mt-bench]] — MT-Bench: 8 categorías, multi-turn — `WIP`.
- [[entities/arena-hard]] — Arena-Hard: prompts difíciles, comparación pairwise — `WIP`.
- [[entities/llama-2-70b]] — Modelo base de Yuan 2024 — `WIP`.

## Summaries

Un resumen por paper ingerido. **Tier-1** (resumen completo) en negrita; el resto reside en `sources.md`.

- **[[summaries/yuan-2024-self-rewarding]]** — Yuan et al. 2024, paper ancla.
- **[[summaries/rafailov-2023-dpo]]** — Rafailov et al. 2023, pilar técnico DPO.
- **[[summaries/wu-2024-meta-rewarding]]** — Wu et al. 2024, extensión meta-judge.
- **[[summaries/azar-2023-ipo]]** — Azar et al. 2023, marco teórico ΨPO/IPO.
- **[[summaries/wang-2025-temporal-sr]]** — Wang et al. 2025, crítica empírica central.
- **[[summaries/ziebart-2008-maxent-irl]]** — Ziebart et al. 2008, puente IRL ↔ DPO.
- **[[summaries/zheng-2023-llm-as-judge]]** — Zheng et al. 2023, fundamento LLM-as-a-Judge.
- **[[summaries/wirth-2017-pbrl-survey]]** — Wirth et al. 2017 (JMLR), survey PbRL.

Tier-2 (entrada en [[sources]] sin página propia hasta que se necesite): Ouyang 2022, Christiano 2017, Schulman 2017, Bai 2022, Lee 2023, Chen 2024 (SPIN), Wu 2024 (SPPO), Ng-Russell 2000, Abbeel-Ng 2004, Madaan 2023, Zelikman 2022, Huang 2023, Saunders 2022, Du 2024, Liang 2024, Kim 2024, Hejna-Sadigh 2023, Zhou 2025, Zhang 2025, Shafayat 2025, Li 2025, Ethayarajh 2024, Hong 2024, Goodfellow 2014.

## Synthesis

Análisis transversal — el "núcleo argumentativo" de la presentación.

- **[[synthesis/pilar-1-irl-implicito]]** — Pilar 1: ¿Es Self-Rewarding una instancia de IRL? Conexión formal vía MaxEnt-IRL ↔ DPO.
- **[[synthesis/pilar-2-multi-agente-cooperativo]]** — Pilar 2: framing como common-payoff cooperative game.
- **[[synthesis/pilar-3-dpo-en-loop]]** — Pilar 3: cómo encaja DPO en M_t → M_{t+1}, rol de β.
- **[[synthesis/pipeline-completo]]** — Pipeline end-to-end con pseudocódigo — `WIP`.
- **[[synthesis/diagnostico-fallos]]** — Modos de fallo documentados: saturación, sesgo de auto-preferencia, collapse, falla en razonamiento — síntesis de Wang 2025, Zhou 2025, Zhang 2025, Shafayat 2025, Self-Preference Bias 2024.
- **[[synthesis/comparacion-metodos]]** — RLHF vs RLAIF vs SPIN vs Self-Rewarding vs Meta-Rewarding vs SPPO — `WIP`.
- **[[synthesis/genealogia-intelectual]]** — Sadigh 2017 → Christiano 2017 → Wirth 2017 → Ouyang 2022 → STaR/Self-Improve/Self-Refine → Yuan 2024 → críticas 2025 — `WIP`.

---

**Convenciones del índice:**

- Una línea por página: `- [[ruta/slug]] — resumen de una línea`.
- Páginas tier-1 marcadas con `**negrita**` en el listado de summaries.
- Actualizar en cada ingest. Mantener ordenado por categoría, no por fecha.
