---
name: wirth-2017-pbrl-survey
description: Resumen del JMLR survey de Preference-Based RL (Wirth et al. 2017) — puente formal IRL ↔ RLHF pre-LLM, JMLR vol. 18.
metadata:
  type: summary
  tags: [pbrl, irl, survey, fundamentos, jmlr, peer-reviewed]
  last_updated: 2026-05-12
---

# Wirth et al. (2017) — A Survey of Preference-Based Reinforcement Learning Methods

## Cita

Christian Wirth, Riad Akrour, Gerhard Neumann, Johannes Fürnkranz. *A Survey of Preference-Based Reinforcement Learning Methods*. **Journal of Machine Learning Research (JMLR)** vol. 18, pp. 1–46, 2017.

- **URL (JMLR)**: https://jmlr.org/papers/v18/16-634.html
- **Peer-reviewed journal** (no arXiv) — valioso para diversidad editorial de la bibliografía.

## Resumen técnico

Survey canónica de **Preference-Based Reinforcement Learning (PbRL)** anterior a la era LLM. Formaliza el problema de aprender de preferencias en lugar de recompensas numéricas escalares, y taxonomiza los métodos existentes.

**Motivación clásica** del PbRL:
1. **Reward shaping es difícil** — diseñar `r` que induzca el comportamiento deseado requiere experticia y prueba-error.
2. **Humanos asignan preferencias con menor cognitive load** que recompensas numéricas. "Esto es mejor que aquello" es más fiable que "esto vale 7.3".
3. **Reduce dependencia de conocimiento experto del dominio**.

**Taxonomía** de PbRL (versión simplificada del survey):

| Categoría | Idea | Ejemplos |
|---|---|---|
| **Basados en utilidad** | Aprender una función de utilidad `u(s)` ó `u(τ)` y derivar política. | Chu & Ghahramani 2005 (Gaussian processes); Sadigh et al. 2017 (active preferences). |
| **Basados en preferencias directas** | Aprender política directamente, sin función de utilidad intermedia. | Akrour et al. 2012 (preference-based policy iteration). |
| **Bayesianos** | Inferencia probabilística sobre el espacio de recompensas/políticas. | Ramachandran & Amir 2007 (Bayesian IRL). |

**Modelo formal**: el PbRL trabaja con un conjunto de preferencias pairwise `D = {(τ_i ≻ τ_j)}`. El objetivo es encontrar una política `π` tal que las trayectorias inducidas por `π` sean preferidas.

**Modelo de Bradley-Terry** (usado por la mayoría de métodos):
```
P(τ_i ≻ τ_j) = exp(U(τ_i)) / [ exp(U(τ_i)) + exp(U(τ_j)) ]
```

donde `U` es una función de utilidad parametrizada.

## Aporte clave para la presentación

Tres aportes complementarios:

1. **Puente formal IRL ↔ RLHF pre-LLM**. Antes de [[summaries/christiano-2017-deep-rl-preferences]] (que aplica PbRL a deep RL), Wirth 2017 ya tenía toda la maquinaria conceptual. Este survey hace la conexión genealógica: PbRL es la teoría madre de la cual RLHF es un caso particular en LLMs.

2. **Diversidad editorial**: peer-reviewed en JMLR (no arXiv). Útil para mostrar que el portafolio bibliográfico no es solo preprints. Para una defensa de maestría rigurosa, citar JMLR refuerza la solvencia académica.

3. **Vocabulario formal**: "preference dataset", "utility function", "preference-based policy iteration" — el survey provee el lenguaje preciso que después aparece en DPO y derivados.

## Sin fórmulas nuevas (sintetiza fórmulas estándar de PbRL)

La fórmula central es Bradley-Terry (ya citada arriba), que reaparece en:
- [[summaries/christiano-2017-deep-rl-preferences]] (deep RL desde preferencias).
- [[summaries/rafailov-2023-dpo]] (DPO).
- [[summaries/yuan-2024-self-rewarding]] (construcción de pares).

## Resultados empíricos

Survey — no aporta resultados nuevos. Cubre el estado del arte 2010-2017 en PbRL clásico (robótica, control).

## Limitaciones / observaciones

1. **No cubre LLMs** (es de 2017). El boom RLHF post-2022 queda fuera de su scope.
2. **Asume MDP tabular o con features lineales** en la mayoría de métodos discutidos.
3. **No discute self-feedback** — todos los métodos asumen feedback externo (humano o simulado).

A pesar de estas limitaciones, **define el marco** que la era LLM hereda directamente. Es la pieza pre-2017 más importante para nuestra genealogía intelectual.

## Cómo usarlo en la presentación

Sugerencia: **una sola cita explícita** en la slide de "genealogía intelectual" o "fundamentos":

> *"PbRL como teoría general predata a RLHF en una década (Wirth et al. 2017, JMLR). RLHF en LLMs es un caso particular del marco de Wirth: aprender política desde preferencias pairwise vía Bradley-Terry."*

Esto demuestra al profesor que el alumno entiende la **genealogía** del campo, no solo los papers recientes.

## Conexiones en el wiki

- [[concepts/pbrl]] — Preference-Based RL en general (WIP).
- [[concepts/preference-pair]] — Bradley-Terry y construcción de pares.
- [[concepts/irl]] — IRL clásico que PbRL extiende.
- [[summaries/christiano-2017-deep-rl-preferences]] — aplica PbRL a deep RL.
- [[summaries/rafailov-2023-dpo]] — DPO como instancia de PbRL en LLMs.
- [[summaries/sadigh-2017-active-preferences]] — método específico de PbRL discutido en el survey (WIP).
- [[synthesis/pilar-1-irl-implicito]] — el puente PbRL ↔ IRL formaliza la analogía.
