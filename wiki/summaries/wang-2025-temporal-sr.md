---
name: wang-2025-temporal-sr
description: Resumen de Temporal Self-Rewarding (Wang et al. 2025) — diagnóstico cuantitativo del modo de fallo central de Self-Rewarding, arXiv:2508.06026.
metadata:
  type: summary
  tags: [critica, self-rewarding, saturacion, modo-de-fallo, temporal]
  last_updated: 2026-05-12
---

# Wang et al. (2025) — Temporal Self-Rewarding Language Models

## Cita

Yidong Wang, Xin Wang, Cunxiang Wang, Junfeng Fang, Qiufeng Wang, Jianing Chu, Xuran Meng, Shuxun Yang, Libo Qin, Yue Zhang, Wei Ye, Shikun Zhang. *Temporal Self-Rewarding Language Models: Decoupling Chosen-Rejected via Past-Future*. arXiv:2508.06026, agosto 2025.

- **arXiv**: https://arxiv.org/abs/2508.06026

## Resumen técnico

Análisis crítico de [[summaries/yuan-2024-self-rewarding]]. Identifica un modo de fallo cuantitativo:

> **La mejora sincronizada de respuestas elegidas (y_w) y rechazadas (y_l) estrecha progresivamente la diferencia representacional entre muestras contrastivas, degradando la señal de aprendizaje.**

**Mecanismo formal**: en cada iteración `t`, ambas respuestas vienen del mismo modelo `M_t`. A medida que `M_t` mejora:

- Tanto `y_w` como `y_l` mejoran simultáneamente.
- La distancia representacional `d(y_w, y_l)` decae.
- En la pérdida DPO, el gradiente es:
  ```
  ∇ L_DPO ∝ σ( β·[r̂(y_w) − r̂(y_l)] ) · ∇[r̂(y_w) − r̂(y_l)]
  ```
- Si `r̂(y_w) − r̂(y_l) → 0`, el gradiente se anula. El loop deja de aprender.

**Propuesta del paper** — framework dual para evitar este colapso:

1. **Anchored Rejection**: fijar `y_l` (respuesta rechazada) usando **el modelo inicial pasado** `M_0`, no el actual `M_t`. La respuesta rechazada permanece "vieja" — preserva el gap representacional.
2. **Future-Guided Chosen**: curar `y_w` (respuesta elegida) usando predicciones de **la próxima generación del modelo** `M_{t+1}` (estimada vía algún proxy). La respuesta elegida se anticipa al futuro — agranda el gap.

Resultado: el gap `r̂(y_w) − r̂(y_l)` se mantiene grande iteración tras iteración, prolongando la señal de aprendizaje.

## Aporte clave para la presentación

Es **la crítica empírica más sustantiva** publicada después de Yuan 2024. Aporta:

1. **Diagnóstico cuantitativo** del modo de fallo principal de Self-Rewarding — la saturación post-3-iteraciones que Yuan reporta sin explicar.
2. **Mecanismo formal** (no solo observación): conecta la saturación a la dinámica del gradiente DPO.
3. **Solución probada**: muestra que decoupling temporal (Anchored Rejection + Future-Guided Chosen) permite extender el loop más allá de 3 iteraciones.

**Imprescindible** en la slide de "Limitaciones / Diagnóstico de fallos" ([[synthesis/diagnostico-fallos]]).

## Fórmulas centrales

Formulación del gap representacional iterativo:

```
Δ_t = E_{(x,y_w,y_l) ~ AIFT(M_t)} [ r̂_{M_t}(x, y_w) − r̂_{M_t}(x, y_l) ]
```

Hipótesis del paper: `Δ_t → 0` cuando `t` crece, en Self-Rewarding original.

**Anchored Rejection**: redefinir `y_l ~ M_0(·|x)` (sampling de la versión inicial). Esto mantiene `r̂(y_l)` aproximadamente constante.

**Future-Guided Chosen**: redefinir `y_w` usando un estimador de `M_{t+1}` antes de entrenar. Operacionalmente, esto puede implementarse vía multiple-step lookahead o vía bootstrap del propio modelo con mayor sampling.

## Resultados empíricos

[VERIFICAR números específicos en el paper — el agente de investigación reportó solo el mecanismo y la propuesta, no las cifras finales. Recomendable consultar arXiv:2508.06026 directamente para AlpacaEval, MT-Bench, y número de iteraciones sostenidas.]

Hallazgo central reportado: el gap representacional se mantiene a lo largo de **más iteraciones** que en Self-Rewarding original, y la performance sigue mejorando donde Yuan 2024 satura.

## Limitaciones de este paper (meta-crítica)

1. La propuesta **agrega complejidad operacional**: requiere mantener `M_0` y estimar `M_{t+1}`.
2. Anchored Rejection depende de la calidad de `M_0`; si el SFT inicial es muy pobre, los `y_l` quedan demasiado fáciles.
3. Future-Guided Chosen es una heurística — no hay garantía teórica de que el "lookahead" mejore consistentemente.

## Cómo usarlo en la presentación

Sugerencia: en la slide de **diagnóstico de fallos**, incluir:

- Una gráfica conceptual: `Δ_t` (gap representacional) decayendo en Self-Rewarding original vs manteniéndose plano en Temporal SR.
- Una línea: *"Wang et al. 2025 (arXiv:2508.06026) muestran que el gradiente DPO colapsa porque `y_w` y `y_l` mejoran juntos. La solución: fijar `y_l` con el modelo inicial."*

## Conexiones en el wiki

- [[summaries/yuan-2024-self-rewarding]] — paper criticado.
- [[summaries/rafailov-2023-dpo]] — la dinámica del gradiente DPO que se analiza.
- [[concepts/preference-pair]] — la construcción de pares que el paper modifica.
- [[synthesis/pilar-3-dpo-en-loop]] — el rol de β y el drift acumulativo (contexto).
- [[synthesis/diagnostico-fallos]] — esta crítica como modo de fallo #1.
