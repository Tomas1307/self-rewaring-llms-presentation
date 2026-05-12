---
name: diagnostico-fallos
description: Síntesis de los modos de fallo documentados de Self-Rewarding — saturación, sesgo de auto-preferencia, collapse, falla en razonamiento.
metadata:
  type: synthesis
  tags: [criticas, fallos, reward-hacking, saturacion, sesgos, defensa]
  last_updated: 2026-05-12
---

# Diagnóstico de fallos de Self-Rewarding

> **Tesis.** La saturación tras 3 iteraciones reportada por Yuan 2024 está respaldada por **tres líneas independientes de evidencia empírica** + un **mecanismo causal documentado** (self-preference bias). Presentarlas como cuerpo coherente de crítica — no anécdotas aisladas — es esencial para una defensa académica honesta.

## 1. Por qué importa esta página

Una presentación que solo celebre Self-Rewarding sin discutir sus fallos quedará vulnerable a preguntas. Esta página consolida las críticas para que la defensa pueda anticipar y responder.

## 2. Los cuatro modos de fallo principales

### 2.1. Saturación / colapso de la señal de preferencia

**Quién lo documenta**: [[summaries/wang-2025-temporal-sr]] (Wang et al. 2025, *Temporal Self-Rewarding Language Models*, arXiv:2508.06026).

**Mecanismo**: en cada iteración, tanto `y_w` como `y_l` mejoran (porque vienen del mismo modelo mejorado). La **diferencia representacional** entre ellos se estrecha. Formalmente, en la pérdida DPO:

```
gradient ∝ σ( β·[r̂(y_w) − r̂(y_l)] ) · [r̂(y_w) − r̂(y_l)]
```

Si `r̂(y_w) ≈ r̂(y_l)`, el gradiente tiende a cero. **El loop se anula a sí mismo.**

**Propuesta de mitigación**: 
- *Anchored Rejection*: fijar `y_l` con el modelo inicial `M_0`, no `M_t`.
- *Future-Guided Chosen*: generar `y_w` con una predicción de `M_{t+1}` (modelo futuro).

**Implicación para la presentación**: Yuan 2024 evalúa solo 3 iteraciones. Más allá no se mantiene la mejora.

### 2.2. Inconsistencia entre rewards internos

**Quién lo documenta**: Zhou et al. 2025 (*Self-Consistency of the Internal Reward Models*, arXiv:2502.08922).

**Mecanismo**: Self-Rewarding tiene **dos reward models implícitos**:

1. El **juez generativo** (LLM-as-a-Judge): asigna `s(y|x)` vía prompt explícito.
2. El **reward DPO implícito**: `r̂_θ(x,y) = β · log(π_θ(y|x) / π_ref(y|x))`.

**Hallazgo central**: estos dos rewards **predicen preferencias inconsistentes** entre sí en una fracción significativa de pares. El juez generativo dice `y_w ≻ y_l`, pero el DPO implícito asigna `r̂(y_l) > r̂(y_w)`.

**Propuesta de mitigación** (SCIR): penalizar inconsistencia, usar solo pares con predicción consistente entre ambos rewards.

**Implicación**: el "juez interno" no es coherente consigo mismo. Esto socava la legitimidad del paradigma — si los rewards internos discrepan, ¿cuál es el verdadero objetivo de optimización?

### 2.3. Falla en dominios verificables (razonamiento matemático)

**Quién lo documenta**: Zhang et al. 2025 (*Process-based Self-Rewarding Language Models*, Findings of ACL 2025, arXiv:2503.03746); Shafayat et al. 2025 (*Can Large Reasoning Models Self-Train?*, arXiv:2505.21444).

**Mecanismo**: Yuan 2024 evalúa en instruction-following abierto (AlpacaEval, MT-Bench) — dominios "subjetivos" donde el juez tiene margen para tener razón. En razonamiento matemático estricto:

- El juez interno tiene **acceso al mismo modelo** que produce errores. Si `M_t` no sabe resolver un problema, `M_t` como juez tampoco sabe verificarlo correctamente.
- Shafayat 2025 muestra que el RL prolongado con self-reward sobre razonamiento conduce a **reward hacking que causa colapso súbito** de performance.

**Propuesta de mitigación** (Process-SRLM): juicio paso-a-paso (step-wise LLM-as-a-Judge), no respuesta-final. Integrar long-thought reasoning.

**Implicación**: el paradigma no es universal. Funciona en dominios donde el juez tiene margen subjetivo. Falla cuando hay verdad objetiva verificable y el modelo no la conoce.

### 2.4. Sesgo de auto-preferencia (self-enhancement bias)

**Quién lo documenta**: [[summaries/zheng-2023-llm-as-judge]] (Zheng et al. 2023) identifica el sesgo cualitativamente. *Self-Preference Bias in LLM-as-a-Judge* (arXiv:2410.21819) lo mide cuantitativamente.

**Mecanismo**: los LLM judges prefieren outputs de su **propia familia** (mismo modelo, modelos hermanos). La causa documentada: **perplexity más baja** sobre textos estilísticamente familiares → score más alto.

En Self-Rewarding **el juez es el mismo modelo que entrena**. El sesgo se amplifica máximamente:

- `M_t` genera `y` con su distribución preferida.
- `M_t` como juez asigna alto `s(y|x)` *precisamente porque* `y` es similar a su propia distribución.
- El loop refuerza la distribución preferida, no necesariamente la calidad real.

**Implicación**: parte del aumento reportado en AlpacaEval podría ser auto-preferencia inflada, no mejora genuina. Una métrica honesta debería juzgarse con un evaluador **independiente** (GPT-4 externo, humanos).

Yuan 2024 mitiga parcialmente al evaluar con GPT-4 Turbo (modelo distinto a Llama-2-70B), pero el sesgo en la construcción interna de pares persiste.

### 2.5. Degeneración del pensamiento (Degeneration of Thought, DoT)

**Quién lo documenta**: Liang et al. 2024 (*MAD: Multi-Agent Debate*, EMNLP 2024, arXiv:2305.19118).

**Mecanismo**: un LLM con auto-reflexión tiende a **atrincherarse** en su solución inicial aun cuando es incorrecta. La auto-crítica suele confirmar la posición original en lugar de revisarla.

En Self-Rewarding, el juez `M_t` evalúa generaciones del mismo `M_t`. Hay incentivo estructural a producir juicios consistentes con la generación, no a contradecirla.

**Mitigación documentada** (MAD): introducir múltiples agentes con **posiciones contrarias** ("tit-for-tat") moderados por un juez separado.

**Implicación**: el setup single-model de Self-Rewarding es estructuralmente vulnerable a DoT. Meta-Rewarding (introduciendo un meta-judge separado) y CORY/MAE (introduciendo múltiples agentes) abordan este límite.

## 3. Mecanismo causal unificado: el drift acumulativo

Los cuatro fallos comparten una raíz común:

```
En cada iteración t:
  - M_t define la noción de "buena respuesta" (vía el juez).
  - M_t genera y_w bajo SU noción.
  - M_t entrena para producir más de y_w.
  - Resultado: M_{t+1} converge a SU propia distribución preferida,
    no necesariamente a la distribución óptima.

Acumulado en t iteraciones:
  - π_ref se mueve con t (no es ancla fija).
  - KL(M_t ‖ M_0) crece sin cota.
  - El "experto interno" (juez) se aleja del juez ideal externo.
```

Este es el "**cooperative collapse**" — versión cooperativa del mode collapse adversarial. Los dos sub-agentes (generador y juez) convergen juntos a explotar el mismo sesgo, no a mejorar la política conjunta.

## 4. Cuerpo de crítica como matriz

| # | Crítica | Paper | Métrica/evidencia | Mitigación propuesta |
|---|---|---|---|---|
| 1 | Saturación de señal | Wang 2025 (arXiv:2508.06026) | `r̂(y_w) − r̂(y_l) → 0` con t | Anchored Rejection + Future-Guided Chosen |
| 2 | Rewards internos inconsistentes | Zhou 2025 (arXiv:2502.08922) | Disagreement rate entre judge y DPO implícito | SCIR (filtrar pares consistentes) |
| 3 | Falla en matemáticas | Zhang 2025 (arXiv:2503.03746) | Degradación en GSM8K-like | Process-SRLM (juicio step-wise) |
| 4 | Colapso por reward hacking | Shafayat 2025 (arXiv:2505.21444) | Performance crash en RL prolongado | (Sin solución cerrada) |
| 5 | Self-preference bias | arXiv:2410.21819, Zheng 2023 | Win rate diferencial mismo-modelo vs externo | Juez externo / panel diverso |
| 6 | Degeneración del pensamiento | Liang 2024 (arXiv:2305.19118) | DoT en self-reflection ingenuo | Multi-agent debate con juez separado |

## 5. Cómo presentar las críticas

**Opción A — Slide única "Limitaciones"**: una slide con tabla resumen de las 6 críticas. Honesto y económico.

**Opción B — Slide "Diagnóstico de fallos"**: dedicarle 2-3 slides:
1. *El loop como objeto de estudio*: ¿qué garantías de convergencia tiene?
2. *Tres líneas de crítica empírica*: Wang 2025, Zhou 2025, Zhang 2025.
3. *Un mecanismo causal*: self-preference bias + drift acumulativo.

**Recomendado**: opción B si hay tiempo. La defensa de maestría se beneficia de mostrar pensamiento crítico, no solo entusiasmo.

## 6. Frase para defensa

> *"Self-Rewarding funciona en 3 iteraciones sobre tareas de instruction-following abiertas. Hay tres líneas independientes de evidencia empírica — Wang 2025, Zhou 2025, Zhang 2025 — que muestran su saturación o degradación más allá, y un mecanismo causal documentado: el sesgo de auto-preferencia del juez interno se amplifica cuando juez y generador son el mismo modelo. Meta-Rewarding (Wu 2024) y enfoques multi-agente (CORY, MAE) abordan estos límites separando los roles."*

## 7. Lo que NO está resuelto

Preguntas abiertas que la presentación puede dejar como "future work":

1. **Convergencia formal.** No hay teorema que garantice o acote la saturación. (Hay un preprint reciente que lo propone — arXiv:2601.22513 — pero requiere verificación.)
2. **Calibración del juez.** Ningún paper reporta sistemáticamente cómo evoluciona la calibración (no solo el ranking) del juez interno con `t`.
3. **Generalización fuera de subjetivo.** ¿Funciona en código? ¿En razonamiento legal? Process-SRLM sugiere que no, sin modificaciones.
4. **Comparación uniforme.** Solo SPPO ([[summaries/wu-2024-sppo]]) ha hecho una comparación head-to-head con setup uniforme (Mistral-7B-Instruct-v0.2, UltraFeedback 60k). Falta replicación independiente.

## 8. Lecturas relacionadas

- [[summaries/wang-2025-temporal-sr]] — diagnóstico cuantitativo del modo de fallo principal.
- [[summaries/zheng-2023-llm-as-judge]] — sesgos del paradigma LLM-as-a-Judge.
- [[summaries/wu-2024-meta-rewarding]] — solución parcial vía meta-judge separado.
- [[synthesis/pilar-3-dpo-en-loop]] — el rol de β y el drift acumulativo.
- [[concepts/reward-hacking]] — definición del modo de fallo general (WIP).
