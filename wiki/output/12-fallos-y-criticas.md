# 12 — Fallos y críticas de Self-Rewarding

> **Prerequisitos**: capítulos 02, 04, 06, 08.

> **Lo que sabrás al final**: los seis modos de fallo principales documentados con evidencia empírica, sus mecanismos causales, sus mitigaciones, y exactamente qué decir si en la defensa un profesor pregunta "¿pero esto realmente funciona?".

> ⚠️ **Este capítulo es el más importante para defender la presentación**. Una presentación que solo celebra Self-Rewarding sin discutir sus fallos queda vulnerable. Aquí te preparas para esa conversación.

---

## 1. Por qué importa este capítulo

Yuan 2024 reporta resultados impresionantes (Llama 2 70B supera Claude 2, Gemini Pro, GPT-4 0613 en AlpacaEval 2.0 tras 3 iteraciones). Pero hay **tres líneas independientes de evidencia empírica** posteriores que muestran los límites del método. Conocerlas:

1. Permite defender honestamente.
2. Demuestra pensamiento crítico — no eres "fan" del paper, sabes evaluarlo.
3. Posiciona Self-Rewarding como **paradigma con límites identificados**, no como solución universal.

---

## 2. Los seis modos de fallo principales

### Modo 1 — Saturación del gradiente DPO

**Quién lo documenta**: Wang et al. 2025, *Temporal Self-Rewarding Language Models*, arXiv:2508.06026.

**El mecanismo (paso a paso)**:

Recordemos del cap 02 que el gradiente DPO es:
```
∇_θ L_DPO ∝ σ( β · (r̂(y_l) − r̂(y_w)) ) · ( ∇log π_θ(y_w|x) − ∇log π_θ(y_l|x) )
```

En cada iteración, **tanto `y_w` como `y_l` vienen del mismo modelo `M_t`**. A medida que `M_t` mejora:
- `y_w` es buena.
- `y_l` también mejora (porque viene del mismo modelo mejor).
- La **diferencia representacional** entre `y_w` y `y_l` se estrecha.
- `r̂(y_w) ≈ r̂(y_l)` → el factor `σ(...) → 0.5`.
- **El gradiente tiende a cero**.

> 💡 **Cooperative collapse**: el loop "se anula a sí mismo". No es que el modelo "ya esté entrenado óptimamente" — es que **se quedó sin señal**.

> 📊 **El mecanismo de saturación visualizado**:
>
> ```
>   Iteración t=0 (recién empezando)
>   ────────────────────────────────────────
>
>   Distribución de calidad de respuestas:
>
>      malas  ░░░░░░██████░░░░░░  buenas
>             pobres ↑   ↑ buenas
>                    │   │
>                    y_l │
>                        y_w
>
>   r̂(y_w) − r̂(y_l) = GRANDE (e.g., 3.0)
>   σ(GRANDE) ≈ 1
>   ∇L = ALTA — el modelo aprende rápido
>
>
>   Iteración t=2 (el modelo ya mejoró)
>   ────────────────────────────────────────
>
>   Distribución de calidad:
>
>      malas  ░░░░░░░░░░██████░░  buenas
>                          ↑↑
>                          ||
>                       y_l y_w  ← ambas son buenas!
>
>   r̂(y_w) − r̂(y_l) = PEQUEÑO (e.g., 0.3)
>   σ(PEQUEÑO) ≈ 0.6
>   ∇L = BAJA — el modelo aprende poco
>
>
>   Iteración t=4 (saturación)
>   ────────────────────────────────────────
>
>   Distribución de calidad:
>
>      malas  ░░░░░░░░░░░░░██████  buenas
>                              ↑↑
>                              ||
>                           y_l y_w  ← indistinguibles
>
>   r̂(y_w) − r̂(y_l) ≈ 0
>   σ(0) = 0.5
>   ∇L ≈ 0 — el modelo NO APRENDE más
>
>   ════════════════════════════════════════
>   COOPERATIVE COLLAPSE: el loop se anula
>   ════════════════════════════════════════
> ```

> 📊 **La solución de Temporal SR**: forzar que `y_w` y `y_l` vengan de **épocas distintas del modelo**:
>
> ```
>   Iteración t=4 SIN Temporal SR (saturación)
>   ──────────────────────────────────────────
>
>      y_w ← M_t   ─┐
>      y_l ← M_t   ─┴── ambas del MISMO M_t → indistinguibles
>
>
>   Iteración t=4 CON Temporal SR (margen preservado)
>   ──────────────────────────────────────────
>
>      y_w ← M_{t+EMA}  ← "futuro estimado", mejor que t
>      y_l ← M_0        ← SFT inicial (ancla fija)
>
>      r̂(y_w) − r̂(y_l) = SIGUE GRANDE
>      ∇L = SIGUE ALTA
>      Loop continúa funcionando
> ```

**Evidencia cuantitativa**: Wang 2025 muestra que la **distancia representacional** (medida con embeddings) entre `y_w` y `y_l` decrece monótonamente con `t`.

**Propuestas de mitigación**:

1. **Anchored Rejection**: fijar `y_l` desde el modelo inicial `M_0` (SFT), no `M_t`. Esto garantiza que `y_l` no mejore.
2. **Future-Guided Chosen**: generar `y_w` desde un modelo "futuro" (estimado vía EMA del entrenamiento). Mantiene un margen artificial.

**Implicación para la presentación**: Yuan 2024 evalúa solo 3 iteraciones porque más allá no hay mejora sostenida. Esto es **una limitación estructural**, no una elección arbitraria.

---

### Modo 2 — Sesgo acumulado en modelos pequeños

**Quién lo documenta**: Wang Z. et al. 2024, *CREAM*, arXiv:2410.12735, ICLR 2025.

**Mecanismo**: con modelos ~7B, las etiquetas de preferencia del juez interno se vuelven **progresivamente sobre-confiadas**. El juez se cree mucho más confiable de lo que realmente es. Las etiquetas se vuelven ruidosas pero confiadas → DPO refuerza decisiones malas.

**Diferencia con Modo 1**: la saturación de Wang 2025 es **geométrica** (`r̂(y_w) − r̂(y_l) → 0`). El sesgo acumulado de CREAM es **de calibración** — el juez se vuelve menos confiable, no menos discriminativo.

**Evidencia**: CREAM muestra que Self-Rewarding sin regularización **regresa** en modelos 7B tras varias iteraciones. Con CREAM, sigue mejorando.

**Mitigación (CREAM)**: regularizar por consistencia entre iteraciones consecutivas. Si `M_{t-1}` y `M_t` discrepan sobre un par, atenuar.

**Implicación**: Self-Rewarding **no funciona out-of-the-box en modelos pequeños**. Yuan 2024 usa 70B donde el problema es menos severo.

---

### Modo 3 — Inconsistencia entre rewards internos

**Quién lo documenta**: Zhou et al. 2025, *SCIR: Self-Consistency of the Internal Reward Models*, arXiv:2502.08922.

**Mecanismo**: el LLM tiene **dos reward models implícitos**:

1. **Juez generativo (LLM-as-Judge)**: `s(y|x)` vía prompt explícito.
2. **Reward DPO implícito**: `r̂_θ(x,y) = β · log(π_θ(y|x) / π_ref(y|x))`.

**Hallazgo central**: estos dos rewards **predicen preferencias inconsistentes** entre sí en una fracción significativa de pares. El juez generativo dice `y_w ≻ y_l`, pero el DPO implícito asigna `r̂(y_l) > r̂(y_w)`.

> 💡 **Esto es muy profundo**: significa que **el modelo es internamente inconsistente sobre qué es "bueno"**. Hay al menos dos voces internas que dan respuestas contradictorias.

**Implicación filosófica**: si el modelo no sabe internamente cuál es su objetivo real, ¿qué garantiza que Self-Rewarding optimice algo coherente?

**Mitigación (SCIR)**: filtrar pares donde las dos señales internas coinciden; descartar pares inconsistentes.

---

### Modo 4 — Falla en dominios verificables (razonamiento matemático)

**Quién lo documenta**: Zhang et al. 2025, *Process-SRLM*, arXiv:2503.03746, ACL 2025; Shafayat et al. 2025, *Can LRMs Self-Train?*, arXiv:2505.21444.

**Mecanismo**: Yuan 2024 evalúa en instruction-following abierto (AlpacaEval, MT-Bench) — dominios "subjetivos" donde el juez tiene margen para tener razón. En razonamiento matemático estricto:

- El juez interno tiene **acceso al mismo modelo** que produce errores.
- Si `M_t` no sabe resolver un problema, `M_t` como juez **tampoco** sabe verificarlo correctamente.
- Shafayat 2025 documenta que RL prolongado con self-reward sobre razonamiento conduce a **reward hacking que causa colapso súbito** de performance.

**Evidencia**: Self-Rewarding aplicado a benchmarks de math **degrada** la performance del modelo. No mejora — empeora.

**Mitigaciones**: Process-SRLM (juicio paso a paso, cap 09), ScPO (consistency entre samples, cap 09), GRPO con reward verificable (cap 13).

**Implicación**: el paradigma **no es universal**. Funciona en dominios donde el juez tiene margen subjetivo. Falla en dominios con verdad objetiva verificable cuando el modelo no la conoce.

---

### Modo 5 — Self-preference bias (auto-preferencia)

**Quién lo documenta**: Zheng et al. 2023 (cualitativo); *Self-Preference Bias in LLM-as-a-Judge*, arXiv:2410.21819 (cuantitativo).

**Mecanismo**: los LLMs como jueces prefieren outputs **estilísticamente familiares** — los que su propio modelo produciría. La causa documentada: **perplexity más baja** sobre textos familiares → score más alto.

En Self-Rewarding el juez **es el mismo modelo que entrena**. El sesgo se **amplifica máximamente**:

```
M_t genera y con su distribución preferida.
   ↓
M_t como juez asigna alto s(y|x) PRECISAMENTE porque y es familiar.
   ↓
DPO refuerza la distribución preferida.
   ↓
M_{t+1} es aún más "él mismo".
```

**Implicación**: parte del aumento reportado en AlpacaEval 2 **podría ser auto-preferencia inflada**, no mejora genuina de calidad.

**Mitigación parcial**: Yuan 2024 evalúa con GPT-4 Turbo (distinto a Llama 2 70B) como juez externo en la métrica final. Esto reduce pero no elimina el problema — el sesgo en la **construcción interna de pares** persiste.

---

### Modo 6 — Degeneración del pensamiento (DoT)

**Quién lo documenta**: Liang et al. 2024, *MAD: Multi-Agent Debate*, EMNLP 2024, arXiv:2305.19118.

**Mecanismo**: un LLM con auto-reflexión tiende a **atrincherarse** en su solución inicial aun cuando es incorrecta. La auto-crítica frecuentemente **confirma la posición original** en lugar de revisarla.

En Self-Rewarding, el juez `M_t` evalúa generaciones del mismo `M_t`. Hay **incentivo estructural** a producir juicios consistentes con la generación, no a contradecirla.

**Mitigación documentada (MAD)**: introducir **múltiples agentes con posiciones contrarias** ("tit-for-tat") moderados por un juez separado.

**Implicación**: el setup single-model de Self-Rewarding es **estructuralmente vulnerable** a DoT. Meta-Rewarding (cap 07) y Multi-Agent Evolve (cap 11) abordan este límite separando los roles.

---

## 3. El mecanismo causal unificado

Los seis fallos comparten **una raíz común**: el drift acumulativo del juez.

```
En cada iteración t:
  - M_t define la noción de "buena respuesta" (vía el juez interno).
  - M_t genera y_w bajo SU noción.
  - M_t entrena para producir más de y_w.
  - Resultado: M_{t+1} converge a SU propia distribución preferida,
    no necesariamente a la distribución óptima.

Acumulado en t iteraciones:
  - π_ref se mueve con t (no es ancla fija).
  - KL(M_t || M_0) crece sin cota.
  - El "experto interno" (juez) se aleja del juez ideal externo.
```

> 💡 **Frase clave**: "**Cooperative collapse** — versión cooperativa del mode collapse adversarial. Los dos sub-agentes (generador y juez) convergen juntos a explotar el mismo sesgo, no a mejorar la política conjunta."

---

## 4. Matriz resumen de las críticas

| # | Crítica | Paper | Métrica/evidencia | Mitigación |
|---|---|---|---|---|
| 1 | Saturación de señal | Wang 2025 | `r̂(y_w) − r̂(y_l) → 0` | Temporal SR (anchored rejection + future-guided chosen) |
| 2 | Sesgo acumulado (modelos ~7B) | CREAM 2024 | Retroceso de rendimiento tras varias iter | Regularización por consistencia inter-iter |
| 3 | Rewards internos inconsistentes | SCIR 2025 | Disagreement rate entre judge y DPO implícito | Filtrar pares consistentes |
| 4 | Falla en matemáticas | Zhang 2025, Shafayat 2025 | Degradación en GSM8K-like, colapso súbito | Process-SRLM, ScPO, GRPO |
| 5 | Self-preference bias | arXiv:2410.21819, Zheng 2023 | Win rate diferencial mismo vs externo | Juez externo / panel diverso |
| 6 | Degeneración del pensamiento | Liang 2024 (MAD) | DoT en self-reflection ingenuo | Multi-agent debate, Meta-Rewarding |

---

## 5. Cómo presentar las críticas en la slide

Tres opciones:

### Opción A — Una sola slide "Limitaciones"

Tabla resumen con las 6 críticas, una línea cada una. **Honesto y económico**, ocupa una slide.

### Opción B — Slide dedicada "Diagnóstico de fallos"

2-3 slides:
1. *El loop como objeto de estudio*: ¿qué garantías de convergencia tiene? Ninguna fuerte.
2. *Tres líneas de crítica empírica*: Wang 2025, Zhou 2025, Zhang 2025.
3. *Un mecanismo causal*: self-preference bias + drift acumulativo.

### Opción C — Crítica integrada en cada slide del método

Cada slide del método tiene una sección "limitación" con la crítica más relevante. **Distribuye el peso** pero puede diluir el mensaje crítico.

**Recomendado**: opción B si hay tiempo (5-10 min). La defensa de maestría se beneficia de **mostrar pensamiento crítico**, no solo entusiasmo.

---

## 6. Frase para defensa

> *"Self-Rewarding funciona en 3 iteraciones sobre tareas de instruction-following abiertas. Hay seis modos de fallo documentados — saturación del gradiente DPO (Wang 2025), sesgo acumulado en modelos pequeños (CREAM 2024), inconsistencia entre rewards internos (SCIR 2025), falla en razonamiento matemático (Zhang 2025, Shafayat 2025), self-preference bias amplificado, y degeneración del pensamiento. Todos comparten una raíz: el drift acumulativo del juez sin ancla externa. Mitigaciones: Meta-Rewarding entrena el juez explícitamente, Temporal SR desacopla `y_w` y `y_l` temporalmente, Process-SRLM evalúa paso a paso, métodos de consistency filtran señales ruidosas. La línea no es 'Self-Rewarding funciona perfecto' sino 'Self-Rewarding abre un paradigma con límites conocidos y soluciones parciales emergiendo'."*

---

## 7. Preguntas frecuentes que un profesor podría hacer

### Q: "¿Por qué solo 3 iteraciones en Yuan 2024?"

A: "Porque más allá no hay mejora sostenida. Wang 2025 lo formaliza: la diferencia representacional entre `y_w` y `y_l` se estrecha con `t`, el gradiente DPO tiende a cero. Es **saturación matemática**, no elección arbitraria."

### Q: "¿La mejora reportada no es self-preference inflada?"

A: "Parcialmente. Yuan 2024 mitiga evaluando con GPT-4 Turbo externo. Pero el sesgo en la construcción interna de pares persiste. Una métrica más honesta usaría panel diverso de jueces. Aún así, los números son lo suficientemente grandes que no se atribuyen solo a sesgo."

### Q: "¿Funciona en mi dominio?"

A: "Depende. En instruction-following abierto, sí. En math estricto, no — usa Process-SRLM o GRPO. En tareas con respuesta única correcta, ScPO. En visión-lenguaje, CSR o Vision-SR1. **El paradigma no es universal**."

### Q: "¿Y safety?"

A: "Pregunta abierta importante. Sin ancla externa (constitución, humano, verificador), el sistema puede derivar. Constitutional AI lo aborda con principios explícitos. Pero modelos que se auto-mejoran sin supervisión continua son un problema de research activo."

### Q: "¿Cómo se compara con DeepSeek-R1?"

A: "Son complementarios. R1 usa reward verificable + GRPO en razonamiento. Self-Rewarding usa LLM-as-Judge + DPO en instruction-following. Cubren regímenes distintos. Ambos validan la dirección 'auto-mejora sin humanos post-SFT'."

---

## 8. Lo que NO está resuelto

Preguntas abiertas honestas para "future work":

1. **Convergencia formal**: no hay teorema que acote la saturación. Hay preprints proponiéndolo pero falta consolidación.
2. **Calibración del juez en el tiempo**: ningún paper reporta sistemáticamente cómo evoluciona la calibración (no solo ranking) del juez interno.
3. **Generalización fuera de subjetivo**: ¿funciona en derecho? ¿En medicina? Process-SRLM sugiere que no, sin modificaciones específicas.
4. **Comparación replicable**: solo SPPO ha hecho comparación uniforme. Falta replicación independiente más amplia.

---

## 9. Qué viene

[13-grpo-deepseek.md](13-grpo-deepseek.md) — el paradigma alternativo: rewards verificables + GRPO + R1. La "otra rama" de la evolución.

---

## Lecturas relacionadas

- [[synthesis/diagnostico-fallos]] — versión densa.
- [[summaries/wang-2025-temporal-sr]] — modo 1 en detalle.
- Entradas CREAM, SCIR, Process-SRLM, Shafayat 2025 en [[sources]].
