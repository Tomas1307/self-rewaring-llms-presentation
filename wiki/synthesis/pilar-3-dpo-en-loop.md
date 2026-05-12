---
name: pilar-3-dpo-en-loop
description: Pilar 3 — mecánica precisa de cómo DPO encaja en la iteración M_t → M_{t+1} de Self-Rewarding; rol de β; por qué DPO y no PPO.
metadata:
  type: synthesis
  tags: [pilar, dpo, loop, beta, optimizacion]
  last_updated: 2026-05-12
---

# Pilar 3 — DPO como motor del loop

> **Tesis del pilar.** DPO ([[summaries/rafailov-2023-dpo]]) es lo que hace viable Self-Rewarding. Sin DPO, el loop requeriría entrenar un reward model separado e infraestructura RL pesada (PPO). DPO permite optimizar la política directamente sobre pares auto-generados, cerrando el ciclo de auto-mejora en una sola pérdida cerrada.

## 1. El loop con DPO en el centro

Cada iteración `M_t → M_{t+1}` ejecuta cuatro pasos:

```
1. Generación
   - Tomar prompt x del pool X_t.
   - Muestrear N=4 respuestas y_1, ..., y_N ~ M_t(· | x).
   - Hyperparams (Yuan 2024): T=0.7, top-p=0.9.

2. Auto-evaluación
   - El MISMO M_t, con prompt de LLM-as-a-Judge, asigna s(y_i | x) ∈ {0,...,5}.
   - Promedia tres evaluaciones por respuesta para reducir varianza.
   - Rúbrica: 5 criterios binarios (relevancia, cobertura, utilidad, claridad, experticia).

3. Construcción de pares
   - y_w = argmax_i s(y_i | x)
   - y_l = argmin_i s(y_i | x)
   - Descartar el par si s(y_w) = s(y_l) (sin señal de aprendizaje).
   - Agregar (x, y_w, y_l) al dataset AIFT(M_t).

4. DPO update
   - Inicializar M_{t+1} = M_t.
   - π_ref = M_t (congelado para esta iteración).
   - Optimizar L_DPO sobre AIFT(M_t).
```

Tamaños reportados por Yuan 2024:
- AIFT(M_1) = 3.964 pares → produce M_2.
- AIFT(M_2) = 6.942 pares → produce M_3.

## 2. La pérdida DPO en el contexto del loop

Reescrita con `π_ref = M_t`, `π_θ = M_{t+1}`:

```
L_DPO(θ; M_t) = − E_{(x, y_w, y_l) ~ AIFT(M_t)} [
    log σ(
        β · log( π_θ(y_w | x) / M_t(y_w | x) )
      − β · log( π_θ(y_l | x) / M_t(y_l | x) )
    )
]
```

**Símbolos**:
- `π_θ` = `M_{t+1}` en entrenamiento.
- `M_t` = referencia congelada (la versión anterior del modelo).
- `β` = coeficiente KL (Yuan 2024: **β = 0.1**, verificado en apéndice).
- `σ` = sigmoide logístico.
- `(x, y_w, y_l)` = par de preferencia auto-generado.

**Equivalencia formal con RLHF KL-restringido** (ver [[synthesis/pilar-1-irl-implicito]] §2):

```
max_π  E_{x, y~π} [ r(x,y) ]  −  β · KL( π ‖ π_ref )
```

DPO recupera la `r` implícita `r̂_θ(x,y) = β · log(π_θ(y|x) / π_ref(y|x))` sin entrenar un reward model separado.

## 3. El rol crítico de β

`β` controla cuánto puede divergir `M_{t+1}` de `M_t`. Es el multiplicador KL del problema RLHF original.

| Régimen | Efecto en una iteración | Efecto acumulativo |
|---|---|---|
| β alto (`>` 0.5) | Conservador: `M_{t+1}` casi igual a `M_t`. | Mejora lenta pero estable; bajo riesgo de drift. |
| β medio (≈ 0.1) | Estándar en literatura DPO. Yuan 2024 usa este valor. | Mejora visible por iteración; control aceptable. |
| β bajo (`<` 0.05) | Agresivo: `M_{t+1}` puede alejarse mucho. | Aceleración pero riesgo de *distribution drift* y *reward hacking*. |

**Subtlety crítica del loop**: `π_ref` **se actualiza cada iteración** a `M_t`. El "ancla" se mueve. Tras 3 iteraciones, `M_3` puede haber recorrido KL acumulado considerable respecto al SFT inicial `M_1`. Esto explica tanto las mejoras como las degradaciones observadas:

- Yuan 2024 reporta mejora monotónica en 3 iteraciones (9.94% → 15.38% → 20.44% en AlpacaEval 2.0).
- [[summaries/wang-2025-temporal-sr]] muestra que más allá de 3-4 iteraciones la diferencia `r̂(y_w) − r̂(y_l)` colapsa y el gradiente DPO se anula.

Ver [[synthesis/diagnostico-fallos]] §3 para el análisis completo de drift.

## 4. La "doble mejora" emergente

Yuan 2024 reporta dos métricas que mejoran simultáneamente:

1. `M_{t+1}` es **mejor actor** que `M_t` — medido por win rate en AlpacaEval 2.0:
   - M_1 SFT: 9.94%
   - M_2 (DPO sobre AIFT(M_1)): 15.38%
   - M_3 (DPO sobre AIFT(M_2)): 20.44%
2. `M_{t+1}` es **mejor juez** que `M_t` — medido por correlación de scores con anotaciones humanas (reportado en el paper sin entrenamiento explícito del rol juez).

**¿Por qué mejora el juez sin entrenarlo explícitamente?** Porque las representaciones internas para generar bien y para evaluar bien son **compartidas**. Mejor representación de calidad → mejores generaciones *y* mejores juicios. Este es el corazón del paradigma self-rewarding: la mejora se acopla.

[[summaries/wu-2024-meta-rewarding]] desacopla esto explícitamente entrenando el juez en un canal DPO separado (meta-judge), obteniendo mejoras más sostenidas.

## 5. ¿Por qué DPO y no PPO?

Diferencias prácticas en el contexto del loop iterativo:

| Aspecto | PPO (RLHF clásico) | DPO (Self-Rewarding) |
|---|---|---|
| Reward model | Separado, requiere entrenamiento explícito. | Implícito en la política. |
| Sampling | On-policy durante optimización. | Offline; pares se generan antes y se reutilizan. |
| Infraestructura | RL pesada (rollouts, value head, GAE). | Solo logits y backprop estándar. |
| Costo cómputo | ~3-5× más caro. | Línea base. |
| Estabilidad | Sensible a hiperparámetros, baseline. | Más estable; β es el único hiperparámetro nuevo. |
| Cierre del loop | Difícil de iterar. | Trivial — repetir muestreo + DPO. |

**El loop completo se puede ejecutar offline**: generar → juzgar → DPO → repetir. Sin infraestructura RL. Esta es la razón pragmática por la que Self-Rewarding emerge en 2024 y no antes: DPO (Mayo 2023, NeurIPS 2023) lo hizo viable.

## 6. Cómo argumentarlo en la presentación

Tres elementos para la slide:

1. **Pseudocódigo del loop completo** — destacar que la línea 4 es DPO, no PPO.
2. **Ecuación de la pérdida DPO** con β = 0.1 explícito y `π_ref = M_t`.
3. **Tabla comparativa DPO vs PPO** en el contexto iterativo — costo, estabilidad, viabilidad del cierre.

**Frase para defensa:**

> *"DPO es lo que hace posible Self-Rewarding. La equivalencia formal con RLHF KL-restringido garantiza que estamos optimizando el mismo objetivo, pero sin entrenar un reward model separado y sin sampling on-policy. El loop entero corre offline: generar, juzgar, DPO, repetir. Con β = 0.1 (Yuan 2024) controlamos cuánto puede divergir M_{t+1} de M_t en cada iteración."*

## 7. Detalles técnicos verificables

Hiperparámetros confirmados de Yuan 2024 (apéndice del paper):

- **DPO**: β = 0.1, LR inicial 1e-6 con decay a 1e-7, batch size 16, dropout 0.1, early stopping cada 200 steps evaluado con Claude 2 sobre 253 ejemplos de validación.
- **SFT** (etapa inicial): LR inicial 5.5e-6 → 1.1e-6 con cosine decay.
- **Generación**: T=0.7, top-p=0.9, N=4 respuestas por prompt.
- **Modelo base**: Llama-2-70B.

De [[summaries/wu-2024-meta-rewarding]] (apéndice):
- DPO: 10 épocas, LR 5e-6, β = 0.1.
- SFT: 10 épocas, LR 5e-8, batch global 32, cosine schedule, checkpoint época 5.
- 4 iteraciones sobre 20.000 prompts generados con Llama-2-70B-Chat 8-shot.

## 8. Variantes post-DPO relevantes para discusión

Aunque Yuan 2024 usa DPO estándar, vale mencionar (slide opcional o backup):

- **[[summaries/azar-2023-ipo]]** (IPO): identifica que DPO puede colapsar bajo preferencias cuasi-deterministas y propone una pérdida regularizada con ψ = identidad. Directamente relevante para diagnosticar la saturación de Self-Rewarding tras varias iteraciones.
- **KTO** (Ethayarajh 2024): elimina la necesidad de pares — usa solo etiquetas binarias deseable/no-deseable. Podría reducir la dependencia del juez interno en Self-Rewarding.
- **ORPO** (Hong 2024): elimina el modelo de referencia `π_ref`. Simplificación extrema.

Ver [[synthesis/comparacion-metodos]] (WIP) para la tabla comparativa completa.

## 9. Lecturas relacionadas

- [[concepts/dpo]] — derivación detallada de la pérdida desde el problema KL-restringido.
- [[concepts/kl-beta]] — análisis del coeficiente β.
- [[concepts/self-rewarding-loop]] — pseudocódigo completo del loop.
- [[summaries/rafailov-2023-dpo]] — paper original DPO.
- [[summaries/yuan-2024-self-rewarding]] — paper ancla con los hiperparámetros.
- [[synthesis/pilar-1-irl-implicito]] — equivalencia teórica con MaxEnt-IRL.
- [[synthesis/diagnostico-fallos]] — cómo el loop puede degradarse a pesar de DPO.
