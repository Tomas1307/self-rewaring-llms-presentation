---
name: azar-2023-ipo
description: Resumen de IPO (Azar et al. 2023) — marco teórico general ΨPO y diagnóstico de por qué DPO puede colapsar, arXiv:2310.12036, AISTATS 2024.
metadata:
  type: summary
  tags: [dpo, ipo, teoria, preferencias, colapso, regularizacion]
  last_updated: 2026-05-12
---

# Azar et al. (2023) — A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO)

## Cita

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, Rémi Munos (DeepMind). *A General Theoretical Paradigm to Understand Learning from Human Preferences*. arXiv:2310.12036, octubre 2023. **AISTATS 2024**.

- **arXiv**: https://arxiv.org/abs/2310.12036

## Resumen técnico

Los autores identifican **dos aproximaciones implícitas** en RLHF/DPO que pueden fallar:

1. **Sustituir preferencias pairwise por recompensas pointwise** — asume el modelo de Bradley-Terry, lo cual no siempre se cumple.
2. **Asumir que un reward model entrenado generaliza out-of-distribution** — supone que la `r̂` aprendida es confiable más allá del soporte del dataset.

Para diagnosticar formalmente, derivan un objetivo general llamado **ΨPO** expresado directamente en términos de preferencias pairwise:

```
Objetivo ΨPO:  max_π  E_{(x, y_w, y_l) ~ D} [ Ψ( p*(y_w ≻ y_l | x) ) ]
              − β · KL( π ‖ π_ref )
```

donde:
- `Ψ`: función no-decreciente que mapea probabilidades de preferencia a "utilidad".
- `p*(y_w ≻ y_l | x)`: probabilidad de preferencia bajo la política aprendida.

**Casos particulares**:
- `Ψ(p) = log(p / (1 − p))` → DPO (asume Bradley-Terry implícitamente).
- `Ψ(p) = p` (identidad) → **IPO** (Identity Preference Optimization).

**Hallazgo clave**: cuando las preferencias son **cuasi-deterministas** (`p* → 1` o `p* → 0`), el `Ψ = log(p/(1-p))` de DPO **diverge**, causando que la pérdida sobreajuste agresivamente:

- El log-ratio `log(π_θ(y_w)/π_ref(y_w))` crece sin acotar.
- La política se aleja arbitrariamente de la referencia.
- La regularización KL implícita se rompe.

IPO con `Ψ = identidad` no tiene este problema: la pérdida regulariza implícitamente y evita el colapso de KL.

## Aporte clave para la presentación

Llena la **laguna teórica** sobre **por qué DPO puede colapsar** bajo varias iteraciones de Self-Rewarding. Relevante porque:

1. En Self-Rewarding, los pares `(y_w, y_l)` vienen del propio modelo. Después de varias iteraciones, el modelo puede generar respuestas **muy distinguibles** desde el punto de vista del juez interno → preferencias cuasi-deterministas (`s(y_w)=5, s(y_l)=0`).
2. Este es precisamente el régimen donde DPO falla según Azar 2023.
3. **IPO regulariza implícitamente** y podría usarse como reemplazo de DPO en el loop para evitar el sobreajuste tardío.

Útil para [[synthesis/diagnostico-fallos]] §2.1 y como variante a discutir en backup slides.

## Fórmula central

**Pérdida IPO** (simplificada):

```
L_IPO(π_θ) = E_{(x, y_w, y_l) ~ D} [
    (
        log( π_θ(y_w | x) · π_ref(y_l | x) / [ π_θ(y_l | x) · π_ref(y_w | x) ] )
        − 1 / (2 β)
    )^2
]
```

**Símbolos**:
- `π_θ`, `π_ref`: igual que en DPO.
- `β`: coeficiente de regularización (juega el mismo rol que en DPO).
- El término cuadrático **penaliza** que el log-ratio se aleje del valor objetivo `1/(2β)` — esto regulariza implícitamente y evita la divergencia.

**Comparación intuitiva**:
- DPO: `−log σ(β·diff)` — saturated cuando diff es grande, pero `π_θ` puede divergir libremente.
- IPO: `(diff − 1/(2β))²` — penaliza explícitamente desviaciones grandes; mantiene `π_θ` cerca de `π_ref`.

## Resultados empíricos

El paper enfoca el análisis teórico. Empíricamente muestra:

- En preferencias sintéticas casi-deterministas, DPO diverge mientras IPO converge.
- En tareas reales (TL;DR summarization), IPO iguala o supera a DPO con menor varianza entre seeds.

(Detalles numéricos en el paper original — apéndice AISTATS 2024.)

## Limitaciones

1. **Asume conocer `p*`** o tener acceso al dataset de preferencias en cantidad suficiente.
2. **No resuelve completamente** el problema de generalización OOD del reward implícito.
3. **β sigue siendo crítico** y debe ajustarse.

## Variantes relacionadas (línea post-DPO)

- **KTO** (Ethayarajh 2024, arXiv:2402.01306): alineación basada en teoría prospectiva; no requiere pares.
- **ORPO** (Hong 2024, arXiv:2403.07691, EMNLP 2024): elimina `π_ref` y combina SFT + alineación en un paso.

Ver `synthesis/comparacion-metodos.md` (WIP) para la tabla comparativa completa.

## Conexiones en el wiki

- [[summaries/rafailov-2023-dpo]] — paper que IPO generaliza.
- [[concepts/dpo]] — derivación detallada de DPO; relevante para contraste.
- [[synthesis/pilar-3-dpo-en-loop]] — cómo β y el reference model evolucionan en el loop.
- [[synthesis/diagnostico-fallos]] — el colapso DPO que IPO diagnostica formalmente.
- [[summaries/yuan-2024-self-rewarding]] — el loop donde el colapso aparece tras 3 iteraciones.
