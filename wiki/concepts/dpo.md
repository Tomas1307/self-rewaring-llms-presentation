---
name: dpo
description: Direct Preference Optimization — derivación completa de la pérdida desde el problema KL-restringido de RLHF, intuición y rol de β.
metadata:
  type: concept
  tags: [dpo, optimizacion, preferencias, kl, bradley-terry]
  last_updated: 2026-05-12
---

# DPO — Direct Preference Optimization

> Pérdida de clasificación binaria que entrena una política sobre pares de preferencia `(x, y_w, y_l)` sin entrenar un reward model separado ni usar PPO. Fuente: [[summaries/rafailov-2023-dpo]].

## 1. El problema que resuelve

RLHF clásico (estilo InstructGPT) tiene tres etapas:
1. **SFT** sobre demostraciones humanas.
2. **Entrenar un reward model `r̂(x,y)`** sobre rankings humanos.
3. **PPO** maximizando `r̂(x,y) − β · KL(π ‖ π_ref)`.

Problemas:
- Reward model puede ser hackeado por la política durante PPO (reward hacking).
- PPO requiere sampling on-policy, infraestructura compleja, hiperparámetros sensibles.
- Cómputo: ~3-5× el costo de SFT.

**DPO elimina las etapas 2 y 3** y las reemplaza por una sola pérdida cerrada sobre los pares de preferencia.

## 2. Derivación en tres pasos

### Paso 1 — Problema KL-restringido

El objetivo RLHF canónico es:
```
max_π  E_{x,y~π} [ r(x,y) ]  −  β · KL( π ‖ π_ref )
```

**Símbolos**:
- `π`: política a optimizar.
- `π_ref`: política de referencia (típicamente el modelo SFT).
- `r(x,y)`: recompensa (desconocida, pero implícita en las preferencias).
- `β`: coeficiente KL (penaliza divergir de `π_ref`).

### Paso 2 — Solución cerrada

Usando multiplicadores de Lagrange:
```
π*(y | x) = (1 / Z(x)) · π_ref(y | x) · exp( r(x,y) / β )
```

donde `Z(x) = Σ_y π_ref(y|x) · exp(r(x,y)/β)` es la función de partición (intratable directamente porque suma sobre todo el espacio de respuestas).

### Paso 3 — Inversión

Despejar `r`:
```
r(x,y) = β · log( π*(y | x) / π_ref(y | x) ) + β · log Z(x)
```

Insertar en el **modelo Bradley-Terry**:
```
P(y_w ≻ y_l | x) = σ( r(x, y_w) − r(x, y_l) )
                 = σ( β · log[π*(y_w|x) / π_ref(y_w|x)]
                      − β · log[π*(y_l|x) / π_ref(y_l|x)]
                      + β · [log Z(x) − log Z(x)] )
                 = σ( β · log[π_θ(y_w|x)/π_ref(y_w|x)]
                      − β · log[π_θ(y_l|x)/π_ref(y_l|x)] )
```

**Punto clave**: `log Z(x)` se cancela. La pérdida solo depende de `π_θ` y `π_ref`.

## 3. La pérdida DPO

Maximizar la log-verosimilitud del modelo Bradley-Terry sobre el dataset de preferencias `D`:

```
L_DPO(π_θ; π_ref) = − E_{(x, y_w, y_l) ~ D} [
    log σ(
        β · log( π_θ(y_w | x) / π_ref(y_w | x) )
      − β · log( π_θ(y_l | x) / π_ref(y_l | x) )
    )
]
```

**Computacionalmente**: solo cuatro forward passes por par (dos por `π_θ`, dos por `π_ref`).

## 4. Recompensa implícita

La política `π_θ` codifica una recompensa implícita:
```
r̂_θ(x, y) = β · log( π_θ(y | x) / π_ref(y | x) )
```

**Intuición**: `r̂` es alto cuando `π_θ` asigna más probabilidad a `y` que `π_ref`. La política y el reward son **la misma red**.

## 5. El rol de β

`β` controla cuánto puede divergir `π_θ` de `π_ref`:

| β | Comportamiento | Efecto |
|---|---|---|
| Alto (≥ 0.5) | Conservador. | `π_θ` casi igual a `π_ref`. |
| Medio (0.1) | Estándar. | Yuan 2024 usa este valor. |
| Bajo (≤ 0.05) | Agresivo. | `π_θ` puede alejarse mucho; riesgo de drift / reward hacking. |

Ver [[concepts/kl-beta]] (WIP) para análisis completo.

## 6. Intuición del gradiente

```
∇_θ L_DPO ∝  − σ( β·[r̂_θ(y_w) − r̂_θ(y_l)] )
              · β · ( ∇_θ log π_θ(y_w|x) − ∇_θ log π_θ(y_l|x) )
```

**Interpretación**:
- Si `r̂(y_w) > r̂(y_l)`: `σ(diff) ≈ 1`, gradiente pequeño → "ya estoy bien, ajusto poco".
- Si `r̂(y_w) ≈ r̂(y_l)`: `σ(0) = 0.5`, gradiente moderado → "necesito separar más".
- Si `r̂(y_w) < r̂(y_l)`: `σ(diff) → 0`, gradiente grande → "estoy muy mal, corrige fuerte".

**Punto crítico para el loop Self-Rewarding**: si tras varias iteraciones `r̂(y_w) ≈ r̂(y_l)` (porque ambas respuestas mejoran juntas), el gradiente se anula. Esto es exactamente la saturación que [[summaries/wang-2025-temporal-sr]] documenta.

## 7. Comparación con PPO en una tabla

| Aspecto | PPO (RLHF clásico) | DPO |
|---|---|---|
| Reward model | Separado, entrenado explícitamente. | Implícito en `π_θ`. |
| Sampling | On-policy. | Offline (los pares se generan antes). |
| Infraestructura | RL completa (rollouts, value head, GAE). | Solo logits + backprop. |
| Costo | ~3-5× SFT. | ~SFT. |
| Estabilidad | Sensible a hiperparams. | Estable; β como único hiperparam nuevo. |
| Cierre de loop iterativo | Difícil. | Trivial. |

## 8. Variantes posteriores

- **IPO** ([[summaries/azar-2023-ipo]]): regulariza con `Ψ = identidad` para evitar colapso bajo preferencias deterministas.
- **KTO** (Ethayarajh 2024): elimina necesidad de pares — solo etiquetas binarias.
- **ORPO** (Hong 2024): elimina `π_ref`, combina SFT + alineación.
- **SPPO** (Wu 2024): formaliza como juego de suma constante con equilibrio de Nash.

## 9. Limitaciones

1. **Asume Bradley-Terry**. Falla con preferencias intransitivas.
2. **Pares ruidosos colapsan la señal** (gradiente → 0).
3. **β crítico**. Ajustar mal degrada calidad.
4. **`π_ref` debe ser razonable**. Si es pobre, DPO hereda el sesgo.
5. **No es on-policy**: distribution drift en regímenes muy off-policy.

## 10. Lecturas relacionadas

- [[summaries/rafailov-2023-dpo]] — paper original.
- [[concepts/kl-beta]] — análisis del β.
- [[concepts/preference-pair]] — Bradley-Terry y construcción de pares.
- [[synthesis/pilar-3-dpo-en-loop]] — cómo DPO encaja en Self-Rewarding.
- [[synthesis/pilar-1-irl-implicito]] — equivalencia con MaxEnt-IRL.
- [[summaries/azar-2023-ipo]] — diagnóstico del colapso de DPO.
