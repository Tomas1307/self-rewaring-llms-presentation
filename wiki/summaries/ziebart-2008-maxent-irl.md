---
name: ziebart-2008-maxent-irl
description: Resumen de Maximum Entropy Inverse Reinforcement Learning (Ziebart et al. 2008) — el puente teórico formal con DPO/RLHF, AAAI 2008.
metadata:
  type: summary
  tags: [irl, maxent, fundamentos, puente-dpo, recompensa-implicita]
  last_updated: 2026-05-12
---

# Ziebart et al. (2008) — Maximum Entropy Inverse Reinforcement Learning

## Cita

Brian D. Ziebart, Andrew Maas, J. Andrew Bagnell, Anind K. Dey (Carnegie Mellon University). *Maximum Entropy Inverse Reinforcement Learning*. **AAAI 2008**.

- **PDF (AAAI)**: https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf

## Resumen técnico

Resuelve la **ambigüedad fundamental de IRL** identificada por Ng & Russell 2000 y Abbeel & Ng 2004: dado un conjunto de trayectorias del experto, **muchas funciones de recompensa** son compatibles (incluso `r ≡ 0`). ¿Cuál elegir?

**Principio MaxEnt** (Jaynes): entre todas las distribuciones sobre trayectorias consistentes con las *feature expectations* observadas, **elegir la de máxima entropía**. Es la elección que añade el mínimo de información extra; cualquier otra distribución codifica supuestos no justificados por los datos.

Bajo este principio, la distribución óptima sobre trayectorias `τ` toma forma exponencial / Boltzmann:

```
P(τ | w) = (1 / Z(w)) · exp( w⊤ Σ_t φ(s_t, a_t) )
```

donde `w⊤ Σ_t φ(s_t, a_t)` es la recompensa acumulada lineal en features y `Z(w)` es la función de partición.

El **objetivo de aprendizaje** es la log-verosimilitud de las trayectorias del experto:
```
max_w  Σ_{τ ∈ D_expert}  log P(τ | w)
```

Con gradiente:
```
∇_w L  =  Σ_{τ ∈ D_expert} Σ_t φ(s_t, a_t)
        −  E_{τ ~ P(τ | w)} [ Σ_t φ(s_t, a_t) ]
```

Es decir: **feature matching estocástico** — ajustar `w` para que la *expectativa* de features bajo la política inducida iguale la del experto.

## Aporte clave para la presentación

Es **el puente formal con DPO/RLHF**. Sin este paper, la afirmación "DPO es IRL implícito" sería pura analogía. Ziebart provee:

1. **La forma cerrada exponencial** `P ∝ exp(reward)` — estructuralmente idéntica a la solución cerrada de [[summaries/rafailov-2023-dpo]] ecuación (4):
   ```
   π*(y|x) ∝ π_ref(y|x) · exp( r(x,y) / β )
   ```
   La única diferencia: la versión RLHF tiene **prior `π_ref`** (medida base no-uniforme); la versión MaxEnt original tiene prior uniforme.

2. **El argumento de optimalidad**: MaxEnt es la distribución *menos comprometida* compatible con los datos. Esto da legitimidad teórica a DPO/RLHF como instancia de IRL.

3. **El concepto de feature matching**: aparece en RLHF como "matching de preferencias humanas" — la política aprendida debe inducir las mismas preferencias que el experto.

Sin Ziebart 2008, [[synthesis/pilar-1-irl-implicito]] sería defendible solo por analogía superficial. Con él, la conexión es **formal**.

## Fórmula central — comparación lado a lado

| MaxEnt-IRL (Ziebart 2008) | RLHF KL-óptimo (Rafailov 2023) |
|---|---|
| `P(τ) ∝ exp( w⊤ φ(τ) )` | `π*(y \| x) ∝ π_ref(y \| x) · exp( r(x,y) / β )` |
| Sin prior (entropía máxima absoluta) | Con prior `π_ref` (medida base) |
| `w⊤ φ` lineal en features | `r(x,y)` arbitrario, parametrizado por el modelo |
| `Z(w)` partición global | `Z(x)` partición condicional al prompt |

**Equivalencia**: si tomamos `π_ref` = distribución uniforme y `r(x,y) = w⊤ φ(x,y)`, la solución RLHF se reduce exactamente a la solución MaxEnt-IRL.

## Resultados empíricos

El paper aplica MaxEnt-IRL a **predicción de rutas de conductores** en una red de carreteras (Pittsburgh). Logra mejor predicción de rutas que IRL clásico no-MaxEnt.

(Los resultados específicos no son centrales para nuestra presentación — lo que importa es la formulación teórica.)

## Limitaciones

1. **Cómputo de `Z` es intratable en general.** Requiere algoritmos forward-backward dinámicos (solo viables para MDPs pequeños) o aproximaciones (sampling, variational). En el contexto LLM esto se resuelve elegantemente: la función de partición se *cancela* al pasar a Bradley-Terry sobre pares de preferencia (ver derivación DPO).

2. **Asume MDP conocido** o trayectorias densas. RLHF no asume MDP explícito — opera directamente sobre pares.

3. **Recompensa lineal en features** en la versión original. Deep MaxEnt IRL (Wulfmeier 2015 y sucesores) extiende a recompensas no lineales — paralela a lo que RLHF hace con redes neuronales como reward model.

## Cómo usarlo en la presentación

Sugerencia: una slide dedicada al **puente teórico** con dos ecuaciones lado a lado (MaxEnt vs RLHF) y una flecha que indica equivalencia.

**Frase tipo**:
> *"La política óptima de RLHF (Rafailov 2023, ec. 4) tiene la forma `π* ∝ π_ref · exp(r/β)`. Esto es exactamente la distribución MaxEnt-IRL de Ziebart 2008 con prior `π_ref`. Por tanto, optimizar con DPO recupera implícitamente una función de recompensa: estamos haciendo IRL sin entrenar un reward model explícito."*

## Conexiones en el wiki

- [[concepts/irl]] — IRL clásico (Ng-Russell), MaxEnt y preference-based.
- [[summaries/ng-russell-2000-irl]] — IRL original que motiva MaxEnt.
- [[summaries/abbeel-ng-2004-apprenticeship]] — feature matching que MaxEnt formaliza.
- [[summaries/rafailov-2023-dpo]] — donde aparece la forma `exp(r/β)` en RLHF.
- [[synthesis/pilar-1-irl-implicito]] — argumentación completa del puente.
- [[summaries/wirth-2017-pbrl-survey]] — preference-based RL como evolución de IRL.
- [[summaries/hejna-sadigh-2023-ipl]] — Inverse Preference Learning, conexión moderna del puente.
