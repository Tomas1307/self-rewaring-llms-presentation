---
name: irl
description: Inverse Reinforcement Learning — formulación clásica (Ng-Russell), MaxEnt (Ziebart), y conexión con preference-based / DPO.
metadata:
  type: concept
  tags: [irl, maxent, ng-russell, ziebart, fundamentos]
  last_updated: 2026-05-12
---

# Inverse Reinforcement Learning

> Dado un MDP sin recompensa y trayectorias de un experto, **recuperar la función de recompensa** que explica esas trayectorias. Marco clásico iniciado por Ng & Russell 2000. Conecta con RLHF/DPO vía MaxEnt-IRL.

## 1. El problema

**RL forward**: dado `r`, encontrar `π*` que maximiza `E[Σ γ^t r(s_t, a_t)]`.

**RL inverso**: dado `π*` (o trayectorias del experto), encontrar `r` tal que `π*` sea óptima respecto a `r`.

Motivación práctica:
- Diseñar `r` "a mano" es difícil — reward shaping mal hecho induce comportamiento indeseado.
- Si tenemos un experto que ya sabe hacer la tarea, es más fácil aprender qué *valora* que diseñarlo explícitamente.

## 2. Las tres formulaciones clave

### 2.1. Ng & Russell 2000 (clásico)

[[summaries/ng-russell-2000-irl]]: formulación como **programa lineal**.

Dado un MDP finito y la política experta `π_E` totalmente conocida, encontrar `r` que maximiza el margen entre la política experta y la siguiente mejor acción:

```
max_r   Σ_s [ Q^π(s, π_E(s)) − max_{a ≠ π_E(s)} Q^π(s, a) ]  −  λ ‖r‖_1

s.t.   (P_{π_E(s)} − P_a) (I − γ P_π)^{-1} r  ≥  0,   ∀ s, a ≠ π_E(s)
```

**Problemas**:
- Ambigüedad: `r ≡ 0` siempre es una solución. Hay infinitas `r` compatibles.
- Asume política experta totalmente conocida (raro en la práctica).
- No escala a estados continuos.

### 2.2. Abbeel & Ng 2004 (apprenticeship learning)

[[summaries/abbeel-ng-2004-apprenticeship]]: en lugar de recuperar `r` explícitamente, **igualar feature expectations** del experto.

Asume `r(s) = w⊤ φ(s)` lineal. Define `μ(π) = E[Σ_t γ^t φ(s_t)]`. Objetivo:
```
encontrar π tal que  ‖μ(π_E) − μ(π)‖_2  ≤  ε,  con ‖w‖_2 ≤ 1
```

Iterar: actualizar `w` (maximiza margen entre experto y políticas previas) → entrenar `π` con RL contra `w` → repetir.

**Aporte**: introduce **feature matching**, que reaparece en RLHF como "matching de preferencias humanas".

### 2.3. Ziebart 2008 (Maximum Entropy IRL)

[[summaries/ziebart-2008-maxent-irl]]: resuelve la ambigüedad eligiendo la **distribución de máxima entropía** consistente con las feature expectations.

```
P(τ | w) = (1 / Z(w)) · exp( w⊤ Σ_t φ(s_t, a_t) )
```

Es la pieza que **conecta IRL con DPO/RLHF**. La forma exponencial `P ∝ exp(reward)` aparece literal en la solución cerrada KL-RLHF de Rafailov 2023.

## 3. Preference-Based IRL — el puente con LLMs

**El problema**: IRL clásico asume trayectorias del experto. Pero los humanos a menudo no pueden producir trayectorias óptimas — solo pueden **comparar** dos opciones y decir cuál es mejor.

**Preference-Based IRL** (Wirth 2017, JMLR) generaliza: el dataset no es trayectorias `{τ_E}` sino **pares comparados** `{(τ_i ≻ τ_j)}`. Bajo Bradley-Terry:
```
P(τ_i ≻ τ_j) = σ( U(τ_i) − U(τ_j) )
```

Maximizar la log-verosimilitud de las preferencias da `U` (o `r` si `U(τ) = Σ r(s,a)`).

**Esto es exactamente lo que hacen DPO y RLHF**: aprender `U` (= `r̂` implícita) a partir de pares `(y_w, y_l)`. RLHF en LLMs es preference-based IRL aplicado a tokens.

## 4. La cadena formal IRL → MaxEnt → RLHF → DPO

```
1. Ng & Russell 2000     →  Plantea el problema IRL
                              (recuperar r a partir de comportamiento).

2. Abbeel & Ng 2004      →  Feature matching:
                              μ(π_E) ≈ μ(π).

3. Ziebart 2008          →  MaxEnt resuelve ambigüedad:
                              P(τ) ∝ exp(w⊤φ(τ)).

4. Wirth 2017 (JMLR)     →  PbRL formaliza preferencias
                              como input en lugar de trayectorias.

5. Christiano 2017       →  Deep PbRL: entrena reward
                              model neural sobre preferencias.

6. Ouyang 2022           →  Aplica a LLMs (InstructGPT):
                              SFT + RM + PPO con KL.

7. Rafailov 2023 (DPO)   →  Cierra el círculo:
                              expresa r implícitamente en π,
                              recupera la forma MaxEnt
                              con prior π_ref.

8. Yuan 2024             →  Self-Rewarding cierra
                              el círculo entero:
                              el "experto" es el propio modelo.
```

## 5. Por qué Self-Rewarding es "IRL implícito"

Self-Rewarding ejecuta IRL **funcional**, aunque no IRL clásico:

- **El experto** no es humano externo, sino el propio modelo en rol de juez.
- **Las trayectorias** no son demos, sino respuestas `y` ranqueadas por preferencia.
- **La recompensa** no se entrena explícitamente, sino que emerge implícita en `π_θ`.
- **El objetivo** es recuperar la `r̂` que mejor explica las preferencias del juez interno.

Es IRL **auto-referencial**: el modelo recupera la recompensa que justifica su propio comportamiento. Ver [[synthesis/pilar-1-irl-implicito]] para la argumentación completa.

## 6. Caveats — no es IRL en sentido estricto

Tres diferencias con IRL clásico (Ng-Russell):

1. **No hay MDP definido**. Un LLM autoregresivo es un MDP de horizonte finito sobre tokens, pero la recompensa es por trayectoria completa (respuesta entera), no por estado.
2. **No hay experto externo**. El "experto" es interno y mejora con `t`.
3. **No se busca `r` como objeto matemático**. La `r̂` queda implícita en `π_θ`.

Por eso es importante en la presentación llamar al pilar 1 "**IRL implícito**" o "**IRL funcional**", no IRL a secas. Honestidad académica.

## 7. Variantes modernas relacionadas

- **Inverse Preference Learning (IPL)** ([[summaries/hejna-sadigh-2023-ipl]], NeurIPS 2023): preference-based RL sin función de recompensa explícita. Análogo directo a DPO en robótica.
- **Bayesian IRL** (Ramachandran & Amir 2007): inferencia probabilística sobre el espacio de recompensas.
- **GAIL** (Ho & Ermon 2016): IRL adversarial; usa un discriminador como reward implícito. Análogo conceptual a GAN.
- **Deep MaxEnt IRL** (Wulfmeier 2015): extiende Ziebart a recompensas no lineales con redes profundas.

## 8. Lecturas relacionadas

- [[summaries/ng-russell-2000-irl]] — formulación clásica.
- [[summaries/abbeel-ng-2004-apprenticeship]] — feature matching.
- [[summaries/ziebart-2008-maxent-irl]] — MaxEnt, puente con DPO.
- [[summaries/wirth-2017-pbrl-survey]] — preference-based RL pre-LLM.
- [[summaries/hejna-sadigh-2023-ipl]] — preference-based RL sin reward explícito.
- [[summaries/rafailov-2023-dpo]] — DPO como instancia LLM de la cadena.
- [[summaries/yuan-2024-self-rewarding]] — cierre auto-referencial.
- [[synthesis/pilar-1-irl-implicito]] — argumentación de la analogía.
