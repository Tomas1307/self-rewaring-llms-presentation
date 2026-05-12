---
name: pilar-1-irl-implicito
description: Pilar 1 — argumentación de por qué Self-Rewarding constituye una instancia funcional de Inverse RL vía la equivalencia MaxEnt-IRL ↔ DPO.
metadata:
  type: synthesis
  tags: [pilar, irl, dpo, maxent, fundamentacion-teorica]
  last_updated: 2026-05-12
---

# Pilar 1 — IRL implícito en Self-Rewarding

> **Tesis del pilar.** Self-Rewarding no es IRL en el sentido estricto de Ng & Russell (2000), pero ejecuta una **recuperación implícita de recompensa** estructuralmente equivalente al óptimo MaxEnt-IRL con prior π_ref, donde el rol de "experto" lo cumple el propio modelo en su modo juez. Esta lectura está consolidada en la literatura RLHF moderna.

## 1. La pregunta honesta

El framing intuitivo dice: *"el modelo aprende una reward function implícita a partir de su propio comportamiento, en lugar de recibirla externamente"*. ¿Es eso IRL?

Hay dos respuestas:

- **IRL estricto, Ng-Russell 2000**: no. IRL clásico asume (i) un MDP conocido o muestreable, (ii) trayectorias de un **experto externo**, (iii) la meta es recuperar `r` tal que esas trayectorias sean óptimas respecto a `r`. Self-Rewarding no tiene experto externo y no busca explícitamente `r` como objeto matemático.
- **IRL funcional / preference-based**: sí. Lo que Self-Rewarding hace es indistinguible — vía la equivalencia DPO ↔ MaxEnt-IRL — de una recuperación de recompensa implícita.

Esta página argumenta la segunda respuesta y dice **cómo defenderla honestamente** en la presentación.

## 2. El puente formal: MaxEnt-IRL ↔ DPO

### 2.1. Forma exponencial de MaxEnt-IRL

[[summaries/ziebart-2008-maxent-irl]] propone, entre todas las distribuciones sobre trayectorias consistentes con las feature expectations del experto, elegir la de **máxima entropía**. La solución toma forma Boltzmann:

```
P(τ | w) = (1 / Z(w)) · exp( w⊤ Σ_t φ(s_t, a_t) )
```

donde:
- `τ` es una trayectoria.
- `φ(s, a)` son features.
- `w` es el vector de pesos (parametrización lineal de la recompensa).
- `Z(w)` es la función de partición.

### 2.2. Solución cerrada del RLHF KL-restringido

Del problema canónico de [[summaries/ouyang-2022-instructgpt]]:

```
max_π  E_{x,y~π}[r(x,y)]  −  β · KL(π ‖ π_ref)
```

La solución óptima es ([[summaries/rafailov-2023-dpo]], ecuación (4)):

```
π*(y|x) = (1 / Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```

### 2.3. Comparación

| MaxEnt-IRL (Ziebart 2008) | RLHF KL-óptimo (Rafailov 2023) |
|---|---|
| `P(τ) ∝ exp( w⊤ φ(τ) )` | `π*(y|x) ∝ π_ref(y|x) · exp( r(x,y) / β )` |
| Sin prior (entropía máxima absoluta) | Con prior `π_ref` |
| `w⊤ φ` es la recompensa lineal | `r(x,y) / β` juega el rol de "energy" |

**La forma es la misma.** MaxEnt-IRL con prior `π_ref` = exactamente la solución RLHF KL-restringida. Esto está dicho explícitamente en el [RLHF Book de Lambert, cap. 5](https://rlhfbook.com/c/05-reward-models): *"reward modeling in RLHF can be viewed as a kind of inverse RL"*.

### 2.4. DPO invierte esta ecuación

[[summaries/rafailov-2023-dpo]] despeja `r` en la ecuación de la solución óptima:

```
r(x,y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

Esto es la **recompensa implícita** asociada a la política. Al optimizar la pérdida DPO (clasificación binaria sobre pares con Bradley-Terry), el modelo `π_θ` se mueve de tal forma que la recompensa implícita inducida `r̂_θ(x,y) = β · log(π_θ(y|x) / π_ref(y|x))` explica los pares de preferencia observados.

**Por tanto: entrenar con DPO = recuperar una `r` implícita = IRL funcional.**

## 3. Self-Rewarding cierra el círculo

En RLHF clásico, los pares `(x, y_w, y_l)` provienen de humanos. Los pares se interpretan como "el experto humano prefiere `y_w` sobre `y_l`". DPO recupera la `r` implícita que mejor explica esas decisiones — IRL desde preferencias humanas.

En **Self-Rewarding** ([[summaries/yuan-2024-self-rewarding]]), los pares provienen del propio modelo actuando como juez. La interpretación cambia:

- El "experto" ya no es humano. Es el **mismo modelo en rol de juez** (LLM-as-a-Judge con rúbrica aditiva de 5 puntos, ver [[concepts/llm-as-a-judge]]).
- Las "preferencias" se construyen tomando `y_w = argmax_y s(y|x)` y `y_l = argmin_y s(y|x)`.
- DPO recupera la `r̂_θ` que mejor explica las decisiones de ese juez interno.

**Self-Rewarding es IRL del comportamiento de su propio juez.**

## 4. Caveats honestos

Estos puntos deben aparecer en la slide para no sobrevender la analogía:

1. **No es IRL estricto.** El "experto" no es externo. Esto rompe la justificación clásica de IRL (que la política observada sea efectivamente óptima).
2. **El experto interno no es óptimo y se mueve.** El juez `M_t` mejora con `t`, pero también arrastra los sesgos del modelo base. No hay garantía de que su política `π_J` sea la verdadera política óptima a recuperar.
3. **El sesgo de auto-preferencia es real y cuantificado** ([[summaries/zheng-2023-llm-as-judge]] documenta el "self-enhancement bias"; el paper arXiv:2410.21819 mide que GPT-4 prefiere sus propios outputs y lo vincula a baja perplexity). Esto se amplifica cuando juez y generador son el mismo modelo, como en Self-Rewarding.
4. **El loop puede colapsar.** [[summaries/wang-2025-temporal-sr]] muestra que la diferencia representacional `r̂(x, y_w) − r̂(x, y_l)` se estrecha con `t`, degradando la señal de aprendizaje. Es decir, la recuperación de `r` implícita se vuelve cada vez menos informativa.

## 5. Cómo argumentarlo en la presentación

Tres pasos en una slide ó dos:

1. **Mostrar la equivalencia formal** (ecuación cerrada KL-RLHF = forma MaxEnt-IRL).
2. **Reconocer la diferencia con IRL clásico**: experto interno, no externo. Llamarlo "IRL auto-referencial" o "IRL implícito".
3. **Citar la consolidación moderna**: [Lambert RLHF Book §5](https://rlhfbook.com/c/05-reward-models) ("reward modeling can be viewed as a kind of inverse RL"), [[summaries/wirth-2017-pbrl-survey]] (puente PbRL ↔ IRL), [[summaries/hejna-sadigh-2023-ipl]] (formaliza "preference-based RL sin reward function explícito").

**Frase para defensa:**

> *"Self-Rewarding no hace IRL en el sentido de Ng & Russell 2000, pero ejecuta una recuperación implícita de recompensa estructuralmente equivalente al óptimo MaxEnt-IRL con prior π_ref, donde el rol de experto lo cumple el propio modelo como juez. Esta es la lectura que la literatura RLHF moderna ha consolidado a partir de Rafailov 2023 y Ziebart 2008."*

## 6. Posibles objeciones y respuestas

- **"Si el experto interno no es óptimo, ¿qué garantiza convergencia?"** — Ninguna garantía clásica de IRL aplica. La justificación es empírica: Yuan 2024 reporta mejora en 3 iteraciones. La crítica empírica posterior ([[summaries/wang-2025-temporal-sr]], Shafayat 2025) muestra que sin diseño cuidadoso el loop colapsa.
- **"¿No es esto simplemente self-training disfrazado?"** — Self-training (e.g., [STaR](https://arxiv.org/abs/2203.14465)) filtra muestras correctas con un oráculo externo o autoconsistencia. Self-Rewarding genera la señal de filtrado internamente vía LLM-as-a-Judge. La diferencia clave es que aquí la señal *es* una preferencia explícita, no una etiqueta binaria.
- **"¿Por qué no usar IRL adversarial (GAIL)?"** — GAIL aprende `r` adversarialmente contra un discriminador. DPO logra lo mismo en forma cerrada al asumir Bradley-Terry, lo que es computacionalmente mucho más barato y estable. La presentación puede mencionarlo como contraste.

## 7. Lecturas relacionadas en el wiki

- [[concepts/irl]] — definiciones formales de IRL clásico, MaxEnt y preference-based.
- [[concepts/dpo]] — derivación detallada de la pérdida DPO desde el problema KL-restringido.
- [[concepts/preference-pair]] — modelo Bradley-Terry y construcción de pares.
- [[summaries/ziebart-2008-maxent-irl]] — paper que provee la forma exponencial.
- [[summaries/rafailov-2023-dpo]] — paper que invierte la ecuación.
- [[summaries/yuan-2024-self-rewarding]] — paper que cierra el círculo.
- [[synthesis/pilar-3-dpo-en-loop]] — cómo DPO se integra mecánicamente en la iteración.
