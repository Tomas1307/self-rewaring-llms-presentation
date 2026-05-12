---
name: pilar-2-multi-agente-cooperativo
description: Pilar 2 — argumentación de que Self-Rewarding constituye un common-payoff cooperative game entre dos sub-agentes lógicos del mismo modelo.
metadata:
  type: synthesis
  tags: [pilar, multi-agent, cooperative, marl, framing]
  last_updated: 2026-05-12
---

# Pilar 2 — Multi-agente cooperativo

> **Tesis del pilar.** Self-Rewarding instancia un **common-payoff cooperative game** entre dos sub-agentes lógicos del mismo modelo: el **generador** (actor) y el **juez** (crítico). Ambos comparten función de pérdida; cada iteración mejora a ambos. La literatura post-Yuan (CORY, Multi-Agent Evolve, Prospector, Meta-Rewarding) lo confirma como abstracción formal, no metáfora.

## 1. La pregunta

El framing intuitivo dice: *"aunque es un solo modelo, opera como dos agentes cooperando — uno genera y otro evalúa, mejorándose mutuamente en cada iteración"*. ¿Es realmente multi-agente cooperativo, o es analogía conceptual?

Respuesta: **es analogía formalizable**, con respaldo teórico vía common-payoff games y respaldo empírico vía la literatura LLM-MARL 2024-2025.

## 2. Los dos sub-agentes

En cada iteración el mismo backbone θ se invoca con **dos contextos distintos**:

| Agente | Rol | Política inducida | Objetivo |
|---|---|---|---|
| **A** (actor / generador) | Genera `y` dado `x`. | `π_A(y \| x) = M_θ(y \| x, prompt_generación)` | Maximizar el reward implícito (= producir respuestas que el juez aprobaría). |
| **J** (judge / crítico) | Asigna score `s(y \| x)`. | `π_J(score \| x, y) = M_θ(score \| x, y, prompt_juez)` | Que sus puntajes correlacionen con calidad real (juzgada por humanos / benchmarks). |

Aunque comparten parámetros θ, son **funcionalmente distinguibles**: distinto prompt, distinto espacio de salida, distinta señal de mejora.

Esto es exactamente la situación de **parameter sharing en cooperative MARL**: dos agentes con la misma red pero distinta política inducida por el rol/contexto.

## 3. ¿Cooperative o competitive?

### 3.1. Argumento cooperativo (dominante)

Ambos sub-agentes comparten la pérdida global de alineación. Específicamente:

- Si `J` mejora (juicio más correlacionado con humanos), `A` recibe pares de preferencia mejor curados → entrena mejor.
- Si `A` mejora (mejores respuestas), `J` recibe muestras más informativas → afina su rúbrica representacional.

Esto es **win-win**, no zero-sum. Encaja con la definición formal de **Common Payoff Markov Game** (también conocido como Multi-Agent MDP, MMDP):

```
Tupla (N, S, {A_i}, P, r) donde:
  - N agentes,
  - estado conjunto s ∈ S,
  - acciones a_i ∈ A_i por agente,
  - transición P(s' | s, a_1, ..., a_N),
  - recompensa COMPARTIDA r(s, a_1, ..., a_N).
```

En Self-Rewarding `N=2`, la recompensa compartida es "calidad de la política conjunta post-iteración" (medida implícitamente por la mejora en AlpacaEval / MT-Bench).

### 3.2. Argumento competitivo (parcial)

Algunos autores leen self-play como suma constante:

- **[[summaries/chen-2024-spin]] (SPIN)**: el "main player" π_θ debe distinguir sus propias generaciones (del "opponent" π_{t-1}) de respuestas humanas. Es un juego de discriminación contra una versión pasada de sí mismo.
- **Wu et al. 2024 (SPPO, arXiv:2405.00675)**: formaliza la alineación como **juego de suma constante de dos jugadores** buscando equilibrio de Nash de von Neumann mediante actualizaciones exponential weights. Provee garantías de convergencia bajo preferencias potencialmente intransitivas.
- **[[summaries/goodfellow-2014-gan]]**: dos redes (generador G, discriminador D) compiten minimax.

Pero esta lectura **no aplica limpiamente a Self-Rewarding original**:

- No hay zero-sum: si `J` mejora, `A` mejora; no hay "perdedor".
- No se busca equilibrio adversarial; al contrario, hay riesgo de *cooperative collapse* (ambos convergen a explotar el mismo sesgo — ver [[synthesis/diagnostico-fallos]]).
- El paper de Meta-Rewarding [[summaries/wu-2024-meta-rewarding]] refuerza la lectura cooperativa al **separar las señales de mejora del actor y del juez en dos canales DPO independientes** — diseño consistente con cooperación entre sub-agentes especializados, no con competencia.

### 3.3. Veredicto

- **Self-Rewarding original**: cooperative common-payoff game.
- **SPIN**: competitive self-play.
- **SPPO**: competitive con equilibrio formalizado.
- **Meta-Rewarding**: cooperative explícito multi-canal (tres roles: actor, judge, meta-judge).
- **GAN**: contraste útil para precisar qué *no* es Self-Rewarding (no es adversarial puro).

## 4. Respaldo en la literatura 2024-2025

La lectura "Self-Rewarding = multi-agent cooperativo" es una abstracción aceptada, no una metáfora pedagógica. Evidencia:

- **CORY** ("Coevolving with the Other You", arXiv:2410.06101): duplica el LLM en dos agentes ("pioneer" y "observer") que intercambian roles periódicamente; entrenamiento con MARL cooperativo; supera PPO en estabilidad y resistencia a *distribution collapse*.
- **Multi-Agent Evolve (MAE)** (arXiv:2510.23595): un único backbone LLM cumple tres roles (Proposer, Solver, Judge) — exactamente la generalización de Meta-Rewarding al caso de tres agentes lógicos.
- **ACC-Collab** (Estornell et al., arXiv:2411.00053): framework actor-critic explícito con dos agentes LLM especializados entrenados conjuntamente.
- **[[summaries/kim-2024-prospector]]**: Actor-Critic con dos LLMs distintos (Actor genera, Critic ranks trayectorias) — peer-reviewed en EMNLP 2024 Findings. Confirma el patrón.
- **[[summaries/wu-2024-meta-rewarding]]**: diseño explícito con tres roles (actor, judge, meta-judge) — extensión natural del framing cooperativo de Self-Rewarding.

## 5. Diferencia con actor-critic clásico

En RL clásico:

- El **crítico** estima `V(s)` o `Q(s,a)` — un escalar.
- El **actor** se actualiza con gradiente ponderado por la ventaja `A(s,a) = Q(s,a) − V(s)`.

En Self-Rewarding:

- El "crítico" (juez) **produce preferencias `(y_w, y_l)`** — no estima una función de valor.
- El "actor" (generador) se actualiza con **DPO sobre esos pares** — no con policy gradient ponderado.

Es más rico que actor-critic clásico: el crítico genera la **estructura completa de preferencia**, no solo una estimación escalar.

## 6. Cómo argumentarlo en la presentación

Tres puntos en una slide:

1. **Definir formalmente**: Self-Rewarding instancia un Common Payoff Markov Game con `N=2` sub-agentes y parameter sharing.
2. **Distinguir de adversarial**: no es GAN ni SPIN. La pérdida es compartida; el equilibrio es win-win, no Nash.
3. **Citar respaldo formal**: CORY, MAE, Meta-Rewarding como instancias de la misma abstracción.

**Frase para defensa:**

> *"Self-Rewarding es un common-payoff cooperative game entre dos sub-agentes del mismo modelo. Aunque comparten parámetros (parameter sharing), sus políticas inducidas π_A y π_J son distinguibles por el contexto y se optimizan con señales separadas. La literatura post-2024 — Meta-Rewarding, CORY, Multi-Agent Evolve, Prospector — confirma este framing y lo extiende."*

## 7. Objeción anticipada

**"¿No es esto un único agente con dos roles, no dos agentes?"**

Respuesta: La distinción entre "un modelo con dos prompts" y "dos agentes" es matemáticamente fluida. Lo que cuenta es:

- Políticas inducidas distinguibles (sí: `π_A ≠ π_J`).
- Señales de mejora separables (sí: en Meta-Rewarding son canales DPO independientes; en Yuan original están entrelazadas pero observables — la habilidad del juez mejora aunque no se entrene explícitamente).
- Espacios de acción distintos (sí: `A` produce texto continuación; `J` produce score numérico + crítica).

Es exactamente la lógica con la que cooperative MARL trata agentes con parameter sharing: las redes comparten pesos pero las políticas son funcionalmente distintas. La frontera entre "un agente con dos modos" y "dos agentes" es convencional, no esencial.

## 8. Posible extensión teórica

Si la presentación quiere ser ambiciosa, puede presentar el setup como un **Decentralized POMDP (Dec-POMDP)** o más limpiamente como **Common Payoff Markov Game**:

- Estado: contexto conversacional `x` + respuesta parcial.
- Agentes: `A` (genera tokens), `J` (evalúa al final del rollout).
- Acciones: `A` ∈ vocabulario, `J` ∈ {0,...,5}.
- Recompensa compartida: éxito en la iteración (mejora de AlpacaEval).

Esto es formalismo opcional para una audiencia rigurosa; el framing "common-payoff" sin Dec-POMDP es suficiente para la audiencia del curso.

## 9. Lecturas relacionadas

- [[concepts/cooperative-marl]] — formalismo de common-payoff games.
- [[concepts/self-rewarding-loop]] — el algoritmo concreto donde los dos sub-agentes interactúan.
- [[summaries/wu-2024-meta-rewarding]] — extensión a tres roles, valida el framing.
- [[summaries/kim-2024-prospector]] — Actor-Critic LLM explícito como referencia.
- [[synthesis/pilar-1-irl-implicito]] — qué aprende cada sub-agente desde la perspectiva IRL.
- [[synthesis/diagnostico-fallos]] — cooperative collapse como modo de fallo.
