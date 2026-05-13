---
name: rlhf
description: Reinforcement Learning from Human Feedback — pipeline canónico de tres etapas (SFT → reward modeling → PPO) popularizado por InstructGPT/ChatGPT.
metadata:
  type: concept
  tags: [rlhf, ppo, reward-model, sft, alineamiento, pipeline-canonico]
  last_updated: 2026-05-12
---

# RLHF — Reinforcement Learning from Human Feedback

> Pipeline de post-entrenamiento que alinea LLMs con preferencias humanas mediante un ciclo SFT → reward model → PPO. Popularizado por OpenAI con InstructGPT ([[summaries/ouyang-2022-instructgpt]]) y ChatGPT. **El paradigma que Self-Rewarding intenta superar.**

## 1. El problema que resuelve

Un LLM preentrenado con next-token prediction adquiere conocimiento extenso pero **no está alineado**:

- Puede generar contenido tóxico, alucinaciones, o respuestas inútiles.
- Definir una función de reward explícita para "respuesta útil, honesta y segura" es prácticamente imposible.

**Insight de RLHF**: aprender la reward **implícitamente** a partir de comparaciones humanas.

## 2. El pipeline de tres etapas

### Etapa 1 — Supervised Fine-Tuning (SFT)

- **Input**: modelo preentrenado + dataset de pares `(prompt, respuesta_humana)` de alta calidad.
- **Proceso**: fine-tuning supervisado estándar con cross-entropy.
- **Output**: modelo `π_SFT` que sigue instrucciones razonablemente bien.

Este modelo actuará como `π_ref` en las etapas siguientes.

### Etapa 2 — Reward Model (RM)

- **Input**: prompts + `K` respuestas candidatas por prompt (generadas con `π_SFT`).
- **Anotación**: humanos rankean las `K` respuestas o las comparan pairwise.
- **Pérdida** (modelo Bradley-Terry):

```
L_RM = − E_{(x, y_w, y_l) ~ D} [ log σ( r_φ(x, y_w) − r_φ(x, y_l) ) ]
```

donde `r_φ(x, y)` es un LLM con cabeza escalar que predice un score numérico.

- **Output**: reward model congelado `r̂`.

### Etapa 3 — Optimización con PPO

Maximizar el reward sujeto a regularización KL contra `π_SFT`:

```
max_π  E_{x, y ~ π}[ r̂(x, y) ]  −  β · KL( π ‖ π_ref )
```

donde `β` controla cuánto puede divergir la política nueva del SFT. Demasiado bajo → reward hacking; demasiado alto → la política no mejora.

**PPO** (Proximal Policy Optimization, Schulman 2017) optimiza esto con clipping del ratio de probabilidades para evitar pasos destructivamente grandes.

## 3. Los cuatro modelos en memoria

PPO en LLMs requiere mantener simultáneamente:

| Modelo | Rol | Estado |
|---|---|---|
| Política `π_θ` | Genera tokens, se entrena | Trainable |
| Política de referencia `π_ref` | Ancla KL (= `π_SFT`) | Congelada |
| Reward model `r̂` | Asigna reward escalar | Congelado |
| Critic / Value head `V_φ` | Estima `V(s)` para advantage | Trainable |

**Total: 4 modelos**, típicamente cada uno del tamaño del LLM original. Esto explica el costo computacional de RLHF y motiva alternativas:

- **DPO** ([[concepts/dpo]]) elimina los modelos 3 y 4 → solo 2 modelos.
- **GRPO** ([[concepts/grpo]]) elimina el modelo 4 → 3 modelos.

## 4. Formulación matemática completa

### 4.1. Objetivo

```
J(π) = E_{x ~ D_prompt, y ~ π(·|x)} [ r̂(x, y) ]  −  β · E_{x} [ KL( π(·|x) ‖ π_ref(·|x) ) ]
```

### 4.2. Estimador de policy gradient con clipping (PPO)

```
L_CLIP(θ) = E_t [ min( ρ_t(θ) · A_t,  clip(ρ_t(θ), 1−ε, 1+ε) · A_t ) ]
```

donde:
- `ρ_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`: ratio de probabilidades.
- `A_t`: advantage estimada (típicamente GAE).
- `ε`: clip threshold (típicamente 0.2).

### 4.3. Loss total

```
L_PPO = L_CLIP(θ)  +  c_v · L_value(φ)  −  c_H · H(π_θ)  −  β · KL(π_θ ‖ π_ref)
```

con `c_v` peso del value loss y `c_H` coeficiente de entropía.

## 5. Limitaciones

### 5.1. Costo de anotación humana

- **Miles de comparaciones** son necesarias por iteración.
- **Variabilidad inter-anotador**: distintos humanos juzgan distinto.
- **Sesgos**: culturales, lingüísticos, preferencia por respuestas largas.

### 5.2. El cuello de botella humano

Si el LLM supera el nivel humano en una tarea, **¿quién anota?** Los humanos no pueden evaluar bien respuestas que no entienden completamente.

Esta es la motivación filosófica de Self-Rewarding ([[summaries/yuan-2024-self-rewarding]]): se necesita feedback superhumano para entrenar modelos superhumanos.

### 5.3. Reward model congelado

Una vez entrenado, `r̂` no mejora. A medida que la política mejora, puede haber **distribution shift**: las respuestas generadas se alejan de las que `r̂` aprendió a evaluar. El RM se vuelve obsoleto y la política puede explotar sus fallos (reward hacking).

### 5.4. Complejidad de PPO

- **4 modelos en memoria** (ver §3).
- **Hiperparámetros sensibles**: β, ε, c_v, c_H, learning rate, batch size.
- **Inestabilidad** en LLMs grandes: divergencia común.
- **Difícil de iterar**: cada cambio requiere re-correr la pipeline completa.

## 6. Variantes y mejoras documentadas

- **RLHF iterativo (Llama 2, Touvron 2023)**: re-anotar después de cada round para refrescar `r̂`. Mitiga el problema del RM congelado pero multiplica el costo.
- **Rejection sampling fine-tuning (RAFT, Llama 2)**: usar `r̂` para seleccionar las mejores generaciones y hacer SFT sobre ellas. Simplifica pero pierde la exploración fina de PPO.
- **GRPO (Shao 2024)**: PPO sin critic, baseline grupal. Ver [[concepts/grpo]].
- **DPO (Rafailov 2023)**: elimina RM y RL online. Ver [[concepts/dpo]].

## 7. RLHF como punto de partida narrativo

En la narrativa de la presentación, RLHF es el **punto cero** del que se aleja la evolución:

```
RLHF (humanos)
   ↓ reemplazar humanos por LLM
RLAIF (modelo externo)
   ↓ usar el mismo modelo
Self-Rewarding (juez = generador)
   ↓ añadir meta-nivel
Meta-Rewarding (juez del juez)
```

En paralelo, la rama algorítmica:

```
PPO (4 modelos)
   ↓ eliminar critic
GRPO (3 modelos)
   ↓ eliminar RM y RL online
DPO (2 modelos)
```

Ver [[synthesis/comparacion-metodos]] para la tabla canónica.

## 8. Conexión con IRL

RLHF puede verse como **IRL con preferencias** (preference-based IRL, [[summaries/wirth-2017-pbrl-survey]]):

1. El "experto" son los humanos vía preferencias.
2. El RM `r̂` es la **reward inferida**.
3. La política PPO óptima es la que maximiza esa `r̂` con regularización KL.

Esta conexión es desarrollada en [[synthesis/pilar-1-irl-implicito]] y respaldada empíricamente por [[summaries/joselowitz-2025-irl-llm]].

## 9. Hiperparámetros típicos

Basados en InstructGPT (Ouyang 2022) y Llama 2 (Touvron 2023):

| Parámetro | Valor típico |
|---|---|
| `β` (KL coeff) | 0.01 – 0.1 |
| `ε` (PPO clip) | 0.2 |
| Learning rate política | 1e-6 – 5e-6 |
| Batch size | 16 – 64 prompts |
| Rollouts por prompt | 4 – 16 |
| Épocas PPO | 4 |

## 10. Lecturas relacionadas

- [[concepts/dpo]] — simplificación que elimina RM y RL.
- [[concepts/grpo]] — simplificación que elimina critic.
- [[concepts/rlaif]] — reemplazo de humanos por LLM en la anotación.
- [[concepts/self-rewarding-loop]] — caso límite: el modelo se anota a sí mismo.
- [[summaries/ouyang-2022-instructgpt]] — paper canónico de RLHF a escala.
- [[summaries/christiano-2017-deep-rl-preferences]] — primera aplicación de preferencias humanas en deep RL.
- [[summaries/schulman-2017-ppo]] — algoritmo PPO original.
- [[summaries/acosta-2026-deep-research]] §2 — síntesis original en el deep research.
