---
name: grpo
description: Group Relative Policy Optimization — variante on-policy de PPO que elimina el critic estimando el baseline a partir de un grupo de respuestas al mismo prompt.
metadata:
  type: concept
  tags: [grpo, ppo, on-policy, deepseek, rl-algorithms, critic-free]
  last_updated: 2026-05-12
---

# GRPO — Group Relative Policy Optimization

> Algoritmo de RL on-policy que mantiene la exploración de PPO pero **elimina el critic** estimando el baseline a partir de un grupo de respuestas al mismo prompt. Propuesto por Shao et al. (2024, *DeepSeekMath*); popularizado por DeepSeek-R1.

## 1. El problema que resuelve

PPO en LLMs requiere mantener **4 modelos en memoria**:

| Modelo | Rol |
|---|---|
| Política actual `π_θ` | Se entrena |
| Política de referencia `π_ref` | Ancla KL |
| Reward model `r̂` | Asigna recompensa |
| Critic / Value head `V_φ` | Estima `V(s)` para calcular advantage |

El critic, en particular, es otra red del tamaño de la política. Duplica el costo de memoria. GRPO se pregunta: **¿se puede prescindir del critic?**

## 2. Idea central

En lugar de estimar `V(s)` con una red separada, GRPO **muestrea `G` respuestas** al mismo prompt y usa su distribución de rewards como baseline:

```
Para cada prompt x:
  1. Muestrear G respuestas {y_1, ..., y_G} de π_θ_old.
  2. Calcular reward r_i = r(x, y_i) para cada una.
  3. Normalizar dentro del grupo:
       A_i = (r_i − mean(r_1..G)) / std(r_1..G)
  4. Usar A_i como advantage en el objetivo PPO clippado.
```

**El grupo es el baseline.** Sin red de value adicional.

## 3. Objetivo de optimización

GRPO mantiene la pérdida tipo PPO con clipping del ratio, pero reemplaza la advantage GAE por la advantage grupal:

```
L_GRPO(θ) = E_{x, {y_i}~π_θ_old} [
    (1/G) Σ_i  min(
        ρ_i(θ) · A_i,
        clip(ρ_i(θ), 1−ε, 1+ε) · A_i
    )
]  −  β · KL( π_θ ‖ π_ref )
```

donde:
- `ρ_i(θ) = π_θ(y_i|x) / π_θ_old(y_i|x)`: ratio de probabilidades.
- `A_i`: advantage normalizada dentro del grupo (paso 3 arriba).
- `ε`: parámetro de clipping (igual que PPO).
- `β · KL`: regularización contra `π_ref`.

## 4. Función de reward

GRPO requiere una **función de reward computable directamente**, no un reward model entrenado. Esto la hace particularmente apta para dominios verificables:

- **Matemáticas**: `r = 1` si la respuesta numérica final es correcta, `r = 0` si no.
- **Código**: `r = 1` si el código pasa los tests, `r = 0` si no.
- **Formato**: penalizaciones por no seguir formato `<think>...</think><answer>...</answer>`.

Esto contrasta con RLHF, donde `r` es un LLM entrenado sobre preferencias humanas, costoso y susceptible a hacking.

## 5. Comparación con PPO y DPO

| Aspecto | PPO | GRPO | DPO |
|---|---|---|---|
| Tipo | On-policy | On-policy | Offline |
| Critic | Sí (red separada) | **No** (baseline grupal) | No |
| Reward model | Sí (entrenado) | No — función verificable | No — implícito |
| Modelos en memoria | 4 | 3 (π_θ, π_ref, función reward) | 2 |
| Exploración | Sí | Sí | No |
| Apto para razonamiento | Sí | **Sí, óptimo** | Limitado |
| Costo computacional | Alto | Medio | Bajo |

## 6. Comportamientos emergentes

Durante el entrenamiento de DeepSeek-R1-Zero (RL puro con GRPO, sin SFT), emergieron comportamientos no programados (ver [[entities/deepseek-r1]]):

- **Auto-verificación**: el modelo verifica sus propias respuestas antes de comprometerse.
- **Reflexión**: identifica errores en su razonamiento y los corrige mid-stream.
- **Razonamiento de cadena larga**: el modelo aprende a usar miles de tokens internos antes de emitir respuesta final.
- **"Aha moment"**: el modelo descubre estrategias de razonamiento más efectivas a mitad del entrenamiento, evidenciado en saltos discretos de performance.

Estos son análogos al **self-play** en juegos: el modelo descubre estrategias que ningún humano le enseñó.

## 7. Limitaciones

1. **Requiere reward verificable.** No aplica a tareas subjetivas (instruction-following abierto, escritura creativa). Aquí Self-Rewarding y DPO sobre preferencias siguen siendo necesarios.
2. **Costo de muestreo.** Necesita `G` rollouts por prompt (típicamente `G = 16` o `64`). Más caro que DPO offline.
3. **Estimación ruidosa con `G` pequeño.** Si `G` es bajo, la advantage normalizada tiene alta varianza.
4. **Reward hacking en RL prolongado.** Mismo problema que PPO en LLMs — el modelo puede aprender a explotar la función reward sin cumplir el objetivo real.

## 8. Relación con Self-Rewarding

GRPO y Self-Rewarding comparten **filosofía de auto-mejora** pero difieren en mecanismo (ver [[summaries/acosta-2026-deep-research]] §7.3):

| Self-Rewarding | GRPO / R1-Zero |
|---|---|
| Juez = LLM (rúbrica subjetiva) | Reward = función verificable |
| Tareas abiertas | Tareas con respuesta correcta |
| DPO sobre pares offline | RL on-policy con clipping |
| Loop de iteraciones del modelo `M_t` | Loop de rollouts del mismo modelo |

**Son complementarios**, no competidores. Cubren regímenes distintos del espacio de tareas LLM.

## 9. Fuentes y referencias

- **Paper original**: Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv (entrada en [[sources]]).
- **Aplicación masiva**: DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948. Ver [[entities/deepseek-r1]].
- **Síntesis interna**: [[summaries/acosta-2026-deep-research]] §3.5 y §7.

## 10. Lecturas relacionadas

- [[concepts/rlhf]] — PPO clásico que GRPO simplifica.
- [[concepts/dpo]] — alternativa offline con compromisos distintos.
- [[entities/deepseek-r1]] — caso de estudio principal.
- [[synthesis/comparacion-metodos]] — tabla canónica donde GRPO aparece.
