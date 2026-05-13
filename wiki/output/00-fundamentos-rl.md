# 00 — Fundamentos de Reinforcement Learning para alguien que viene de NLP

> **Para qué este capítulo**: como experto en NLP ya tienes intuiciones sobre transformers, sampling y MLE. Te falta el vocabulario y la maquinaria de RL para entender por qué el "RL" de RLHF/DPO es lo que es. Aquí construimos eso desde cero, pero conectándolo siempre con cosas que ya conoces.

> **Lo que sabrás al final**: qué es un MDP, qué es una policy, qué significa optimizar una policy, qué es policy gradient, qué hace PPO, qué es la divergencia KL y por qué aparece en todos lados.

---

## 1. El problema que RL resuelve (que SFT no resuelve)

En NLP estándar entrenas con **MLE** (maximum likelihood estimation): tienes un dataset de pares `(input, output)` y maximizas `log P(output | input)`. Tu modelo aprende a imitar al dataset.

Esto funciona cuando:
- Tienes el output correcto para cada input.
- Quieres que el modelo lo reproduzca.

Falla cuando:
- **No tienes el output correcto**, solo sabes que "respuesta A es mejor que B".
- **Quieres optimizar una métrica** que no es una distribución (e.g., "respuesta útil", "respuesta no tóxica", "respuesta corta").
- **Hay múltiples outputs válidos** y no quieres comprometerte con uno.

RL ataca exactamente este escenario: aprender comportamiento a partir de **una señal escalar de calidad** (reward), no de demostraciones supervisadas.

> 💡 **Intuición clave**: SFT te dice "imita esto". RL te dice "maximiza esta puntuación". La puntuación puede venir de cualquier fuente — un simulador, un humano, otro modelo, una calculadora.

---

## 2. Markov Decision Process (MDP) — el lenguaje de RL

Un MDP es una abstracción que captura "agente actuando en un entorno". Cinco componentes:

| Símbolo | Nombre | Para LLMs significa... |
|---|---|---|
| `S` | Estado | Contexto actual (prompt + tokens generados hasta ahora). |
| `A` | Acción | El siguiente token a generar. |
| `P(s' \| s, a)` | Transición | Determinista: concatenar el token al contexto. |
| `R(s, a)` | Reward | Puntuación de la respuesta (típicamente al final). |
| `γ` | Discount factor | Cuánto valen los rewards futuros vs presentes. En LLMs típicamente `γ = 1`. |

### El MDP de un LLM

```
Estado inicial s_0 = prompt
Acción a_0 = primer token generado → s_1 = [prompt, token_0]
Acción a_1 = segundo token → s_2 = [prompt, token_0, token_1]
...
Estado terminal s_T = secuencia completa, recibe reward R(s_T).
```

> 💡 **Observación**: el MDP de un LLM es **un caso degenerado** del MDP general:
> - Transiciones determinísticas (no hay aleatoriedad del entorno).
> - Reward solo al final (no por token).
> - Espacio de acciones enorme pero discreto (todo el vocabulario).

Esto es importante: muchas complejidades del RL clásico no aparecen en LLMs.

### Policy `π`

Una **policy** `π(a | s)` es una distribución sobre acciones dada el estado. En LLMs:

```
π(a | s) = softmax(LLM(s))_a
```

**La policy es el LLM**. Generar texto = samplear de `π`. Entrenar el LLM = cambiar `π`.

> 📐 **Fórmula**: probabilidad de generar una secuencia completa `y = (a_0, a_1, ..., a_{T-1})` dado prompt `x`:
> ```
> π(y | x) = Π_{t=0}^{T-1} π(a_t | x, a_0, ..., a_{t-1})
>          = Π_t π(a_t | s_t)
> ```
> Esto es exactamente la "likelihood de la secuencia" que ya conoces de NLP.

### Return `G`

El **return** es la suma de rewards de una trayectoria:

```
G(τ) = Σ_{t=0}^{T-1} γ^t · R(s_t, a_t)
```

donde `τ = (s_0, a_0, s_1, a_1, ..., s_T)` es la trayectoria completa.

En LLMs con reward solo al final y `γ = 1`:

```
G(τ) = R(s_T) = R(prompt, respuesta_completa)
```

### Objetivo

El objetivo de RL es encontrar la policy `π*` que maximice el return esperado:

```
π* = argmax_π  E_{τ ~ π}[ G(τ) ]
```

> 💡 **Léelo así**: "encuentra la policy tal que, si muestreo trayectorias siguiéndola, en promedio obtengo el mayor return posible."

---

## 3. Value functions — la herramienta para razonar sobre futuros

Aunque no las usaremos directamente en DPO, las value functions son cruciales para PPO (y por ende para RLHF clásico). Tres conceptos:

### State value `V^π(s)`

"Si estoy en estado `s` y sigo la policy `π`, ¿qué return puedo esperar?"

```
V^π(s) = E_{τ ~ π, s_0=s}[ G(τ) ]
```

### Action value `Q^π(s, a)`

"Si estoy en estado `s`, ejecuto acción `a`, y luego sigo `π`, ¿qué return puedo esperar?"

```
Q^π(s, a) = E[ R(s,a) + γ · V^π(s') ]
```

### Advantage `A^π(s, a)`

"¿Cuánto mejor es la acción `a` que la 'acción promedio' que tomaría `π` en `s`?"

```
A^π(s, a) = Q^π(s, a) − V^π(s)
```

> 💡 **Intuición**: si `A^π(s, a) > 0`, la acción `a` es mejor que el promedio de lo que haría `π`. Si `A^π(s, a) < 0`, es peor. El **advantage** es la cantidad que vamos a empujar hacia arriba (o abajo) al entrenar.

### Ecuación de Bellman (mención breve)

```
V^π(s) = E_{a ~ π}[ R(s, a) + γ · V^π(s') ]
```

"El valor de un estado es el reward inmediato más el valor del siguiente estado." Es la base recursiva de muchos algoritmos RL clásicos (Q-learning, etc.). En LLMs raramente la usamos directamente, pero PPO la usa para estimar el critic.

---

## 4. Policy gradient — cómo entrenar una policy

El truco fundamental: queremos **maximizar** `J(π) = E_τ[G(τ)]`. Si `π` está parametrizada por `θ` (los pesos del LLM), queremos calcular `∇_θ J`.

### El teorema de policy gradient

> 📐 **Fórmula central de RL**:
> ```
> ∇_θ J(π_θ) = E_{τ ~ π_θ}[ Σ_t ∇_θ log π_θ(a_t | s_t) · G(τ) ]
> ```
>
> O en forma de "ganador per-paso" con advantage:
> ```
> ∇_θ J(π_θ) = E[ Σ_t ∇_θ log π_θ(a_t | s_t) · A^{π_θ}(s_t, a_t) ]
> ```

### Cómo leer esta fórmula

`∇_θ log π_θ(a | s)` es **el gradiente de la log-likelihood** de la acción `a` en estado `s`. Esto es **lo mismo** que usas en MLE — solo que aquí lo **ponderas** por el return (o el advantage).

> 💡 **Tres líneas de intuición**:
> 1. Si tomé acción `a` y resultó bien (`A > 0`), entonces empuja los pesos para hacer `a` **más probable** en estado `s`.
> 2. Si tomé acción `a` y resultó mal (`A < 0`), empuja para hacerla **menos probable**.
> 3. La magnitud del empujón es proporcional a cuán bien/mal resultó.

### Algoritmo "naïve" (REINFORCE)

```
1. Inicializar política π_θ (e.g., un LLM SFT).
2. Repetir:
   a. Generar trayectoria τ con π_θ (samplear tokens).
   b. Calcular G(τ) (evaluar el reward al final).
   c. Calcular ∇_θ log π_θ(τ) · G(τ).
   d. Actualizar θ ← θ + α · gradiente.
```

Esto funciona pero **tiene problemas**:
- **Alta varianza**: G(τ) puede variar mucho de trayectoria a trayectoria.
- **No reusa muestras**: cada trayectoria se descarta tras un solo update.
- **Pasos destructivos**: con learning rate mal calibrado, una mala muestra puede arruinar la política.

PPO y otros algoritmos modernos resuelven estos problemas.

### On-policy vs off-policy

Concepto **crítico**:

- **On-policy**: las muestras usadas para entrenar `π_θ` vienen de `π_θ` misma (la política actual).
- **Off-policy**: las muestras vienen de otra política (e.g., una versión anterior, o una política de "exploración" distinta).

> 💡 **Por qué importa para LLMs**:
> - REINFORCE/PPO son **on-policy**: necesitas regenerar respuestas cada vez que actualizas el modelo. Caro.
> - DPO es **off-policy** (más bien "supervisado offline"): los pares de preferencias se generan una vez y se reusan. Mucho más barato.

---

## 5. KL divergence — la regularización omnipresente

> 📐 **Definición**: la **KL divergence** entre dos distribuciones `P` y `Q` mide cuán diferentes son:
> ```
> KL(P || Q) = Σ_x P(x) · log( P(x) / Q(x) )
> ```
>
> Propiedades:
> - `KL(P || Q) ≥ 0`, igualdad sólo si `P = Q`.
> - No es simétrica: `KL(P || Q) ≠ KL(Q || P)`.
> - No es una distancia formal, pero se comporta como una "distancia direccional".

### Por qué aparece en RLHF/DPO

Cuando entrenas un LLM con RL, el modelo puede **derivar muy lejos** de su versión original. Esto es malo porque:
1. Pierde fluidez/coherencia adquiridas en pretraining + SFT.
2. Puede "hackear el reward" — encontrar trucos que dan reward alto pero respuestas inútiles (e.g., spam de palabras clave).

Para evitarlo, se añade un **término de regularización KL** que penaliza divergir de una **política de referencia** `π_ref` (típicamente el modelo SFT pre-RL):

```
J_KL-regularizado(π) = E_τ[G(τ)]  −  β · KL( π || π_ref )
```

donde `β` controla la fuerza de la regularización.

> 💡 **Lectura intuitiva**: "Maximiza el reward, pero **no te alejes mucho** del modelo SFT."
>
> - `β → 0`: ignora KL, optimiza puramente reward. Riesgo de reward hacking.
> - `β → ∞`: KL domina, política queda pegada a SFT. No mejora.
> - `β = 0.1` es el valor estándar usado en RLHF y DPO.

### Una observación importante

> ⚠️ **No confundir** la KL del objetivo con la KL del clipping de PPO. Son cosas distintas.
> - **KL del objetivo** (la que estás viendo): penaliza divergir de `π_ref` durante todo el entrenamiento.
> - **Clipping de PPO**: previene que un paso individual de gradiente cambie `π` demasiado bruscamente.

---

## 6. PPO — el algoritmo que RLHF usa

Proximal Policy Optimization (Schulman 2017) es el algoritmo de policy gradient más popular para LLMs.

### Idea 1 — Ratio de importancia

PPO permite reusar samples generados con `π_θ_old` (versión "vieja" de la política) para entrenar la versión nueva `π_θ`. Usa el **ratio de probabilidades**:

```
ρ_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
```

Si `ρ_t > 1`, la nueva política hace `a_t` más probable. Si `ρ_t < 1`, menos probable.

### Idea 2 — Clipping

Para evitar pasos destructivos, PPO **trunca** el ratio:

> 📐 **Loss de PPO** (sin todos los detalles):
> ```
> L_CLIP(θ) = E[ min( ρ_t(θ) · A_t,  clip(ρ_t(θ), 1−ε, 1+ε) · A_t ) ]
> ```
>
> Lectura:
> - Calcula el "objetivo" como `ρ · A` (similar a policy gradient).
> - Pero también calcula la versión "clipped" donde `ρ` está limitado a `[1−ε, 1+ε]`.
> - Toma el **mínimo** de los dos → previene actualizaciones demasiado optimistas.

Con `ε = 0.2`, el ratio puede como máximo cambiar la probabilidad de una acción en un 20% por paso.

### Idea 3 — Critic (value head)

Para estimar el advantage `A_t`, PPO entrena una **value network** `V_φ(s)` en paralelo a la política. Esta red toma estados y predice el value `V^π(s)`. El advantage se estima con GAE (Generalized Advantage Estimation), un truco para reducir varianza.

> ⚠️ **Esto es lo que cuesta**: en LLMs, el critic suele ser **otro LLM** del mismo tamaño que la política. Por eso PPO requiere 4 modelos en memoria (política, política de referencia, reward model, critic).

### Algoritmo simplificado

```
1. Inicializar π_θ, V_φ, π_ref.
2. Repetir:
   a. Generar N respuestas con π_θ_old.
   b. Calcular rewards con reward model r̂.
   c. Calcular advantages A_t con V_φ (usando GAE).
   d. Para K épocas sobre los N samples:
      - Update θ minimizando L_CLIP + β·KL(π_θ || π_ref).
      - Update φ minimizando MSE de V_φ vs return observado.
   e. π_θ_old ← π_θ.
```

### Por qué PPO es "casi un default" en RL para LLMs

- Estable (con suficiente cuidado).
- Reusa samples (más eficiente que REINFORCE).
- Hiperparámetros bien estudiados.
- Implementaciones maduras (TRL, OpenRLHF).

### Por qué hay alternativas (DPO, GRPO)

- 4 modelos en memoria es caro.
- Sensible a hiperparámetros.
- Inestable en escalas grandes.
- No iterativo de forma natural.

---

## 7. Off-policy y "supervised RL"

Una clase de métodos modernos transforma el problema de RL en algo que se parece a **supervised learning**.

### Idea

Si tienes un dataset `D = {(x, y_w, y_l)}` (prompts con respuesta preferida y no preferida), puedes derivar una pérdida supervisada que **funciona como si hubieras hecho RL**. Esa es la idea detrás de **DPO** (que veremos en detalle en el capítulo 02).

> 💡 **Diferencia de mentalidad**:
>
> | RL on-policy (PPO) | RL offline (DPO) |
> |---|---|
> | Generar muestras durante entrenamiento | Dataset fijo |
> | "Política × reward × política" | "Una pérdida supervisada que ya incorpora todo" |
> | Iterativo por naturaleza | Una sola pasada (aunque puedes iterar manualmente) |
> | Cuatro modelos | Dos modelos |
> | Inestable | Estable |
> | Caro | Barato |

---

## 8. Glosario rápido

| Término | Significado en una línea |
|---|---|
| Policy `π` | Distribución sobre acciones. Para LLMs: el propio modelo. |
| Reward `r(s,a)` | Puntuación escalar de una acción en un estado. |
| Return `G(τ)` | Suma de rewards de una trayectoria. |
| Value `V^π(s)` | Return esperado desde `s` siguiendo `π`. |
| Advantage `A^π(s,a)` | Cuán mejor es `a` que el promedio en `s`. |
| Trajectory `τ` | Secuencia de estados y acciones. Para LLMs: una respuesta completa. |
| On-policy | Samples vienen de la política actual. |
| Off-policy | Samples vienen de otra distribución. |
| KL divergence | Diferencia direccional entre dos distribuciones. |
| Policy gradient | `∇_θ E[G] = E[∇_θ log π · A]`. La fórmula maestra. |
| PPO | Policy gradient + clipping del ratio + critic. |
| Reward model | Red entrenada que aproxima `r(s,a)`. En RLHF, aprende de preferencias humanas. |
| Reference policy `π_ref` | Política congelada que ancla la nueva política (típicamente SFT). |

---

## 9. Conexiones con NLP que ya conoces

Para fijar las cosas:

| En NLP / SFT | En RL para LLMs |
|---|---|
| Likelihood `P(y \| x)` | Policy `π(y \| x)` |
| Loss MLE = `−log P(y \| x)` | Reward = `r(x, y)` (señal escalar arbitraria) |
| Backprop estándar | Policy gradient + KL regularization |
| Dataset `(x, y)` | Dataset `(x, y_w, y_l)` o reward model |
| Una sola pasada | Iteraciones de generación + update |
| Sampling con temperature | Sampling **es** ejecutar la policy |

> 💡 **Frase para fijar**: "En RL para LLMs, el modelo es la policy, generar texto es ejecutar la policy, y entrenar el modelo es policy gradient o una variante."

---

## 10. Qué viene en el siguiente capítulo

Con esto ya tienes los bloques para entender RLHF:

- **MLE** te dio el modelo base (pretraining).
- **SFT** te dio el modelo de instrucciones (fine-tuning supervisado).
- **RLHF** te va a dar el modelo alineado a preferencias humanas usando policy gradient + KL regularization + reward model.

Continúa con [01-rlhf.md](01-rlhf.md).

---

## Lecturas relacionadas en el wiki

- [[concepts/rlhf]] — versión densa de RLHF.
- [[concepts/dpo]] — derivación matemática completa de DPO.
- [[summaries/schulman-2017-ppo]] — paper original de PPO.
- [[summaries/christiano-2017-deep-rl-preferences]] — primera aplicación de preferencias humanas en RL profundo.
