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

### Ecuación de Bellman

> 📐 **Ecuación de Bellman para `V^π`**:
> ```
> V^π(s) = E_{a ~ π(·|s)}[ R(s, a) + γ · E_{s' ~ P(·|s,a)}[ V^π(s') ] ]
> ```
>
> En forma compacta: `V^π(s) = r̄(s) + γ · V^π(s')` (en expectación).

**Cómo leerla**: el valor de un estado es el reward inmediato (en expectación bajo la política) más el valor del siguiente estado (descontado por `γ`).

> 💡 **¿Por qué es útil?** Es **recursiva**: relaciona `V(s)` con `V(s')`. Esto permite:
> 1. **Algoritmos iterativos** (value iteration, policy iteration en RL clásico).
> 2. **Estimación con bootstrap**: `V_φ(s_t)` se entrena con `r_t + γ V_φ(s_{t+1})` como target (TD learning).
> 3. **TD error**: `δ_t = r_t + γ V(s_{t+1}) − V(s_t)`. Si `δ ≠ 0`, hay un error a corregir.

### Variantes de Bellman

| Versión | Para | Fórmula |
|---|---|---|
| `V^π` | Política fija | `V^π(s) = E_a[R(s,a) + γE[V^π(s')]]` |
| `Q^π` | Política fija, acción dada | `Q^π(s,a) = R(s,a) + γE_{s',a'}[Q^π(s',a')]` |
| `V*` (óptimo) | Política óptima | `V*(s) = max_a [R(s,a) + γE[V*(s')]]` |
| `Q*` (óptimo) | Política óptima | `Q*(s,a) = R(s,a) + γE_{s'}[max_{a'} Q*(s',a')]` |

> ⚠️ **En LLMs raramente las usamos en su forma clásica** porque:
> - El espacio de estados es enorme (todos los contextos posibles).
> - Bellman iteration es para MDPs tabulares.
> - PPO **sí usa Bellman implícitamente** para entrenar el critic (TD learning con bootstrap).

---

## 4. Policy gradient — cómo entrenar una policy

El truco fundamental: queremos **maximizar** `J(π) = E_τ[G(τ)]`. Si `π` está parametrizada por `θ` (los pesos del LLM), queremos calcular `∇_θ J`.

### 4.1. El problema antes del truco

```
J(θ) = E_{τ ~ π_θ}[ G(τ) ] = Σ_τ P(τ; θ) · G(τ)
```

donde `P(τ; θ) = P(s_0) · Π_t π_θ(a_t|s_t) · P(s_{t+1}|s_t, a_t)`.

**Problema**: para calcular `∇_θ J` querríamos:
```
∇_θ J = Σ_τ ∇_θ P(τ; θ) · G(τ)
```

Pero `∇_θ P(τ; θ)` es difícil — la trayectoria entera depende de `θ` a través de muchos pasos. No podemos muestrear porque `∇P` no es una distribución.

### 4.2. El log-derivative trick

> 📐 **Identidad fundamental** (vale para cualquier `P(x; θ)`):
> ```
> ∇_θ P(x; θ) = P(x; θ) · ∇_θ log P(x; θ)
> ```
>
> **Demostración**: por la regla de la cadena, `∇ log P = ∇P / P`, así que `P · ∇ log P = ∇P`. ∎

Aplicado a trayectorias:
```
∇_θ P(τ; θ) = P(τ; θ) · ∇_θ log P(τ; θ)
```

Sustituyendo:
```
∇_θ J = Σ_τ P(τ; θ) · ∇_θ log P(τ; θ) · G(τ)
      = E_{τ ~ π_θ}[ ∇_θ log P(τ; θ) · G(τ) ]
```

**¡Ahora sí es una expectación!** Podemos estimarla muestreando trayectorias.

### 4.3. Simplificación: ∇log P(τ)

```
log P(τ; θ) = log P(s_0) + Σ_t [ log π_θ(a_t|s_t) + log P(s_{t+1}|s_t,a_t) ]
```

Solo `log π_θ` depende de `θ`. Los demás términos (transiciones del entorno, estado inicial) tienen gradiente cero:
```
∇_θ log P(τ; θ) = Σ_t ∇_θ log π_θ(a_t | s_t)
```

### 4.4. Resultado: el policy gradient theorem

> 📐 **Fórmula central de RL** (Williams 1992, "REINFORCE"):
> ```
> ∇_θ J(π_θ) = E_{τ ~ π_θ}[ Σ_t ∇_θ log π_θ(a_t | s_t) · G(τ) ]
> ```
>
> O en forma de "ganador per-paso" con advantage:
> ```
> ∇_θ J(π_θ) = E[ Σ_t ∇_θ log π_θ(a_t | s_t) · A^{π_θ}(s_t, a_t) ]
> ```

### 4.5. Cómo leer esta fórmula

`∇_θ log π_θ(a | s)` es **el gradiente de la log-likelihood** de la acción `a` en estado `s`. Esto es **lo mismo** que usas en MLE — solo que aquí lo **ponderas** por el return (o el advantage).

> 💡 **Tres líneas de intuición**:
> 1. Si tomé acción `a` y resultó bien (`A > 0`), entonces empuja los pesos para hacer `a` **más probable** en estado `s`.
> 2. Si tomé acción `a` y resultó mal (`A < 0`), empuja para hacerla **menos probable**.
> 3. La magnitud del empujón es proporcional a cuán bien/mal resultó.

> 📊 **Comparación visual con MLE**:
>
> ```
>   MLE (supervisado):
>     ∇_θ J_MLE = E_{(x,y) ~ D}[ ∇_θ log p_θ(y|x) ]
>                  ↑ todos los y observados son "buenos" (peso 1)
>
>   Policy Gradient:
>     ∇_θ J_PG  = E_{τ ~ π_θ}[ Σ_t ∇_θ log π_θ(a_t|s_t) · A_t ]
>                  ↑                                       ↑
>                  mismo gradiente             ponderado por "cuán bueno"
>
>   Cuando A_t = 1 para todos:  Policy gradient = MLE.
>   Cuando A_t es variable:    Policy gradient PESA cada paso según calidad.
> ```

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

Proximal Policy Optimization (Schulman 2017) es el algoritmo de policy gradient más popular para LLMs. Está construido sobre tres ideas: ratio de importancia, clipping, y critic.

### 6.1. El problema con REINFORCE/vanilla policy gradient

El policy gradient básico tiene tres problemas:

1. **Alta varianza**: `G(τ)` varía mucho entre trayectorias. El gradiente es ruidoso.
2. **Una sola pasada por sample**: si tomas un paso de gradiente y los pesos cambian, los samples ya no son "tuyos" (vienen de la política vieja). Hay que regenerar → caro.
3. **Pasos destructivos**: un solo update grande puede hacer que la política colapse.

PPO ataca los tres.

### 6.2. Idea 1 — Importance sampling: reusar samples viejos

Queremos calcular `E_{τ ~ π_θ}[...]` pero solo tenemos samples de `π_θ_old`. **Importance sampling** nos rescata:

```
E_{x ~ p}[f(x)] = E_{x ~ q}[ (p(x)/q(x)) · f(x) ]
```

Aplicado al gradiente:
```
∇_θ J = E_{τ ~ π_θ_old}[ Σ_t (π_θ(a_t|s_t) / π_θ_old(a_t|s_t)) · ∇_θ log π_θ · A_t ]
                                                                              ↑
                                                  esto se simplifica usando log-derivative trick
```

Definiendo el **ratio**:
```
ρ_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
```

el objetivo a maximizar (versión "surrogate") es:
```
L^{CPI}(θ) = E_t[ ρ_t(θ) · A_t ]
```

(CPI = Conservative Policy Iteration; la versión "naive" antes del clipping).

> 💡 **Lectura intuitiva**:
> - `ρ_t > 1`: la nueva política hace `a_t` **más probable** que la vieja.
> - `ρ_t < 1`: menos probable.
> - Multiplicado por `A_t`: si la acción fue buena (`A > 0`), quieres `ρ > 1`. Si fue mala, quieres `ρ < 1`.

### 6.3. Idea 2 — Por qué necesitamos clipping

**Problema**: si `A_t > 0` (acción buena), el objetivo `L^CPI = ρ · A` crece linealmente con `ρ`. Un solo paso de gradiente puede llevar `ρ → ∞` (la política da probabilidad 1 a esa acción). **Pero importance sampling solo es válida cuando `π_θ ≈ π_θ_old`** — si te alejas mucho, la estimación es ruidosa y catastrófica.

> 📊 **Por qué `ρ` no debe crecer sin cota**:
>
> ```
>   Si tomamos solo un sample con A=+5 y ρ se va a 100:
>     - El "gradiente" sugiere actualizar 100·5 = 500.
>     - Pero π_θ ya no se parece a π_θ_old.
>     - El advantage A=+5 fue medido con π_θ_old.
>     - El "verdadero" A bajo π_θ podría ser negativo.
>     - El update destruye la política.
> ```

**Solución de PPO**: **truncar el ratio** a un rango pequeño `[1−ε, 1+ε]` (típicamente `ε = 0.2`).

> 📐 **Pérdida de PPO (CLIP version)**:
> ```
> L^{CLIP}(θ) = E_t[ min( ρ_t(θ) · A_t,  clip(ρ_t(θ), 1−ε, 1+ε) · A_t ) ]
> ```

### 6.4. Por qué `min` y `clip` juntos — análisis caso por caso

Hay cuatro escenarios:

> 📊 **Tabla de comportamiento de `L^CLIP`**:
>
> ```
>   ╔══════════════════════╤══════════════════╤══════════════════════╗
>   ║       A_t > 0        │     A_t < 0       │     Objetivo:      ║
>   ║     (buena acción)   │   (mala acción)   │                     ║
>   ╠══════════════════════╪═══════════════════╪═════════════════════╣
>   ║                      │                   │                     ║
>   ║   ρ < 1−ε:          │  ρ < 1−ε:        │ NO penalizamos      ║
>   ║   "ya bajaste mucho  │   "bien, ya       │ a alejarte si va   ║
>   ║   una acción buena   │   redujiste"      │ en la dirección    ║
>   ║   - retrocede"       │                   │ correcta           ║
>   ║                      │                   │                     ║
>   ║   1−ε ≤ ρ ≤ 1+ε:    │  1−ε ≤ ρ ≤ 1+ε:  │ Aprende normal     ║
>   ║   normal             │  normal           │                     ║
>   ║                      │                   │                     ║
>   ║   ρ > 1+ε:          │  ρ > 1+ε:        │ Si vas en dirección║
>   ║   gradient = 0       │   "subiste una   │ MALA, te dejamos   ║
>   ║   (no aprendas más,  │   acción mala -   │ retroceder.        ║
>   ║   ya subiste mucho)  │   retrocede!"     │ Si vas en BUENA y   ║
>   ║                      │                   │ ya subiste mucho,  ║
>   ║                      │                   │ pausa.              ║
>   ╚══════════════════════╧═══════════════════╧═════════════════════╝
> ```

> 💡 **Lectura corta**: PPO **siempre permite corregir errores** (retroceder de acciones malas) pero **detiene la optimización** cuando ya has subido mucho una acción buena. Esto previene update destructivos sin sacrificar la capacidad de corrección.

### 6.5. Idea 3 — Critic (value head) y GAE

Para estimar el advantage `A_t` necesitamos `V^π(s_t)`. PPO entrena una **red separada** (el critic) `V_φ(s)`.

#### GAE — Generalized Advantage Estimation

El advantage "real" sería:
```
A_t = G_t − V^π(s_t)  (donde G_t es el return desde t)
```

Pero `G_t` es ruidoso (suma de muchos rewards). Una alternativa con menos varianza pero más sesgo es:
```
A_t = r_t + γ · V(s_{t+1}) − V(s_t)   (1-step TD error)
```

GAE **interpola** entre los dos usando un parámetro `λ ∈ [0, 1]`:

```
A_t^GAE = Σ_{k=0}^{∞} (γλ)^k · δ_{t+k}
```

donde `δ_t = r_t + γ V(s_{t+1}) − V(s_t)` es el **TD error**.

- `λ = 0`: solo 1-step TD (bajo varianza, alto sesgo).
- `λ = 1`: todos los pasos (alta varianza, bajo sesgo) — equivalente a Monte Carlo.
- `λ = 0.95` (típico): balance.

> 📊 **GAE en una imagen**:
>
> ```
>            λ = 0                    λ = 1
>         (alto sesgo)          (alta varianza)
>              │                       │
>              │                       │
>              └───────────────────────┘
>                          │
>                       λ = 0.95
>                       (balance)
>                          │
>                          ↓
>                  A^GAE_t para PPO
> ```

> ⚠️ **Esto es lo que cuesta**: en LLMs, el critic suele ser **otro LLM** del mismo tamaño que la política. Por eso PPO requiere 4 modelos en memoria (política, política de referencia, reward model, critic).

### 6.6. Loss total de PPO

> 📐 **Loss total**:
> ```
> L_PPO = L^CLIP(θ) − c_v · L^V(φ) + c_H · H(π_θ)
> ```
>
> donde:
> - `L^CLIP`: política clippeada.
> - `L^V = MSE( V_φ(s_t), G_t )`: entrenar el critic.
> - `c_H · H(π_θ)`: bono por entropía para fomentar exploración.

Para RLHF, **además** se añade el término KL:
```
L_RLHF-PPO = L_PPO − β · KL(π_θ || π_ref)
```

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
