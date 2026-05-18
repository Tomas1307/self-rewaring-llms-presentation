# Mapping LLM ↔ RL — explicación con dudas resueltas

> **Para qué este archivo**: cuando vienes de NLP y empiezas a leer RL aplicado
> a LLMs, hay un grupo de confusiones muy específicas que aparecen casi siempre.
> Este documento explica cada concepto en orden y resuelve, en el punto donde
> aplica, las preguntas concretas que surgieron en la conversación.

> **Convención**: los bloques `💬 Tu preguntaste: "..."` recogen preguntas
> literales que aparecieron durante la conversación didáctica original. Justo
> debajo va la respuesta puntual a esa pregunta. El resto del texto es la
> explicación lineal del tema.

---

## 1. El mapping LLM ↔ RL (lo más importante)

Cuando aplicamos RL a un LLM, traducimos los conceptos así:

| Concepto en RL | En un LLM significa... |
|---|---|
| **Estado `s`** | El contexto actual: `prompt + tokens generados hasta ahora` |
| **Acción `a`** | El **próximo token** a generar |
| **Espacio de acciones** | El **vocabulario** completo (~50.000 tokens) |
| **Política `π(a\|s)`** | La distribución sobre el vocabulario que el LLM produce |
| **Reward `r`** | Una puntuación, **típicamente al final** (sobre la respuesta completa) |
| **Trayectoria `τ`** | La generación entera: prompt → token_0 → token_1 → ... → respuesta completa |

### Visual del MDP de un LLM

```
Estado s_0:  "Explícame qué es un transformer."     ← prompt inicial
Acción a_0:  "Un"                                    ← token elegido
Estado s_1:  "Explícame qué es un transformer. Un"  ← contexto crecido
Acción a_1:  "transformer"
Estado s_2:  "Explícame qué es un transformer. Un transformer"
...
Estado s_T:  "Explícame... [respuesta completa]"
Reward:      r(s_T) = 3.5   ← solo al final
```

Cada generación de un token = un paso en el MDP. La generación completa = una trayectoria.

> 💡 **Detalle simplificador en LLMs**: en RL clásico hay rewards en cada paso. En LLMs típicamente solo hay un reward al final. Los rewards intermedios son cero. Esto hace el problema un poco más simple que el RL "general".

> 💬 **Tu preguntaste**: *"¿cuál es mi espacio de estados? ¿el vocabulario? ¿esas son las acciones? ¿o la palabra más probable siguiente?"*
>
> **Respuesta puntual**:
> - **¿El vocabulario es el espacio de estados?** NO. El vocabulario es el **espacio de acciones** (qué token elegir).
> - **¿Las palabras son las acciones?** SÍ. Cada token es una acción.
> - **¿La palabra más probable siguiente?** No exactamente. La acción es **cualquier token muestreado de la distribución `π(a|s)`**, no necesariamente el más probable. Si haces greedy sampling sí es el más probable, pero con temperature/top-k cualquier token con probabilidad positiva es una acción válida.
> - **El estado es el contexto acumulado** (prompt + tokens generados hasta ese punto). Es lo que el LLM ve como input antes de decidir el próximo token.

---

## 2. ¿Qué es exactamente la política? ¿Son los pesos? ¿Es el LLM?

Distinción semántica importante:

- La **política `π_θ(a|s)`** es una **función matemática** que mapea estados a distribuciones sobre acciones.
- Los **pesos `θ`** son los **parámetros** que definen esa función específica.
- El **LLM** es la implementación neural concreta de esa función.

**Las tres cosas refieren a lo mismo**, vistas desde tres ángulos:

| Perspectiva | Cómo lo ves |
|---|---|
| Matemática | `π_θ(a\|s)` — una distribución condicional |
| Estadística | Una función parametrizada por `θ` |
| Implementación | Un LLM (red neuronal) con pesos `θ` |

**Frase mental**: "el LLM **es** la política". Cuando dices "el LLM hace X", estás diciendo "la política `π_θ` hace X". Cuando dices "fine-tuning del LLM", estás diciendo "actualizar los pesos `θ` que definen la política".

### Política vs política óptima

- **Política `π`**: cualquier estrategia. Puede ser buena, mala o mediocre.
- **Política óptima `π*`**: la mejor de todas. La que maximiza el return esperado.

En RL **partes de una política y la mueves hacia algo mejor**. En LLMs partes del SFT y haces RLHF/DPO para acercarte (no necesariamente llegar) a `π*`.

> 💬 **Tu preguntaste**: *"¿la política está parametrizada por los pesos? ¿la política son los pesos?"*
>
> **Respuesta puntual**: Casi, pero hay un matiz. La política es la **función** `π_θ(a|s)`; los pesos `θ` son los **parámetros** que la definen. En la práctica, "el LLM" y "la política" son intercambiables — el LLM **es** la política. Lo que parametriza esa política son sus pesos.

> 💬 **Tu preguntaste**: *"¿la gracia es hacerle fine-tuning a los pesos para orientarlos a hacer algo que nos gusta o preferimos?"*
>
> **Respuesta puntual**: Sí, exactamente. RLHF, DPO y Self-Rewarding **mueven los pesos `θ` en el espacio de pesos** para que la política resultante genere respuestas más alineadas con preferencias humanas (o con lo que el reward model captura como "preferible"). Cada update de gradient descent es un paso pequeño en ese espacio. La trayectoria de pesos es:
> ```
> θ_SFT → θ_1 → θ_2 → ... → θ_final
> ```

---

## 3. ¿De dónde partimos? ¿Modelo base o SFT?

- **Modelo base**: el LLM pre-entrenado con next-token prediction sobre internet. Sabe completar texto pero no sigue instrucciones bien.
- **Modelo SFT**: el base + fine-tuning supervisado sobre `(instrucción, respuesta_humana)`. Ya sigue instrucciones razonablemente.

**RLHF, DPO y Self-Rewarding empiezan desde SFT**, no desde el modelo base puro. El SFT es tu `π_0` (política inicial), y desde ahí mueves los pesos.

> 💬 **Tu preguntaste**: *"¿empezamos con un LLM base entiendo yo sí?"*
>
> **Respuesta puntual**: Casi. Empezamos con el LLM **SFT**, que es el modelo base + un fine-tuning supervisado previo. El SFT ya sigue instrucciones, y a partir de ahí RLHF/DPO refinan más.

---

## 4. ¿Qué es una trayectoria en este contexto?

Una trayectoria `τ` es la **secuencia completa de generación de una respuesta**:

```
τ = (s_0, a_0, s_1, a_1, ..., s_T)
```

Si generas 100 tokens, tu trayectoria tiene 100 pasos. Cada `a_t` es un token, cada `s_t` es el contexto acumulado.

> 💡 **Esto importa para policy gradient**: la fórmula `∇_θ E[return]` involucra una expectación sobre trayectorias. Cuando entrenas RLHF, samplear una trayectoria = generar una respuesta completa con la política actual.

> 💬 **Tu preguntaste**: *"el policy gradient theorem no lo termino de entender, ¿a qué te refieres con trayectoria?"*
>
> **Respuesta puntual**: Una trayectoria es **una respuesta completa generada por el LLM**. Empieza con el prompt (`s_0`), y cada token generado es una acción que avanza al siguiente estado. La trayectoria termina cuando el LLM emite un token de fin o llega al límite de longitud. Para entrenar con policy gradient, sampleas muchas trayectorias (= generas muchas respuestas) con la política actual y calculas el gradiente sobre ellas.

---

## 5. Policy gradient sin misterio

Tu política `π_θ` está parametrizada por `θ`. Quieres maximizar `E[return]`. El teorema dice:

> 📐 **Policy gradient**:
> ```
> ∇_θ E[return]  =  E[ Σ_t  ∇_θ log π_θ(a_t | s_t) · A_t ]
> ```

### Conexión con MLE (lo que ya conoces)

`∇_θ log π_θ(a|s)` es **el mismo gradiente** que usas en MLE en NLP. Es backprop estándar sobre la log-likelihood de la acción tomada.

```
MLE en NLP:
    ∇_θ J = E[ ∇_θ log p_θ(y | x) ]
                          ↑ peso implícito = 1 (todos los y son "correctos")

Policy gradient:
    ∇_θ J = E[ ∇_θ log π_θ(a | s) · A_t ]
                          ↑ mismo gradiente,    ↑ ponderado por advantage
```

Si `A_t = 1` para todas las acciones, **policy gradient = MLE**.

La única diferencia conceptual: en RL las acciones tienen distintas "calidades" (advantages), y eso pondera el gradiente.

### ¿Es SGD?

Sí, en la práctica se usa **SGD/Adam/AdamW**. "Estocástico" porque cada update se basa en una muestra finita de trayectorias (mini-batch), no en la expectación exacta.

> 💬 **Tu preguntaste**: *"el delta pesos log política al final eso es el gradiente descendente estocástico? eso no me acuerdo bien"*
>
> **Respuesta puntual**: Sí. El "delta pesos × log política" es exactamente un step de SGD aplicado al policy gradient. La fórmula `∇_θ log π_θ(a|s) · A_t` es el gradiente; lo multiplicas por un learning rate `α` y restas (porque maximizas en lugar de minimizar, técnicamente sumas):
> ```
> θ ← θ + α · ∇_θ log π · A_t
> ```
> Es **backprop estándar** sobre el LLM, ponderado por `A_t`.

> 💬 **Tu preguntaste**: *"¿es el loss que usamos? ¿como la sumatoria de la resta del likelihood max ratio?"*
>
> **Respuesta puntual**: Lo que estás recordando con "likelihood max ratio" probablemente es **PPO** (que veremos después), no policy gradient básico. PPO sí usa un **ratio de likelihoods** `ρ_t = π_θ / π_θ_old` (esto se llama importance sampling) y un clipping de ese ratio. Eso es una mejora **encima** del policy gradient básico. Por ahora quedémonos con el caso simple (REINFORCE): gradiente de log-likelihood × advantage.

---

## 6. Logits ≠ política

Los **logits** son las salidas pre-softmax del LLM. Para cada posición, el LLM produce un vector de tamaño `|vocab|` con números reales.

La **política** se obtiene aplicando softmax a los logits:

```
LLM(contexto) → logits ∈ R^|vocab| → softmax → π(a|s)
                  ↑                              ↑
            no es probabilidad         sí es probabilidad (suma 1)
```

Cuando dices "el LLM produce una política", técnicamente quieres decir "el LLM produce logits, que después de softmax son la política".

`log π_θ(a|s)` = `log_softmax(logits)[a]` — esto es lo que aparece en la fórmula del policy gradient.

> 💬 **Tu preguntaste**: *"¿o son los logits? ¿o qué era eso?"*
>
> **Respuesta puntual**: Logits y política son cosas distintas pero relacionadas. **Logits** son los números crudos pre-softmax. **Política** es la distribución de probabilidad post-softmax. El gradiente del policy gradient (`∇log π`) se calcula sobre la log-probabilidad post-softmax (que en código se obtiene con `log_softmax(logits)`), pero en términos de la red neuronal, sigues haciendo backprop a través de la red entera — los logits son solo un punto intermedio en el forward pass.

---

## 7. `A_t`: rango y por qué se normaliza

`A_t` (el **advantage**) es un **escalar real arbitrario**. Puede ser positivo, negativo, grande o pequeño. No tiene rango fijo.

- `A_t > 0`: la acción fue mejor que el promedio que tomaría la política → empuja hacia arriba.
- `A_t < 0`: peor que el promedio → empuja hacia abajo.
- `A_t ≈ 0`: promedio → gradiente pequeño.

### ¿Por qué se normaliza?

Dos razones distintas:

**Razón 1 — Escala absoluta**: el update es `θ ← θ + α · ∇log π · A_t`. Si `A_t` puede valer `+1000`, el update es enorme y los pesos saltan por todos lados. Cambias el reward de "0-1" a "0-100" y el entrenamiento se rompe sin normalizar.

**Razón 2 — Varianza alta**: aunque `A_t` esté en escala razonable, varía mucho entre samples. Una trayectoria puede tener `A = +5`, otra `A = -3`. Esa varianza hace que el gradiente promedio sea ruidoso.

### Cómo se normaliza

Sobre cada batch de trayectorias:
```
mean_A = mean(A_t en el batch)
std_A  = std(A_t en el batch)
A_t_norm = (A_t − mean_A) / (std_A + ε)
```

Esto deja los advantages con media 0 y desviación 1.

> 💡 **Justificación teórica**: restar una constante (baseline) a `A_t` **no cambia el gradiente esperado** — solo reduce varianza. Esta propiedad permite normalizar sin sesgar el entrenamiento.

> 💬 **Tu preguntaste**: *"¿A_t es un escalar que se estima como un valor entre cuánto y cuánto?"*
>
> **Respuesta puntual**: No tiene rango fijo. Puede ser cualquier número real. Típicamente está en `[-5, +5]` después de normalización en el batch, pero sin normalizar puede valer cualquier cosa según la escala del reward.

> 💬 **Tu preguntaste**: *"¿por qué si A_t no se normaliza el gradiente estalla?"*
>
> **Respuesta puntual**: Por dos razones combinadas. Una es **escala absoluta** — si `A_t` puede valer `+1000` (porque tu reward tiene magnitud grande), los updates `α · ∇log π · 1000` son enormes y los pesos divergen. La otra es **varianza alta** — entre muestras del batch `A_t` puede saltar mucho, lo que hace al gradiente ruidoso. Normalizar `(A_t − mean) / std` arregla ambos problemas: pone el advantage en escala `~N(0, 1)` y permite usar learning rates predecibles.

---

## 8. `V(s_t)`: ¿futuros o histórico?

**Futuros, no histórico.** Es una **predicción**, no un promedio de lo observado.

> 📐 **Definición**:
> ```
> V^π(s_t) = E[ Σ_{k=t}^{T} γ^k · r_k  |  empezar en s_t, seguir π ]
> ```
>
> "Estando en `s_t`, si sigo la política `π` desde ahora, ¿cuál es el reward total que **espero** acumular hasta el final?"

### Cómo se calcula en la práctica

No conoces `V(s)` exactamente. Lo **aprendes** con otra red (el **critic** o **value head**) usando **TD learning** vía Bellman:

```
Target para el critic en el paso t:
    V_target(s_t) = r_t + γ · V_φ(s_{t+1})
                    ↑ observado     ↑ predicho por el critic

Loss del critic:
    L_critic = MSE( V_φ(s_t), V_target(s_t) )
```

Iterativamente, las predicciones se calibran a sí mismas.

> ⚠️ **Importante**: `V(s)` es una **función**, no un valor histórico. Para un mismo estado `s`, `V(s)` tiene un único valor dada la política. No es "el promedio de lo que ya viste" — es "el valor esperado en expectación, según tu modelo del mundo".

### Actor-critic loop

- **Actor** = la política (el LLM).
- **Critic** = la red que predice `V_φ(s)`.
- Se entrenan **al mismo tiempo**, alternando updates. El critic le da advantages al actor; el actor genera trayectorias que entrenan al critic.

> 💬 **Tu preguntaste**: *"¿el V s_t qué era?"*
>
> **Respuesta puntual**: Es la **value function** — una estimación de "cuánto reward total espero acumular desde este estado hacia el futuro, si sigo mi política actual". Se aprende con otra red neuronal separada llamada **critic**. En el actor-critic loop, el critic es el cuarto modelo en memoria (después de la política, la referencia para KL, y el reward model).

> 💬 **Tu preguntaste**: *"¿V s_t es el valor esperado de los rewards futuros? ¿o de lo que se ha observado?"*
>
> **Respuesta puntual**: De los **rewards futuros**, no de lo observado. Es una **predicción/expectación**, no un histórico de rewards pasados. Conceptualmente: "si me paro aquí y miro hacia adelante con mi política actual, ¿qué tanto reward espero recolectar?". Se aprende vía TD learning con Bellman, usando el reward observado de UN paso + la predicción del propio critic para el siguiente estado como target.

---

## 9. ¿De dónde viene el reward? Los 4 paradigmas

**Depende del paradigma**. Esta es la diferencia clave entre RLHF / RLAIF / Self-Rewarding / GRPO.

### a) RLHF clásico (Ouyang 2022, InstructGPT)

```
1. Humanos rankean pares de respuestas (ÚNICA INTERVENCIÓN HUMANA).
   Dataset: (x, y_w, y_l).

2. Entrenas un reward model r̂(x, y) con pérdida Bradley-Terry.

3. CONGELAS r̂.

4. Durante PPO: cada respuesta generada por la política se pasa a r̂,
   que produce un número. Ese número es el reward.

   r̂ actúa como "un humano destilado en una red neuronal".
```

Los humanos anotan **una vez al principio**. Durante el RL, el reward model actúa solo.

### b) RLAIF (Lee 2023) / Constitutional AI (Bai 2022)

Mismo pipeline que RLHF, pero las comparaciones las hace un **LLM externo** (juez), no humanos. El reward model se entrena con esas comparaciones AI-generadas.

### c) Self-Rewarding (Yuan 2024) — el paper ancla

El **mismo modelo** que entrenas actúa como juez. Le pides al modelo que se autoevalúe con una rúbrica.

**Detalle clave**: aquí **no se entrena un reward model separado**. Los pares de preferencia generados por el juez interno van **directo a DPO**, que codifica el reward implícitamente vía `r̂ = β · log(π_θ / π_ref)`.

### d) GRPO / DeepSeek-R1

El reward es una **función verificable**, no una red:
- `r = 1` si la respuesta a un problema matemático es correcta, `0` si no.
- `r = 1` si el código pasa los tests.

No hay reward model ni juez LLM. Hay un verificador automático. Solo funciona en dominios con respuesta verificable.

### Tabla resumen

| Método | ¿Quién da el reward durante el entrenamiento? |
|---|---|
| RLHF | Reward model entrenado con rankings humanos |
| RLAIF | Reward model entrenado con rankings de LLM juez externo |
| Constitutional AI | Reward model entrenado con rankings según principios |
| Self-Rewarding | El propio modelo (sin reward model — DPO directo) |
| GRPO / R1 | Función verificable computable |

> 💬 **Tu preguntaste**: *"¿el reward lo da un judge? ¿o cómo se da ese reward?"*
>
> **Respuesta puntual**: Depende del paradigma. En **RLHF clásico** lo da un **reward model** (red neuronal entrenada con rankings humanos), no un judge. En **RLAIF** sí lo da un judge LLM externo. En **Self-Rewarding** lo da el propio modelo actuando como juez (sin reward model separado — el reward queda codificado implícitamente en DPO). En **GRPO/R1** lo da una función verificable (correcto/incorrecto). La tabla de arriba resume las 4 opciones.

> 💬 **Tu preguntaste**: *"¿cómo se sabe? ¿debería ser un conjunto de datos validados ya por alguien?"*
>
> **Respuesta puntual**: Sí, en **RLHF clásico** los humanos validan/rankean el dataset original. Concretamente: anotadores humanos comparan pares de respuestas y eligen cuál es mejor; ese dataset `(x, y_w, y_l)` validado es lo que se usa para entrenar el reward model. **Los humanos anotan UNA vez al principio.** Durante el RL, el reward model actúa solo. InstructGPT usó ~50.000 comparaciones humanas — caro, y por eso surgen RLAIF (LLM juez) y Self-Rewarding (juez = mismo modelo).

---

## 10. SFT: ¿LoRA o full fine-tuning?

Ambos son posibles. **Depende de tu presupuesto**.

- **Full fine-tuning**: actualizas todos los pesos del modelo. Caro en memoria y cómputo, pero típicamente mejor calidad. **Es lo que hacen los papers grandes (InstructGPT, Yuan 2024).**
- **LoRA / PEFT** (parameter-efficient fine-tuning): congelas la mayoría de pesos y añades adaptadores pequeños entrenables. Mucho más barato. Menor calidad pero "suficiente" para muchas aplicaciones.

En el contexto de **Self-Rewarding (Yuan 2024)**: es **full fine-tuning** sobre Llama 2 70B. No usan LoRA.

En las explicaciones del wiki/output asumimos full fine-tuning porque es lo que hace el paper ancla. En aplicaciones reales, la mayoría usa LoRA por costo.

> 💬 **Tu preguntaste**: *"¿supervised fine-tuned es LoRA? ¿o es fine-tuning completo?"*
>
> **Respuesta puntual**: En el paper ancla (Yuan 2024) es **full fine-tuning** sobre Llama 2 70B. En general, "SFT" puede referirse a cualquiera de los dos — LoRA o full. La elección depende de tu presupuesto de GPU y de cuánta calidad necesitas. Si reproduces papers grandes, normalmente es full. Si haces aplicaciones prácticas con recursos limitados, LoRA es lo estándar.

---

## 11. Auto-verificación: cinco preguntas

Si puedes responder estas cinco sin volver a leer, estás listo para avanzar a PPO:

### Pregunta 1
*En un LLM, ¿qué es el estado y qué es la acción?*

> Respuesta: el estado es el contexto actual (prompt + tokens generados hasta ahora). La acción es el próximo token a generar.

### Pregunta 2
*Si la política es el LLM, ¿qué son los pesos `θ`?*

> Respuesta: los parámetros que definen la política específica. Distintos pesos = distintas políticas.

### Pregunta 3
*Si `A_t = 1` para todas las acciones, ¿en qué se convierte el policy gradient?*

> Respuesta: en MLE (maximum likelihood estimation) estándar.

### Pregunta 4
*¿Por qué normalizamos `A_t`?*

> Respuesta: para que el entrenamiento no dependa de la escala absoluta del reward y para reducir varianza.

### Pregunta 5
*¿De dónde viene el reward en RLHF clásico vs en Self-Rewarding?*

> Respuesta: en RLHF, de un reward model entrenado con rankings humanos. En Self-Rewarding, del propio modelo actuando como juez interno, sin reward model separado (se conecta directo con DPO).

---

## 12. Qué viene después

Con esto resuelto, ya puedes:

- Ir a [../output/01-rlhf.md](../output/01-rlhf.md) — el pipeline completo de RLHF (donde el reward model que vimos se conecta con PPO).
- O ir a [../output/00-fundamentos-rl.md](../output/00-fundamentos-rl.md) §6 — la explicación detallada de PPO con clipping, importance sampling, GAE.
- O ir a [../output/02-dpo.md](../output/02-dpo.md) — saltarte PPO y entender la alternativa que usa Self-Rewarding.

---

## Lecturas relacionadas

- [../output/00-fundamentos-rl.md](../output/00-fundamentos-rl.md) — RL básico desde MDP.
- [../output/00b-informacion-y-entropia.md](../output/00b-informacion-y-entropia.md) — KL, entropía, MaxEnt, Boltzmann.
- [../output/01-rlhf.md](../output/01-rlhf.md) — pipeline canónico de RLHF.
