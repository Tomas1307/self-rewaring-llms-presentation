# Camino a PPO: REINFORCE, los tres dolores, y conceptos auxiliares

> **Para qué este archivo**: continúa donde quedó [01-mapping-llm-rl.md](01-mapping-llm-rl.md). Antes de llegar a PPO necesitamos entender bien **REINFORCE** (el policy gradient básico), **por qué no basta para LLMs**, y un puñado de conceptos auxiliares (on-policy vs off-policy, mini-batch SGD, el reward model y sus sesgos, advantage, discount).

> **Convención**: los bloques `💬 Tu preguntaste: "..."` recogen preguntas literales de la conversación. Justo debajo va la respuesta puntual.

---

## 1. REINFORCE clásico — qué fue, por qué fue innovador

REINFORCE (Williams 1992) fue uno de los primeros algoritmos de **policy gradient** exitosos. La idea matemática es la que ya viste:

```
∇_θ E[return] = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · A_t ]
```

Para entender por qué importó, hay que mirar **qué existía antes** en RL.

### Antes de REINFORCE: métodos value-based

En los 80 y principios de los 90, RL era casi todo **value-based**:

- **Q-learning** (Watkins 1989): aprende `Q(s, a)` = "qué tan bueno es tomar acción `a` en estado `s`". La política se deriva como `π(s) = argmax_a Q(s, a)`.
- **TD learning** (Sutton 1988): aprende `V(s)` directamente vía Bellman.
- **SARSA**: variante on-policy de Q-learning.

Funcionaban bien para problemas chicos y discretos. Pero tenían **tres problemas serios** que REINFORCE resolvió.

### Por qué REINFORCE fue innovador

#### Razón 1 — Escala a espacios de acción grandes o continuos

Q-learning requiere calcular `argmax_a Q(s, a)`. Si tu espacio de acciones es:
- **Continuo** (robot con joints en cualquier ángulo) → el argmax es un problema de optimización por sí mismo.
- **Discreto pero enorme** (vocabulario de 50.000 tokens) → el argmax es caro.

REINFORCE no necesita argmax. Solo necesita **muestrear** de la política y calcular gradientes. Esto es trivial sea cual sea el tamaño del espacio.

#### Razón 2 — Optimiza directamente lo que importa

Q-learning aprende `Q`, y **luego** derivas la política de `Q`. Es indirecto. Si la estimación de `Q` está mal en una región, tu política puede ser arbitrariamente mala.

REINFORCE **optimiza directamente la política** con gradient descent sobre `E[return]`. Cada gradient step mejora lo que te importa.

#### Razón 3 — Funciona con políticas estocásticas

Q-learning tiende a aprender políticas determinísticas (`argmax`). Para problemas donde la política óptima es estocástica (piedra-papel-tijera, donde necesitas jugar al azar), Q-learning falla.

REINFORCE maneja políticas estocásticas naturalmente, porque optimiza la distribución `π(a|s)` directamente.

#### Y la innovación técnica específica: el log-derivative trick

Williams formalizó la identidad `∇P = P · ∇log P` aplicada a trayectorias. Antes había intuiciones sobre "policy improvement" pero no una fórmula limpia y derivable. Williams dio **una fórmula única** (`∇log π · G`) que es matemáticamente correcta y se implementa con backprop estándar.

### REINFORCE clásico vs en LLMs

| Aspecto | REINFORCE clásico | REINFORCE en LLMs |
|---|---|---|
| Política | Red pequeña (MLP, CNN simple) | Transformer de miles de millones de parámetros |
| Estado | Vector de pocos floats | Contexto de cientos de tokens |
| Acción | Pocas opciones | Vocabulario de ~50.000 tokens |
| Reward | En cada paso | **Solo al final** (sparse) |
| Trayectoria | 10-1000 pasos | Cientos de tokens |
| Critic | Opcional, baseline simple | **Casi obligatorio** |
| Regularización | Ninguna típicamente | **KL contra `π_ref`** |

> 💡 **La esencia del algoritmo es la misma**: `θ ← θ + α · ∇log π · A_t`. Lo que cambia es la escala, la sparsidad del reward, y los añadidos para hacerlo manejable.

> 💬 **Tu preguntaste**: *"¿cómo es el reinforce original? ¿en qué difiere del que usamos aquí?"*
>
> **Respuesta puntual**: el algoritmo es matemáticamente el mismo. Lo que cambia entre REINFORCE clásico y "REINFORCE para LLMs" es la escala del modelo (gigante en LLMs), la sparsidad del reward (solo al final en LLMs), la presencia de un critic separado, y la KL regularization. En el contexto del wiki, cuando digo "REINFORCE" me refiero a la idea matemática del policy gradient. PPO es una mejora encima de esa idea.

> 💬 **Tu preguntaste**: *"¿por qué REINFORCE fue bueno en su época? ¿por qué no se usaba un algoritmo antes? ¿por qué fue innovador?"*
>
> **Respuesta puntual**: porque resolvió tres problemas estructurales de Q-learning y métodos value-based: escala a espacios de acción grandes/continuos, optimiza directamente lo que importa (la política), maneja políticas estocásticas. Y técnicamente formalizó el log-derivative trick que conecta policy gradient con backprop estándar — eso le dio una implementación limpia que cualquiera con autodiff podía usar.

---

## 2. On-policy vs off-policy

- **On-policy**: aprendes la política mientras **la ejecutas**. Los samples que usas para entrenar **vienen de la política actual `π_θ`**. Si tu política cambia, los samples viejos ya no son válidos.
- **Off-policy**: aprendes una política mientras **ejecutas otra distinta**. Los samples pueden venir de cualquier fuente — una política vieja, una política diferente, un dataset estático.

No tiene que ver con "saber la política inicialmente" ni con "descubrir la óptima". Esos son los **objetivos** (siempre quieres descubrir la óptima). La distinción on/off es sobre **de dónde vienen los samples**.

### Tabla de algoritmos

| Algoritmo | Tipo | Por qué |
|---|---|---|
| **REINFORCE** | On-policy | Cada gradiente requiere samples de la política actual. |
| **SARSA** | On-policy | El target Q usa la acción que efectivamente tomó la política actual. |
| **Q-learning** | Off-policy | El target Q usa el `max` sobre acciones, independiente de qué tomó. |
| **DPO** | Off-policy / offline | Los pares de preferencia se generan antes del entrenamiento. La política nunca genera nada durante DPO. |
| **PPO** | On-policy "relajado" | Genera samples, pero permite reusarlos varios steps mientras la política no se aleje mucho. |

> 💬 **Tu preguntaste**: *"recuerdo que había on policy y off policy, on policy se sabe inicialmente la política y off policy se descubre la política óptima? no recuerdo como era"*
>
> **Respuesta puntual**: lo que recordabas está mezclado. La distinción real es sobre **de dónde vienen los samples que usas para entrenar**:
> - **On-policy**: samples de tu política de **ahora** (cambia la política → tiras los samples).
> - **Off-policy**: samples de **cualquier otra parte** (puedes reusar samples viejos, datasets fijos, etc.).
>
> No tiene que ver con "saber la política" ni con el objetivo de "encontrar la óptima". Frase mental: *"on-policy = los samples deben venir de la política actual; off-policy = los samples pueden venir de cualquier distribución"*.

---

## 3. Los tres dolores de REINFORCE puro

En LLMs, REINFORCE básico tiene tres problemas graves. PPO resuelve los tres con tres trucos distintos.

### Dolor 1 — REINFORCE es estrictamente on-policy

Si tomas un step de gradient descent y los pesos cambian, ya tienes una política nueva `π_θ'`. **Los samples que tenías ya no son válidos** — vienen de `π_θ`, no de `π_θ'`.

Consecuencia: **cada batch de respuestas se usa una sola vez**. En LLMs donde generar respuestas con un modelo de 70B es lo más caro del proceso, esto es brutalmente ineficiente.

### Dolor 2 — Pasos destructivos

El update `θ ← θ + α · ∇log π · A_t` puede mover demasiado la política si:
- `α` es muy grande,
- O `A_t` tiene magnitud grande,
- O haces varios updates encadenados sobre el mismo batch.

Si un step mueve mucho la política, el LLM puede colapsar (producir texto incoherente, repetitivo, o degenerado). Una vez colapsado, es prácticamente irrecuperable.

### Dolor 3 — Alta varianza del gradiente

`A_t` varía mucho entre samples. Sin reducción de varianza, necesitas batches gigantes para que el promedio sea estable.

### Cómo PPO ataca cada uno

| Dolor | Truco de PPO |
|---|---|
| 1. Estrictamente on-policy | **Importance sampling** — reusar samples viejos con un peso correctivo |
| 2. Pasos destructivos | **Clipping** del peso correctivo — limita cuánto puede cambiar la política por paso |
| 3. Alta varianza | **Critic + GAE** — estima `A_t` con menos ruido |

---

## 4. Aclaraciones técnicas (alpha, "step pequeño", normalizar)

> 💬 **Tu preguntaste**: *"¿el alpha del update te refieres al learning rate sí?"*
>
> **Respuesta puntual**: sí, exactamente. `α` en `θ ← θ + α · ∇log π · A_t` es el learning rate, lo mismo que en cualquier SGD/Adam. Valores típicos para RLHF: `α = 1e-6` a `5e-6` (mucho más conservador que en SFT, que usa `α = 1e-5` a `1e-4`).

> 💬 **Tu preguntaste**: *"¿cuándo dices step pequeño es token por token?"*
>
> **Respuesta puntual**: NO. "Step" no se refiere a tokens. Se refiere a **un update de los pesos del modelo**.
>
> Jerarquía:
> - Un **token** es una acción individual.
> - Una **trayectoria** (= respuesta completa) tiene muchos tokens.
> - Un **batch** tiene varias trayectorias.
> - Un **step de gradient descent** = una actualización de los pesos, calculada sobre el batch.
>
> "Step pequeño" en PPO significa: **el nuevo `θ` no debe estar muy lejos del viejo `θ`**. Se mide en términos de cuánto cambia `π(a|s)` para los `(s, a)` conocidos. No tiene nada que ver con tokens individuales.

> 💬 **Tu preguntaste**: *"¿por qué no es suficiente normalizar A_t?"*
>
> **Respuesta puntual**: porque normalizar `A_t` solo arregla la varianza (Dolor 3), pero **no arregla**:
> - **El Dolor 1 (on-policy)**: aunque normalices `A_t`, el gradiente `∇log π_θ · A_t` sigue requiriendo que `π_θ` sea la política actual. Si quieres reusar samples viejos, necesitas el ratio de importance sampling — eso es lo que añade PPO.
> - **El Dolor 2 (pasos destructivos)**: con `A_t` normalizado en `~[-2, +2]` un solo step típico está bien, pero **varios updates encadenados** sobre el mismo batch pueden acumular y alejar la política demasiado. Eso solo lo arregla **el clipping** del ratio.
>
> Normalizar `A_t` es necesario pero **no suficiente**.

---

## 5. El batch de respuestas — de dónde sale, qué reward recibe

### Los prompts vienen de un dataset

```
Dataset de prompts: ~100.000 instrucciones humanas.
                    Ejemplo:
                      ["Explícame qué es un transformer.",
                       "Escribe un poema sobre el otoño.",
                       "Tradúceme al francés...",
                       ...]

En cada iteración:
  1. Tomar 256 prompts del dataset (muestreo aleatorio).
  2. Para cada prompt x_i:
     a. Pasar x_i por la política actual π_θ.
     b. Generar una respuesta y_i muestreando tokens.
  3. Resultado: 256 pares (x_i, y_i).
```

Los **prompts** vienen de un dataset fijo (humanos los escribieron). Las **respuestas** se regeneran en cada iteración con la política actual.

> 💬 **Tu preguntaste**: *"¿cuándo dices generar 256 respuestas es frente a un conjunto de datos de preguntas?"*
>
> **Respuesta puntual**: sí. Los 256 prompts vienen del dataset de instrucciones. Para cada prompt, generas UNA respuesta con tu política actual. Los prompts son fijos (no cambian); las respuestas se regeneran cada iteración.

### En RLHF no hay "respuesta correcta" por prompt

Aquí hay una confusión a desempacar:

- **En SFT** tienes `(prompt, respuesta_humana)` pares. Entrenas a que el modelo reproduzca la respuesta humana. Hay ground truth.
- **En RLHF** tienes solo prompts. **No hay "respuesta correcta"** que el modelo deba reproducir.

Lo que tienes adicionalmente es **el dataset de comparaciones** que se usó para entrenar el reward model:
```
Comparisons dataset:
  (prompt_1, respuesta_A, respuesta_B, "humano dijo A es mejor")
  (prompt_2, respuesta_C, respuesta_D, "humano dijo D es mejor")
  ...
```

Esto NO se usa durante el RL. Se usó UNA sola vez para entrenar `r̂`. Después `r̂` actúa solo.

> 💬 **Tu preguntaste**: *"¿cuándo tenemos ejemplos de prompts también se tienen ejemplos de respuesta? Y de ahí supongo que similitud coseno entre ambos un threshold y se toma 1 como bueno 0 como malo?"*
>
> **Respuesta puntual**: NO. En RLHF no hay "respuesta correcta" por prompt — solo prompts. La similitud coseno con threshold sería un mecanismo para tareas con respuesta objetiva (e.g., embedding retrieval). En RLHF la calidad es **subjetiva**. El reward lo da el **reward model** (otro LLM con cabeza escalar) que aprendió a imitar el juicio humano sobre pares de respuestas. NO es similitud coseno, NO es un threshold, NO hay ground truth para comparar.

> 💬 **Tu preguntaste**: *"al final de las 256 respuestas se saca un r_t por cada respuesta y se promedia?"*
>
> **Respuesta puntual**: cada respuesta tiene su propio `r̂(x_i, y_i)` (un número por respuesta). Esos rewards **NO se promedian directamente**. Lo que se hace:
> 1. Calcular advantage `A_i` per-sample (usando el critic).
> 2. Calcular gradiente per-sample: `∇log π · A_i`.
> 3. **Promediar los gradientes** sobre el batch: `gradiente_total = (1/256) · Σ gradient_i`.
> 4. Update: `θ ← θ + α · gradiente_total`.
>
> El promedio aparece **al final, sobre los gradientes**. No mezcles "promediar gradientes" (sí se hace) con "promediar rewards" (no se hace).

---

## 6. El reward model — qué es, cómo se entrena, sus sesgos

### Qué es

- Es **otro LLM** (típicamente del tamaño del modelo a entrenar, o más pequeño).
- Está **entrenado específicamente en la tarea de calificar**.
- Sobre **datos humanos** (comparaciones pareadas).
- Con **pérdida Bradley-Terry**.
- Se **congela** después de entrenarlo. No se vuelve a tocar durante el RL.

Una vez congelado, actúa como una **función estimadora de calidad humana**.

### Sesgos documentados

El RM hereda los sesgos de sus anotadores humanos. Esos sesgos se transmiten al LLM que entrena con él.

| Sesgo | Manifestación |
|---|---|
| **Length bias** | RM prefiere respuestas largas; LLM aprende a generar respuestas innecesariamente largas. |
| **Sycophancy** | Si el prompt contiene opinión, RM prefiere respuestas que la validan; LLM se vuelve adulador. |
| **Cultural bias** | Anotadores típicamente angloparlantes/USA → RM impone esa noción de "apropiado". |
| **Formatting bias** | RM premia listas, negritas, estructura visual sin importar contenido; LLM aprende a "performar formato". |

### El cuello de botella humano

Si el anotador humano **no puede juzgar** una respuesta (porque el tema le supera, o porque el modelo es mejor que el humano promedio en esa tarea), el RM hereda esa incapacidad.

Ejemplo: si entrenas un LLM que escribe pruebas matemáticas y los anotadores no son matemáticos, sus rankings serán ruidosos o basados en superficialidades. El RM entrenado con eso será **incapaz de premiar pruebas correctas** vs incorrectas bonitas. El LLM aprende a producir **pruebas que se ven bonitas pero no son correctas**.

Esto es el "cuello de botella humano" — uno de los argumentos centrales del paper de Yuan 2024 para motivar Self-Rewarding.

### Por qué importa para Self-Rewarding

La cadena lógica:
```
1. RLHF requiere un reward model.
2. El RM está sesgado por sus anotadores humanos.
3. Si quieres modelos mejores que un humano promedio, los humanos no
   pueden anotar bien → el RM se vuelve un cuello de botella.
4. Solución propuesta: que el modelo se autoevalúe.
```

Self-Rewarding **acepta** este argumento. Pero atención: **no elimina el sesgo**, solo lo cambia. Ahora el sesgo es de **auto-preferencia** (el modelo prefiere outputs que él mismo produciría). Es el problema central que Wang 2025 y otras críticas identifican.

> 💬 **Tu preguntaste**: *"pero el reward model está sesgado porque si ese califica la salida para juicio humano las limitaciones que tenga para calificar o cosas que hagan falta no las va a penalizar para el otro llm que esta aprendiendo RLHF no?"*
>
> **Respuesta puntual**: tu razonamiento es 100% correcto. El RM hereda los sesgos y limitaciones de sus anotadores, y esos sesgos se transmiten directamente al LLM entrenado con él. Length bias, sycophancy, cultural bias, formatting bias, y el cuello de botella humano son todos consecuencias documentadas. Esto es **uno de los problemas centrales de RLHF** y motiva técnicas posteriores como RLAIF y Self-Rewarding.

> 💬 **Tu preguntaste**: *"ya entiendo no es similitud coseno es otro llm entrenado específicamente en la tarea de calificar dado un entrenamiento que tuvo sobre datos humanos para calificar"*
>
> **Respuesta puntual**: exacto. El reward model **es otro LLM** (típicamente con la misma arquitectura base que el modelo a entrenar, pero con una cabeza escalar en lugar de la cabeza de language modeling). Se entrena sobre comparaciones humanas con Bradley-Terry, y luego se congela. Actúa como un "humano destilado en una red neuronal" que asigna un score escalar a cualquier `(prompt, respuesta)`.

---

## 7. Mini-batch SGD — el estándar de deep learning

### Vanilla GD (gradient descent puro)

```
Repetir:
  1. Calcular gradiente exacto promediando sobre TODO el dataset:
       ∇L(θ) = (1/N) · Σ_{i=1}^{N}  ∇L_i(θ)
  2. Update: θ ← θ − α · ∇L(θ).
```

**Problema**: cada step pasa por todo el dataset. Si `N = 10 millones`, cada step toma horas. Inviable para deep learning.

### SGD puro (Stochastic Gradient Descent)

```
Repetir:
  1. Muestrear UN sample i aleatorio.
  2. Calcular gradiente con ese único sample.
  3. Update.
```

**Ventaja**: cada step es barato.
**Desventaja**: gradiente ruidoso, entrenamiento inestable.

"Estocástico" porque cada update usa un sample aleatorio, no el promedio del dataset.

### Mini-batch SGD (lo que TODO deep learning usa)

```
Repetir:
  1. Muestrear un mini-batch de B samples (típicamente B = 32, 64, 256).
  2. Calcular gradiente promedio:
       ∇L_batch(θ) = (1/B) · Σ_{i=1}^{B}  ∇L_i(θ)
  3. Update: θ ← θ − α · ∇L_batch(θ).
```

**Ventajas**:
- Cada step es manejable.
- Promediar `B` gradientes reduce ruido.
- Paraleliza bien en GPU.

**Adam, AdamW**: variantes de mini-batch SGD con momentum y adaptación del learning rate por parámetro. Mismo esqueleto.

### En RLHF

```
Por iteración:
  1. Tomar 256 prompts.                    ← mini-batch
  2. Generar 256 respuestas con π_θ.
  3. Calcular reward + advantage per-sample.
  4. Calcular gradiente per-sample.
  5. Promediar los 256 gradientes.         ← lo central de mini-batch SGD
  6. Update: θ ← θ + α · gradiente_promedio.
```

> 💬 **Tu preguntaste**: *"no conozco el mini batch SGD estándar"*
>
> **Respuesta puntual**: mini-batch SGD = entrenar con grupitos del dataset a la vez (típicamente 32-256 samples por step), **promediar los gradientes dentro del grupito**, y hacer un update por grupito. Es el estándar de deep learning moderno. Cuando se dice "SGD" en deep learning normalmente se refiere a esto, no al SGD puro de un sample. Adam y AdamW son variantes inteligentes de mini-batch SGD que adaptan el learning rate por parámetro.

---

## 8. El advantage A_t — diferencia, no promedio

### Definición

```
A_t = Q(s_t, a_t) − V(s_t)
```

- `Q(s_t, a_t)` = reward esperado si tomas esta acción específica `a_t` en `s_t`.
- `V(s_t)` = reward esperado si estás en `s_t` y sigues la política, **promediando sobre todas las acciones posibles**.

`A_t` mide: **¿esta acción específica fue mejor o peor que el promedio que mi política habría tomado en este estado?**

- `A_t > 0`: mejor que el promedio → reforzar.
- `A_t < 0`: peor que el promedio → desincentivar.
- `A_t = 0`: típica.

### Ejemplo intuitivo: ajedrez

- `V(s)` = "qué tan buena es esta posición en promedio si juego como suelo jugar".
- `Q(s, a)` = "qué tan buena es esta posición SI hago esta jugada específica".
- `A(s, a) = Q − V` = "esta jugada específica, ¿es mejor o peor que mi jugada promedio en esta posición?".

### Por qué importa "respecto al promedio"

Si solo usaras el reward bruto, reforzarías acciones que dan reward alto **incluso si toda acción en ese estado da reward alto**. No aprenderías nada útil.

El advantage te dice: **dada esta situación, ¿qué fue lo *distintivamente* bueno o malo?** Eso es lo que quieres aprender.

> 💬 **Tu preguntaste**: *"no entiendo el advantage el policy gradient"*
>
> **Respuesta puntual**: el advantage NO es un promedio de rewards. Es una **diferencia respecto al promedio**. Mide cuánto mejor (o peor) fue una acción específica comparada con la acción típica que tu política tomaría en ese mismo estado. Si el advantage es positivo, esa acción fue mejor que tu nivel promedio → refuerza. Si es negativo, peor → desincentiva. Por eso multiplica al gradiente `∇log π`: pondera **cuánto** modificar la probabilidad de cada acción.

---

## 9. Discount factor γ — futuro, no pasado

### Cómo funciona

```
Return desde el paso t:
  G_t = r_t + γ · r_{t+1} + γ² · r_{t+2} + γ³ · r_{t+3} + ...
```

Donde `γ ∈ [0, 1]`. Típicamente `γ = 0.99` en RL clásico.

- `γ = 1`: rewards futuros valen lo mismo que los presentes.
- `γ = 0.9`: rewards a 10 pasos en el futuro valen `0.9^10 ≈ 0.35` del presente.
- `γ = 0`: solo me importa el reward inmediato.

### Por qué usar discount

1. **Matemáticas**: si el episodio puede ser infinito, sin discount la suma de rewards puede divergir. `γ < 1` garantiza convergencia.
2. **Filosóficamente**: en muchos problemas, lo que pasa pronto es más predecible y valioso.
3. **Reducción de varianza**: rewards muy lejanos son ruidosos.

### En LLMs

**Típicamente `γ = 1`** (sin discount) por dos razones:
1. El reward es solo al final. No hay rewards intermedios que descontar.
2. Las trayectorias son cortas (cientos de tokens). No hay riesgo de divergencia.

### ¿Sesga al LLM?

En RL general, `γ < 1` sí introduce un sesgo hacia rewards inmediatos. En LLMs con reward al final y `γ = 1`, no introduce sesgo. El sesgo viene de otras fuentes (sesgos del RM, sesgos de anotadores).

> 💬 **Tu preguntaste**: *"reward con discount es para enfocarnos más en lo que pasó que en lo que esta pasando, pero eso no sesga a un llm?"*
>
> **Respuesta puntual**: dos aclaraciones. (1) El discount **se aplica al futuro**, no al pasado. Hace que rewards lejanos en el futuro pesen menos que los cercanos. (2) En LLMs típicamente `γ = 1` (sin discount) porque el reward es solo al final y las trayectorias son cortas. Así que el discount **no introduce sesgo en LLMs**. El sesgo del que hablábamos viene de otras fuentes (reward model, anotadores), no del discount.

---

## 10. La tabla de sensibilidad — qué representan los valores y por qué 7.39 es catastrófico

### Qué representa la tabla

La tabla mostraba **la nueva probabilidad de un token específico después de un update**. Los valores son **probabilidades** del token (que estaba en `0.10` al inicio).

Fórmula aproximada:
```
π_nueva(a|s) ≈ π_vieja(a|s) · exp(α · A_t)
```

Para `π_vieja = 0.10`:
```
π_nueva = 0.10 · exp(α · A_t)
```

### Ejemplos con la cuenta explícita

| Caso | α | A_t | exp(α · A_t) | π_nueva | Interpretación |
|---|---|---|---|---|---|
| Suave | 1e-6 | 1 | 1.000001 | 0.1000001 | Cambio invisible |
| OK | 1e-4 | 5 | 1.0005 | 0.10005 | Cambio mínimo |
| Notable | 1e-3 | 50 | 1.0513 | 0.10513 | 5% aumento relativo |
| Agresivo | 1e-3 | 500 | 1.6487 | 0.16487 | 65% aumento |
| Catastrófico | 1e-3 | 2000 | **7.389** | "0.7389" | **Imposible físicamente** |

### Por qué 7.39 es catastrófico

Las probabilidades **deben sumar 1** sobre todos los tokens del vocabulario. Si un token específico salta a `0.74` (de `0.10` inicial), los otros 50.000 tokens tienen que bajar para compensar.

Que la aproximación dé `7.39` significa que **el step intentado es físicamente imposible** — la aproximación lineal se rompió. En la red real (con softmax), esto se traduce en:

```
ANTES del update:
  logit_token = 0.0
  logits_otros ~ 0.0
  softmax → π_token = 0.10, π_otros distribuidos en 0.90

DESPUÉS del update destructivo:
  logit_token = +5.0   ← saltó mucho
  logits_otros ~ 0.0
  softmax → π_token ≈ 0.99
            π_otros ≈ 0.00001 cada uno
```

**El modelo colapsó.** Después:
1. Casi todas las generaciones siguientes producen ese token específico.
2. Las respuestas se vuelven repetitivas o degeneradas.
3. El reward dará scores bajos, pero el modelo ya está estructuralmente roto.
4. **Casi imposible recuperarlo.**

PPO con clipping previene esto: si el ratio `π_new / π_old` se acerca a `1+ε` o `1-ε`, el gradiente se anula. Es **un freno automático** que impide steps en la zona catastrófica.

> 💬 **Tu preguntaste**: *"explícame la tabla, eso vendría siendo el valor de la política? o sea no entiendo aun la tabla por qué es catastrófico 7.39?"*
>
> **Respuesta puntual**: los valores de la tabla son **probabilidades de un token específico** (no advantage ni reward). Empieza en 0.10 (10% de probabilidad). Después del update, esa probabilidad cambia según `α · A_t`. Si el resultado es mayor que 1 (como 7.39), significa que el step intentado es físicamente imposible — la aproximación lineal se rompió. En la red real, eso se traduce en **saturación del softmax**: un token recibe casi toda la probabilidad (~99%) y los demás casi nada (~0%). La política colapsa a una distribución degenerada y es prácticamente irrecuperable. **PPO con clipping evita esto** poniendo un freno cuando el ratio `π_new / π_old` se acerca a los bordes `[1−ε, 1+ε]`.

---

## 11. Importance sampling — preview (no entramos a fondo todavía)

PPO permite reusar el mismo batch para varios updates usando **importance sampling**. La idea:

```
Gradiente "viejo" (con samples de π_θ_old):
    ∇log π_θ_old · A_t

Gradiente "corregido" para π_θ_new (reusando los mismos samples):
    ρ_t · ∇log π_θ_new · A_t

    donde ρ_t = π_θ_new(a|s) / π_θ_old(a|s)
```

`ρ_t` corrige el sesgo de reusar samples viejos.

### ¿Por qué se necesita ρ_t?

Cuando reusas samples generados con `π_θ_old` para entrenar `π_θ_new`, hay un sesgo: los samples reflejan las preferencias de `π_θ_old`, no las de `π_θ_new`. El ratio corrige ese sesgo amplificando o reduciendo la contribución de cada sample según cuánto cambió su probabilidad.

> 💬 **Tu preguntaste**: *"PPO para la cuarta pregunta es acercar el gradiente viejo al nuevo por un factor multiplicativo que es el A_t básicamente para no tener que recalcular el gradiente desde 0?"*
>
> **Respuesta puntual**: casi, pero confundiste qué es el factor. **`A_t` ya estaba en REINFORCE** (es el advantage). Lo que añade PPO es **otro factor multiplicativo encima**: el ratio `ρ_t = π_new / π_old`. Ahora tienes dos factores: `ρ_t · ∇log π · A_t`. El advantage pondera "qué tan buena fue la acción"; el ratio corrige el sesgo de reusar samples viejos. Son factores **independientes**.

> 💬 **Tu preguntaste**: *"el importance sampling vamos con eso un poco más adelante porque no hemos visto PPO aun estamos primero entendiendo todo lo de antes de PPO de reinforce y demás no?"*
>
> **Respuesta puntual**: correcto. Importance sampling es el primer truco de PPO, pero antes de entrar a fondo querías consolidar REINFORCE y los conceptos auxiliares. Este archivo cierra esa etapa. Lo que sigue en el próximo es PPO completo: importance sampling formal + clipping + critic/GAE + loss ensamblada.

---

## 12. Auto-verificación: seis preguntas

Si puedes responderlas sin volver a leer, estás listo para entrar a PPO.

### Pregunta 1
*¿Por qué REINFORCE fue innovador respecto a Q-learning?*

> Porque optimiza directamente la política (no value functions), escala a espacios de acción grandes/continuos (no requiere argmax), maneja políticas estocásticas, y se implementa con backprop estándar gracias al log-derivative trick.

### Pregunta 2
*¿On-policy o off-policy? Define una frase.*

> **On-policy**: los samples deben venir de la política actual. **Off-policy**: los samples pueden venir de cualquier distribución (política vieja, dataset fijo, etc.). REINFORCE es on-policy estricto; DPO es off-policy/offline.

### Pregunta 3
*¿Cuáles son los tres dolores de REINFORCE puro?*

> 1. Estrictamente on-policy → no puedes reusar samples.
> 2. Pasos destructivos → un solo step puede colapsar la política.
> 3. Alta varianza del gradiente → ruido.

### Pregunta 4
*¿De dónde sale el reward en RLHF?*

> De un reward model `r̂` (otro LLM con cabeza escalar), entrenado **una vez** con comparaciones humanas usando pérdida Bradley-Terry, y luego **congelado**. NO es similitud coseno. NO hay ground truth de respuesta correcta. Es un "humano destilado".

### Pregunta 5
*¿El advantage `A_t` es un promedio de rewards?*

> NO. Es una **diferencia respecto al promedio**: `A_t = Q(s,a) − V(s)`. Mide cuánto mejor o peor fue una acción específica comparada con la acción típica de la política en ese estado.

### Pregunta 6
*¿Qué es mini-batch SGD?*

> Entrenar con grupitos de B samples (32-256), promediar los gradientes dentro del batch, y hacer un update por batch. Es el estándar de deep learning moderno. En RLHF, B = 256 prompts típicamente.

---

## 13. Qué viene después

Ya tienes los cimientos para PPO completo. Lo que sigue:

1. **Importance sampling formal** — la matemática del ratio `ρ_t`.
2. **Clipping** del ratio — la solución del Dolor 2 (freno automático).
3. **Critic + GAE** — la solución del Dolor 3 (reducir varianza).
4. **Loss completa de PPO** ensamblada.
5. **RLHF como sistema completo** — cómo se conectan SFT, RM, PPO, KL.
6. **DPO** — la alternativa que se salta PPO entero.
7. **LLM-as-Judge** y **Self-Rewarding**.

Este archivo cierra "los fundamentos previos a PPO". El próximo archivo (`03-ppo.md` o similar) entrará a PPO completo.

---

## Lecturas relacionadas

- [01-mapping-llm-rl.md](01-mapping-llm-rl.md) — mapping LLM ↔ RL.
- [../output/00-fundamentos-rl.md](../output/00-fundamentos-rl.md) §6 — PPO en versión densa.
- [../output/01-rlhf.md](../output/01-rlhf.md) — pipeline RLHF completo.
