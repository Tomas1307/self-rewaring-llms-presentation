# 01 — RLHF: Reinforcement Learning from Human Feedback

> **Prerequisitos**: [00-fundamentos-rl.md](00-fundamentos-rl.md). Vas a usar policy, reward, KL divergence y PPO.

> **Lo que sabrás al final**: por qué RLHF existe, cómo funciona su pipeline de tres etapas (SFT → RM → PPO), de dónde viene la fórmula del objetivo, qué papel juega β, y por qué es costoso.

---

## 1. ¿Por qué existe RLHF?

Cuando entrenas un LLM solo con next-token prediction sobre internet, el modelo es **completion-machine**: dado un texto, produce el texto que probablemente lo siga. Esto no es lo mismo que ser **útil** o **alineado** a lo que un humano quiere.

Ejemplo concreto:

```
Prompt: "Explícame cómo funciona un transformer."

Modelo base (solo pretraining): "Explícame cómo funciona un transformer.
Explícame cómo funciona una red neuronal recurrente. Explícame cómo
funciona una red neuronal convolucional. ..."

(El modelo ve el prompt y predice que probablemente sigue otra lista
de preguntas, porque eso es lo que vio en webs de FAQs.)
```

SFT (supervised fine-tuning) resuelve **parte** del problema:

```
Modelo SFT: "Un transformer es una arquitectura de red neuronal que..."
```

Pero SFT tiene dos limitaciones:

1. **Necesita demostraciones de alta calidad**. Carísimas de producir a escala.
2. **No captura matices de "calidad"**. Si una respuesta es "correcta pero pedante" vs "correcta y clara", SFT no sabe cuál preferir, y a menudo el dataset humano contiene ambas.

**RLHF resuelve esto** usando **comparaciones** (más fáciles de obtener que demostraciones) y **optimización de calidad** (no solo imitación).

> 💡 **La pregunta clave que RLHF responde**: ¿cómo aprende un modelo a preferir una respuesta sobre otra cuando ambas son "técnicamente correctas"?

---

## 2. La idea central de RLHF en una frase

> 📐 **RLHF en una línea**:
> "Aprende una **reward function** a partir de comparaciones humanas, y luego entrena el LLM con RL **maximizando esa reward**, regularizando con KL para no alejarse del SFT."

Las palabras clave:
- **Reward function**: una red neuronal `r̂(x, y)` que asigna un score escalar a la respuesta `y` al prompt `x`.
- **Comparaciones humanas**: humanos miran dos respuestas y dicen "esta es mejor".
- **RL maximizando reward**: PPO.
- **KL para no alejarse del SFT**: el término de regularización que vimos en cap 00.

---

## 3. El pipeline de 3 etapas

```
[Modelo pre-entrenado]
       ↓
   SFT (etapa 1)
       ↓
[Modelo π_SFT]
       ↓
   Recopilar preferencias humanas
       ↓
   Entrenar Reward Model (etapa 2)
       ↓
[Reward model r̂ congelado]
       ↓
   PPO con r̂ y regularización KL (etapa 3)
       ↓
[Modelo π_RLHF — alineado]
```

### Etapa 1 — SFT (Supervised Fine-Tuning)

**Esto ya lo conoces.** Tomas el modelo pre-entrenado, le das un dataset de pares `(prompt, respuesta_humana_de_calidad)` y haces fine-tuning con cross-entropy.

**Output**: `π_SFT`, un modelo que sigue instrucciones razonablemente.

> 💡 **Crítico**: `π_SFT` va a actuar como **política de referencia** `π_ref` en las etapas siguientes. Es el "ancla" que evita que el modelo derive demasiado durante RL.

### Etapa 2 — Reward Model

Aquí está la magia.

**Recopilar datos de preferencia**:

```
Para cada prompt x:
  1. Genera K respuestas y_1, ..., y_K con π_SFT (sampling con temperatura).
  2. Muestra los K outputs a un humano.
  3. Humano los rankea de mejor a peor (o compara pares).
```

Esto te da un dataset de tripletas `(x, y_w, y_l)` donde `y_w` es la respuesta preferida ("winner") y `y_l` la no preferida ("loser").

**Modelo Bradley-Terry**:

> 📐 **Modelo Bradley-Terry** (originalmente de estadística clásica): si dos opciones tienen scores `r(y_w)` y `r(y_l)`, la probabilidad de que un humano prefiera `y_w` es:
> ```
> P(y_w ≻ y_l | x) = σ( r(x, y_w) − r(x, y_l) )
> ```
> donde `σ` es la sigmoide.
>
> **Lectura**: si `r(y_w) ≫ r(y_l)`, la sigmoide es ~1 (humano prefiere `y_w` casi seguro). Si `r(y_w) ≈ r(y_l)`, la sigmoide es ~0.5 (humano elige al azar). Si `r(y_w) ≪ r(y_l)`, la sigmoide es ~0 (humano elige `y_l`).

**Entrenamiento del Reward Model**:

Quieres que `r̂_φ(x, y)` asigne scores tales que las preferencias del dataset sean consistentes con Bradley-Terry. Pérdida:

> 📐 **Pérdida del Reward Model**:
> ```
> L_RM(φ) = − E_{(x, y_w, y_l) ~ D} [ log σ( r̂_φ(x, y_w) − r̂_φ(x, y_l) ) ]
> ```
>
> Esto es **cross-entropy binaria** sobre el modelo Bradley-Terry. Para cada par de preferencia, maximizas la probabilidad de que el RM "elija correctamente".

**Arquitectura del RM**: típicamente es un LLM (frecuentemente del mismo tamaño que `π_SFT`) con una **cabeza escalar** que reemplaza la cabeza de language modeling. Toma `(x, y)` y emite un número.

**Output**: `r̂` congelado. Listo para usar.

> 💡 **Punto clave**: a partir de aquí, ya no necesitas humanos. El RM **sustituye** al humano durante el resto del entrenamiento.

### Etapa 3 — PPO con KL regularization

Ahora aplicas PPO (cap 00) sobre la política `π_θ` (inicializada desde `π_SFT`) usando `r̂` como reward.

> 📐 **Objetivo de RLHF**:
> ```
> max_θ  E_{x ~ D, y ~ π_θ(·|x)} [ r̂(x, y) ]  −  β · E_x [ KL( π_θ(·|x) || π_ref(·|x) ) ]
> ```
>
> Léelo así:
> - **Primer término**: en expectativa sobre prompts del dataset y respuestas generadas por la política actual, maximiza el reward asignado por el RM.
> - **Segundo término**: penaliza si la política se aleja del SFT.
> - `β`: cuánto pesa el segundo término. Típicamente 0.01 - 0.1.

**Algoritmo (versión simplificada)**:

```
π_θ ← π_SFT
π_ref ← π_SFT (congelado)
V_φ ← inicializar critic

repetir hasta convergencia:
    1. Generar batch de respuestas con π_θ_old.
    2. Para cada (x, y): calcular r̂(x, y).
    3. Calcular reward "modificado" por KL:
         r_t(x, y) = r̂(x, y) − β · log( π_θ(y|x) / π_ref(y|x) )
    4. Calcular advantages A_t con V_φ.
    5. Update π_θ con loss PPO clippado.
    6. Update V_φ con MSE.
    7. π_θ_old ← π_θ.
```

**Output**: `π_RLHF`, el modelo alineado.

---

## 4. Los cuatro modelos en memoria

Aquí está el costo computacional real:

| # | Modelo | Tamaño | Estado | ¿Por qué? |
|---|---|---|---|---|
| 1 | `π_θ` (política) | LLM completo | Entrenable | Es lo que estamos entrenando. |
| 2 | `π_ref` (referencia) | LLM completo | Congelado | Necesario para calcular KL. |
| 3 | `r̂` (reward model) | LLM completo | Congelado | Asigna rewards. |
| 4 | `V_φ` (critic) | LLM completo | Entrenable | Estima advantages para PPO. |

**Total: 4 LLMs en GPU simultáneamente.** Si tu LLM es de 70B parámetros, son 280B en memoria activa solo para entrenamiento. Por eso PPO es caro.

> ⚠️ **Esto es lo que motiva DPO** (cap 02): si pudieras eliminar el reward model y el critic, te quedarías con 2 modelos en lugar de 4 → mitad de memoria.

---

## 5. Ejemplo concreto: la pérdida del Reward Model

Supón un dataset con un solo par:

```
Prompt: "¿Qué es un transformer?"
y_w (winner): "Un transformer es una arquitectura..."
y_l (loser):  "No sé."
```

El RM asigna:
- `r̂(x, y_w) = 2.1`
- `r̂(x, y_l) = -0.5`

Pérdida:
```
L = − log σ(2.1 − (−0.5)) = − log σ(2.6) = − log(0.931) = 0.071
```

Pérdida pequeña → el modelo "tiene razón" (`y_w` mucho más alto que `y_l`). Si fuera al revés (`r̂(y_l) > r̂(y_w)`), la pérdida sería grande y el gradiente empujaría fuerte para corregir.

---

## 6. Ejemplo concreto: el objetivo de PPO con KL

Supón que `π_θ` genera la respuesta `y` con probabilidad `0.05` y `π_ref` con `0.10`. La razón:
```
π_θ(y|x) / π_ref(y|x) = 0.05 / 0.10 = 0.5
log(0.5) = −0.693
```

El término KL para este token es `β · (−0.693) = −0.0693` (si `β = 0.1`).

Si `r̂(x, y) = 2.0`, el reward modificado es:
```
r_mod = 2.0 − (−0.0693) = 2.069
```

Como la nueva política asigna **menor** probabilidad a `y` que la referencia, el término KL **aumenta** el reward — está diciendo "está bien que te alejes hacia respuestas con reward alto, pero solo un poco".

Si en cambio `π_θ(y|x) = 0.50` (mucho mayor que `π_ref`):
```
log(0.50/0.10) = log(5) = 1.609
r_mod = 2.0 − 0.1 · 1.609 = 1.84
```

Aquí el KL **penaliza** porque la nueva política está empujando esta respuesta mucho más fuerte que la referencia.

> 💡 **Intuición geométrica**: KL impone un "presupuesto de cambio". Puedes hacer respuestas con reward alto más probables, pero no infinitamente. Si una respuesta tiene reward enorme, el modelo aceptará pagar más KL para subirla. Si tiene reward modesto, no.

---

## 7. La fórmula del objetivo en su forma "completa"

Vimos:
```
max_θ  E[r̂(x, y)] − β · KL(π_θ || π_ref)
```

Hay una equivalencia útil que vas a ver en DPO. Recordando que:
```
KL(π_θ || π_ref) = E_{y ~ π_θ}[ log(π_θ(y|x) / π_ref(y|x)) ]
```

Podemos escribir:
```
max_θ  E_{y ~ π_θ}[ r̂(x, y) − β · log(π_θ(y|x) / π_ref(y|x)) ]
```

Esto es **una sola expectativa sobre la política**. El término `−β · log(π_θ/π_ref)` se suma al reward por token. Es exactamente la forma en que PPO implementa la KL en código.

---

## 8. Limitaciones de RLHF

### 8.1. Costo de anotación humana

Anotar 50.000 comparaciones de calidad alta cuesta cientos de miles de dólares. Si quieres iterar (re-anotar tras cada mejora del modelo), multiplica.

### 8.2. Cuello de botella humano

Si tu modelo se vuelve mejor que un humano promedio en una tarea, los humanos ya no pueden anotar bien. Esto es **el argumento filosófico** de Self-Rewarding: necesitas feedback "superhumano" para entrenar modelos superhumanos.

### 8.3. Reward model congelado

Una vez entrenado, `r̂` no aprende más. Pero la política sí va evolucionando. Resultado:

```
π_θ se mueve hacia respuestas de alto reward según r̂.
   ↓
Respuestas se distancian de la distribución que r̂ vio durante entrenamiento.
   ↓
r̂ deja de ser confiable en la nueva región del espacio.
   ↓
Reward hacking: π_θ encuentra "trucos" que dan reward alto pero respuestas malas.
```

Esto se llama **distribution shift** y es un problema crónico.

### 8.4. Complejidad de PPO

- 4 modelos en memoria.
- Hiperparámetros: β, ε, c_value, c_entropy, learning rate, batch size, número de épocas por rollout, GAE λ, número de tokens por rollout...
- Inestable: una mala combinación y el entrenamiento diverge.

### 8.5. Sesgos humanos

Anotadores son personas. Tienen sesgos:
- **Length bias**: prefieren respuestas más largas (sienten "esfuerzo").
- **Cultural bias**: lo que es "respetuoso" varía.
- **Formatting bias**: prefieren listas, negritas, estructura — independiente del contenido.

Estos sesgos se incrustan en `r̂` y luego en `π_RLHF`. Es difícil detectarlos a posteriori.

---

## 9. Hiperparámetros típicos

Tomados de InstructGPT (Ouyang 2022) y Llama 2 (Touvron 2023):

| Parámetro | Valor típico |
|---|---|
| `β` (KL coeff) | 0.01 – 0.1 |
| `ε` (PPO clip) | 0.2 |
| Learning rate política | 1e-6 – 5e-6 |
| Learning rate critic | 1e-5 |
| Batch size (prompts) | 16 – 64 |
| Rollouts por prompt | 4 – 16 |
| Épocas PPO por rollout | 4 |
| GAE λ | 0.95 |

> ⚠️ **Anécdota del campo**: ajustar PPO para LLMs es **arte**. Equipos con experiencia (OpenAI, Anthropic, Meta) lo saben hacer; reproducirlo desde cero es notoriamente difícil. Esta es una razón pragmática para usar DPO.

---

## 10. RLHF como punto de partida narrativo

En el contexto de esta presentación, RLHF es el **paradigma original** del que se aleja la evolución:

```
RLHF (humanos)
   ↓ reemplazar humanos por LLM externo
RLAIF (cap 03)
   ↓ usar el mismo modelo iterativamente
Self-Rewarding (cap 06)
   ↓ añadir meta-nivel
Meta-Rewarding (cap 07)
```

En paralelo, la rama algorítmica:

```
PPO (4 modelos)
   ↓ eliminar critic con baseline grupal
GRPO (3 modelos, cap 13)
   ↓ eliminar RM y RL online
DPO (2 modelos, cap 02)
```

---

## 11. Frase para fijar

> *"RLHF usa preferencias humanas para entrenar un reward model, y luego usa ese reward model para entrenar el LLM con PPO + regularización KL. El reward model resuelve el problema de 'no podemos cuantificar calidad directamente'. PPO + KL resuelve el problema de 'no nos podemos alejar demasiado del SFT base'. Sus limitaciones — costo, cuello de botella humano, RM congelado, complejidad — motivan todas las alternativas que veremos."*

---

## 12. Qué viene

[02-dpo.md](02-dpo.md) — DPO toma la fórmula del objetivo de RLHF y demuestra que tiene **solución cerrada**, permitiendo entrenar sin reward model y sin RL online. Es el corazón técnico de Self-Rewarding.

---

## Lecturas relacionadas

- [[concepts/rlhf]] — versión densa.
- [[summaries/ouyang-2022-instructgpt]] — paper canónico de RLHF a escala.
- [[summaries/christiano-2017-deep-rl-preferences]] — primera aplicación.
- [[summaries/schulman-2017-ppo]] — PPO original.
- [00-fundamentos-rl.md](00-fundamentos-rl.md) — repaso.
