# 00b — Teoría de la Información para Self-Rewarding LLMs

> **Prerequisitos**: [00-fundamentos-rl.md](00-fundamentos-rl.md). Útil si conoces probabilidad básica.

> **Lo que sabrás al final**: qué son entropía, cross-entropy y KL divergence (con intuición geométrica, no solo fórmulas), por qué la KL es asimétrica, qué es el principio de máxima entropía, qué es una distribución de Boltzmann, qué es la función de partición `Z`. Todo con diagramas.

> ⚠️ **Por qué este capítulo es CRÍTICO**: sin esto, los caps 02 (derivación DPO) y 05 (IRL ↔ DPO) tienen huecos. La frase "la solución del problema KL-restringido tiene forma de Boltzmann y `log Z(x)` se cancela" es opaca sin estos fundamentos.

---

## 1. El esquema mental: información como sorpresa

> 💡 **Idea central**: en teoría de la información, **"información" = "sorpresa"**. Algo muy esperado contiene poca información. Algo improbable contiene mucha información.

Formalmente, la información de un evento `x` con probabilidad `p(x)` se define como:

```
I(x) = − log p(x)
```

**Por qué este "log"**:
- Si `p(x) = 1` (evento seguro), `I = − log 1 = 0`. Sin sorpresa, sin información.
- Si `p(x) → 0` (evento improbable), `I → ∞`. Sorpresa infinita.
- Si dos eventos independientes ocurren, sus probabilidades se multiplican (`p(x,y) = p(x)·p(y)`) — y como queremos que las informaciones se **sumen** (`I(x,y) = I(x) + I(y)`), el `log` es el operador correcto (`log(p(x)·p(y)) = log p(x) + log p(y)`).

> 📊 **Visualización de `I(x) = −log p(x)`**:
> ```
>    I(x)
>     |
>   4 +    *
>     |     *
>   3 +      *
>     |       *
>   2 +        *
>     |          *
>   1 +             *
>     |                *
>   0 +-------------------*------+--- p(x)
>     0   0.1  0.3  0.5  0.7    1.0
>
>      Eventos raros ↑           ↓ Eventos comunes
>      (mucha info)              (poca info)
> ```

---

## 2. Entropía — información promedio

Si tienes una **distribución de probabilidad** `P(x)`, la entropía es el **valor esperado** de la información:

> 📐 **Entropía de Shannon**:
> ```
> H(P) = − Σ_x P(x) · log P(x)
> ```
> Equivalentemente: `H(P) = E_{x~P}[ −log P(x) ] = E_{x~P}[ I(x) ]`.

### 2.1. Intuición geométrica

**La entropía mide "cuán incierta" es una distribución.**

- **Mínima entropía** (`H = 0`): toda la masa en un punto. Distribución determinista.
- **Máxima entropía**: masa uniformemente distribuida. Máxima incertidumbre.

> 📊 **Tres distribuciones comparadas** (todas sobre 4 outcomes):
> ```
> Distribución A: [1.00, 0.00, 0.00, 0.00]   H = 0           (cierta)
> Distribución B: [0.70, 0.20, 0.05, 0.05]   H ≈ 1.07 bits   (moderada)
> Distribución C: [0.25, 0.25, 0.25, 0.25]   H = 2 bits      (uniforme, máxima)
>
> Visualización (alturas de barras):
>
>   A:  ████             B:  ████              C:  ██  ██  ██  ██
>       ▏▏▏▏                 ██ ▏▏                 ██  ██  ██  ██
>       (todo aquí)          ██ ▏▏▏                 ██  ██  ██  ██
> ```

### 2.2. ¿Por qué importa para LLMs?

- Un LLM con **alta entropía en su distribución de tokens** es "inseguro" — muchos tokens probables.
- Un LLM con **baja entropía** es "confiado" — un token domina.
- Sampling con `temperatura > 1` aumenta entropía; con `temperatura < 1` la disminuye.

---

## 3. Cross-Entropy — distancia entre dos distribuciones

Si tienes dos distribuciones `P` (la "real") y `Q` (la "predicha"), la cross-entropy mide cuánto te "cuesta" usar `Q` para codificar muestras que vienen de `P`:

> 📐 **Cross-Entropy**:
> ```
> H(P, Q) = − Σ_x P(x) · log Q(x)
> ```

### 3.1. ¿Por qué te suena?

**Porque es la pérdida que ya usas todos los días en NLP**. Cuando entrenas un LLM con next-token prediction:

```
loss = − Σ_t  log P_θ(token_t correcto | contexto)
```

Esto es **exactamente cross-entropy** donde `P = δ` (distribución uno-caliente en el token correcto) y `Q = P_θ` (distribución del modelo).

> 💡 **Frase para fijar**: "MLE = minimizar cross-entropy = maximizar likelihood. Son tres formas de decir lo mismo."

### 3.2. Relación con entropía

```
H(P, Q) = H(P) + KL(P || Q)
```

- `H(P)`: entropía intrínseca de `P` (un límite inferior — no puedes hacer mejor que esto).
- `KL(P || Q)`: el "extra" que pagas por usar `Q` en lugar de la `P` real.

> 📊 **Visualización**:
> ```
>     H(P, Q)  = costo total de codificar P usando Q
>        |
>        +----- H(P)         ← parte irreducible (entropía de P)
>        +----- KL(P || Q)   ← parte por usar la Q "equivocada"
> ```

---

## 4. KL Divergence — el corazón de RLHF/DPO

> 📐 **KL Divergence**:
> ```
> KL(P || Q) = Σ_x P(x) · log( P(x) / Q(x) )
> ```
>
> Equivalentemente: `KL(P || Q) = E_{x~P}[ log(P(x)/Q(x)) ] = H(P,Q) − H(P)`.

### 4.1. Tres propiedades fundamentales

**Propiedad 1**: `KL(P || Q) ≥ 0`. Siempre no-negativa.

**Propiedad 2**: `KL(P || Q) = 0` si y solo si `P = Q`. La KL es 0 únicamente cuando las distribuciones son idénticas.

**Propiedad 3** (la clave): **NO es simétrica**. `KL(P || Q) ≠ KL(Q || P)` en general.

> ⚠️ **Esto es lo más confundente**: la "distancia" KL depende del orden. No es una distancia formal — es una **divergencia**.

### 4.2. Forward vs Reverse KL — la asimetría visualizada

Imagina dos distribuciones unidimensionales:
- `P` (la "verdadera"): bimodal — dos picos.
- `Q` (la que intentamos ajustar): unimodal — un pico.

Hay **dos formas** de "ajustar" `Q` a `P`:

> 📊 **Forward KL: `KL(P || Q)` ("mode-covering" / mean-seeking)**:
> ```
>              P (real, bimodal)
>            ___          ___
>           /   \        /   \
>          /     \      /     \
>      ___/       \____/       \___
>
>              Q (ajustada minimizando KL(P||Q))
>                  ________
>                 /        \
>                /          \
>            ___/            \___
>
>      Q se sitúa AL MEDIO de los dos modos de P.
>      Q cubre AMBOS modos con probabilidad razonable.
>      Q NO ignora ninguna región donde P tiene masa.
> ```
>
> **Lógica**: `KL(P || Q) = E_{x~P}[ log(P/Q) ]`. Si en algún punto `P(x) > 0` pero `Q(x) → 0`, el log explota y la pérdida es infinita. Q **no puede permitirse ignorar** regiones donde P tiene masa.

> 📊 **Reverse KL: `KL(Q || P)` ("mode-seeking" / zero-forcing)**:
> ```
>              P (real, bimodal)
>            ___          ___
>           /   \        /   \
>          /     \      /     \
>      ___/       \____/       \___
>
>              Q (ajustada minimizando KL(Q||P))
>           ___
>          /   \
>         /     \
>      ___/      \________________
>
>      Q escoge UNO de los dos modos de P y lo aproxima bien.
>      Q IGNORA el otro modo completamente.
> ```
>
> **Lógica**: `KL(Q || P) = E_{x~Q}[ log(Q/P) ]`. Si `Q(x) > 0` pero `P(x) → 0`, el log explota. Pero Q **puede poner masa cero** en zonas donde Q misma no quiere ir. Así que Q es libre de ignorar modos.

### 4.3. ¿Cuál se usa en RLHF/DPO y por qué?

> 🎯 **RLHF y DPO usan `KL(π_θ || π_ref)`** — la **reverse KL** desde el punto de vista de la política aprendida.

```
KL( π_θ || π_ref ) = E_{y ~ π_θ}[ log( π_θ(y|x) / π_ref(y|x) ) ]
```

**Razones**:

1. **Computacionalmente factible**: la expectación es sobre `π_θ`, que es lo que estamos generando. Podemos muestrear de `π_θ` durante entrenamiento. (Forward KL requeriría muestrear de `π_ref`, lo cual es posible pero menos práctico.)
2. **Comportamiento "mode-seeking"**: queremos que `π_θ` se concentre en respuestas buenas, no que cubra "todo lo que `π_ref` puede generar".
3. **Estabilidad**: las respuestas muestreadas de `π_θ` están automáticamente "en el soporte" donde tiene sentido.

> ⚠️ **Consecuencia**: la regularización KL puede dejar a `π_θ` **abandonando completamente** ciertas regiones que `π_ref` ocupa. Esto es deseado (queremos especialización) pero también es una de las causas del **mode collapse**.

### 4.4. Una analogía para fijar

Imagina dos personas eligiendo restaurantes:

**Forward KL (`P` = real, `Q` = tu modelo)**: Tu modelo `Q` quiere predecir bien las decisiones de `P`. Si `P` come a veces sushi y a veces pizza, `Q` debe asignar probabilidad a **ambos**. Si `Q` ignora "pizza", paga un precio muy alto cuando `P` elige pizza.

**Reverse KL (`Q` = tu modelo, `P` = referencia)**: Tu modelo `Q` quiere parecerse a `P` pero **es libre de especializarse**. `Q` puede decidir "yo solo recomiendo sushi" — mientras eso sea consistente con que `P` también lo recomiende. Si `P` también recomienda pizza, `Q` no es penalizado por no hacerlo.

> 💡 **En RLHF**: `π_ref` (modelo SFT) puede generar muchas respuestas razonables. `π_θ` no necesita cubrirlas todas — basta con que las que **sí** genera estén respaldadas por `π_ref`. Reverse KL captura esto.

---

## 5. El principio de máxima entropía (MaxEnt)

### 5.1. La pregunta motivadora

Supón que sabes algunas cosas sobre una variable aleatoria `X`:
- `E[X] = 5` (sabes el promedio).
- `Var[X] = 4` (sabes la varianza).

**Pregunta**: ¿qué distribución de `X` debes asumir?

Hay **infinitas distribuciones** que satisfacen estas restricciones. ¿Cuál elegir sin meter información que no tienes?

> 💡 **Principio de Máxima Entropía (Jaynes 1957)**: *"Entre todas las distribuciones consistentes con la información disponible, elige la de **máxima entropía**."*

**Justificación filosófica**: la distribución de máxima entropía es la **menos comprometida** — la que asume la menor cantidad de estructura no justificada por los datos.

### 5.2. Resultados clásicos

| Restricción | Distribución MaxEnt |
|---|---|
| Soporte finito sin restricciones | **Uniforme** |
| Soporte `[0, ∞)` con `E[X] = μ` fijo | **Exponencial** con tasa `1/μ` |
| `E[X]` y `Var[X]` fijos | **Gaussiana** |
| Soporte discreto con restricciones lineales en features | **Boltzmann** (ver §6) |

> 💡 **Esta es la conexión con física**: la distribución de Boltzmann en mecánica estadística es **exactamente** la distribución MaxEnt sujeta a restricciones de energía promedio. No es coincidencia.

---

## 6. Distribución de Boltzmann — la forma de la solución RLHF

### 6.1. Forma general

Si tienes una función "energía" `E(x)` sobre el espacio de estados:

> 📐 **Distribución de Boltzmann**:
> ```
> P(x) = (1/Z) · exp( −E(x) / T )
> ```
>
> donde:
> - `T`: "temperatura" (parámetro).
> - `Z = Σ_x exp( −E(x) / T )`: **función de partición** (constante de normalización).

### 6.2. Intuición

- **Estados de baja energía** son **más probables**.
- **Temperatura alta**: todos los estados son similarmente probables (alta entropía).
- **Temperatura baja**: el estado de mínima energía domina (baja entropía).

> 📊 **Cómo `T` modula la distribución sobre 3 estados con energías E = [0, 1, 2]**:
> ```
> T → ∞     P ≈ [0.33, 0.33, 0.33]     uniforme (alta entropía)
> T = 1     P ≈ [0.67, 0.24, 0.09]     decreciente exponencial
> T = 0.1   P ≈ [0.999, 0.0001, 0]     casi delta en mínimo (baja entropía)
>
>   T → ∞       T = 1         T = 0.1
>   ┌─┬─┬─┐    ┌─┬┐┌─┐        ┌─┐
>   │█│█│█│    │█│ │  │        │█│
>   └─┴─┴─┘    └─┴┴┴─┘        └─┴
> ```

### 6.3. Conexión con RLHF — ¡lo que estabas esperando!

En el cap 02 (DPO) vimos que la solución del problema RLHF KL-restringido es:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```

**Esto es una distribución de Boltzmann con prior `π_ref`**:

| Símbolo en Boltzmann | Símbolo en RLHF |
|---|---|
| Estado `x` | Respuesta `y` |
| Energía `E(x)` | `−r(x, y)` (reward negativo) |
| Temperatura `T` | `β` (KL coefficient) |
| Función de partición `Z` | `Z(x)` (depende del prompt) |
| Prior uniforme | Prior `π_ref(y\|x)` (la política SFT) |

> 💡 **Lectura**:
> - Las respuestas con **mayor reward** son **más probables** (como estados de baja energía en física).
> - `β` actúa como temperatura: si `β → ∞`, la política se vuelve uniforme (= `π_ref`). Si `β → 0`, la política colapsa al `y` de máximo reward.
> - El prior `π_ref` "modula" la distribución: respuestas que `π_ref` ya consideraba probables tienen ventaja.

### 6.4. La función de partición — ¿qué problema da?

```
Z(x) = Σ_y  π_ref(y|x) · exp( r(x,y) / β )
```

`Z(x)` es la suma sobre **todas las respuestas posibles**. Para un LLM, el espacio de respuestas es prácticamente infinito (todas las secuencias de tokens).

**Computacionalmente: imposible**.

> 💡 **Esto es lo que motiva DPO** (cap 02): no podemos calcular `π*(y|x)` directamente porque no podemos calcular `Z(x)`. Pero podemos **transformar el problema** para que `Z(x)` se cancele.

---

## 7. ¿Por qué `log Z(x)` se cancela en DPO?

Ahora estás equipado para entender el "truco" del cap 02 §5.

### 7.1. La fórmula invertida (de cap 02)

Despejando `r` de la solución óptima:
```
r(x, y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

### 7.2. Sustitución en Bradley-Terry

Bradley-Terry dice:
```
P(y_w ≻ y_l | x) = σ( r(x, y_w) − r(x, y_l) )
```

Sustituyendo:
```
P(y_w ≻ y_l) = σ(
    [β · log(π*(y_w|x)/π_ref(y_w|x)) + β · log Z(x)]
  − [β · log(π*(y_l|x)/π_ref(y_l|x)) + β · log Z(x)]
)
```

### 7.3. La cancelación — visualizada

> 📊 **La aritmética del milagro**:
>
> ```
> Reward de y_w:    β · log(π*/π_ref)_w  +  β · log Z(x)
>                                            ↑
>                                       este término
>                                            ↓
> Reward de y_l:    β · log(π*/π_ref)_l  +  β · log Z(x)
>                                            ↑
>                                            |
>                            DEPENDE SOLO DE x.
>                            NO DEPENDE DE y_w NI DE y_l.
>                                            |
> Al restar:                                  |
>                                            ↓
> r(y_w) - r(y_l) = β · log(π*/π_ref)_w
>                 - β · log(π*/π_ref)_l
>                 + β · log Z(x)  -  β · log Z(x)
>                                ↑
>                          ¡CANCELACIÓN!
> ```

> 💡 **Por qué pasa esto**: `Z(x)` es la normalización **del prompt** `x`, no de la respuesta. Cuando comparas dos respuestas al **mismo prompt**, ambas heredan la misma `Z(x)`, así que al restar **desaparece**.

> 🎯 **El payoff**: la pérdida DPO depende **solo de las políticas `π_θ` y `π_ref` evaluadas en `y_w` y `y_l`**. No necesitamos calcular `Z(x)`. No necesitamos sumar sobre todas las respuestas posibles. **Es factible computacionalmente.**

---

## 8. MaxEnt-IRL en una página

Ahora puedes entender el cap 05 más profundamente.

### 8.1. El problema

IRL clásico: dado comportamiento experto, encontrar una reward `r` consistente con él. Pero `r` no es única.

MaxEnt-IRL dice: **entre todas las `r` consistentes**, elige la que produce una distribución sobre trayectorias de **máxima entropía**.

### 8.2. La forma de la solución

> 📊 **Resultado de MaxEnt-IRL** (Ziebart 2008):
> ```
> P(τ) = (1/Z) · exp( w^T · φ(τ) )
> ```
> donde:
> - `τ`: trayectoria.
> - `φ(τ)`: features de la trayectoria.
> - `w`: pesos lineales de la reward.
> - `Z`: función de partición.
>
> **Esto es Boltzmann.** Exactamente la misma forma que la solución RLHF KL-restringido.

### 8.3. La equivalencia visualizada

> 📊 **El puente formal**:
> ```
>   MaxEnt-IRL                          RLHF KL-óptimo
>   (Ziebart 2008)                      (Rafailov 2023)
>
>   P(τ) =  (1/Z) ·                     π*(y|x) =  (1/Z(x)) ·
>            exp(w^T φ(τ))                          π_ref(y|x) ·
>                                                   exp(r(x,y) / β)
>
>            ↑                                       ↑
>            └─────── MISMA FORMA EXPONENCIAL ──────┘
>
>           Sin prior                        Con prior π_ref
>           w^T φ(τ) es reward lineal       r(x,y) es reward arbitraria
>           Sin temperatura explícita        β = temperatura
> ```

### 8.4. La consecuencia profunda

**DPO entrena una política `π_θ` tal que su reward implícita `r̂ = β · log(π_θ/π_ref)` es consistente con los pares de preferencia observados.**

Esto es **exactamente IRL** — recuperar una reward desde comportamiento — pero en forma cerrada y supervisada en lugar de iterativa.

---

## 9. Diagrama mental: cómo todo se conecta

> 📊 **Mapa conceptual del wiki desde teoría de la información**:
> ```
>                  ENTROPÍA
>                      |
>                      ↓
>                CROSS-ENTROPY        ←— Loss MLE en NLP
>                /         \
>               ↓           ↓
>           KL DIVERGENCE   PRINCIPIO MAX-ENT
>           (asimétrica)         |
>               |                ↓
>               ↓           BOLTZMANN
>           Regularización  P ∝ exp(-E/T)
>           en RLHF              |
>               |                ↓
>               |          MaxEnt-IRL
>               |          P(τ) ∝ exp(w^T φ)
>               |                |
>               └────────┬───────┘
>                        ↓
>             Problema RLHF KL-restringido
>             π* ∝ π_ref · exp(r/β)
>                        |
>                        ↓
>                     DPO
>             Invertir + cancelar Z(x)
>             L_DPO = clasificación binaria
>                        |
>                        ↓
>              Self-Rewarding
>             (DPO sobre pares generados internamente)
>                        |
>                        ↓
>                  Tres pilares:
>           IRL implícito (la reward existe)
>           Cooperative MARL (2 roles)
>           DPO en loop (offline iterable)
> ```

---

## 10. Lo que recuerdas si solo recuerdas una cosa

> 🎯 **Frase central**:
> 
> *"La solución del problema RLHF KL-restringido es una distribución de Boltzmann: respuestas con más reward son más probables, ponderadas por `π_ref` y reguladas por la temperatura `β`. Esta forma exponencial es idéntica a MaxEnt-IRL. DPO explota esto: invierte la ecuación para expresar reward en términos de política y referencia, y al sustituir en Bradley-Terry el término `log Z(x)` (la normalización imposible) se cancela porque es constante en el prompt. El resultado es una pérdida supervisada que codifica simultáneamente política y reward."*

---

## 11. Ejercicio mental para verificar comprensión

Sin mirar las fórmulas, intenta responder:

1. **¿Por qué KL es asimétrica?**
   - Porque la integral `Σ P · log(P/Q)` da peso a regiones donde `P` tiene masa. Cambiar el orden cambia las regiones que importan.

2. **¿Por qué la solución del problema KL-restringido tiene forma exponencial?**
   - Porque el principio MaxEnt da Boltzmann como solución a "max entropía sujeto a restricciones promedio". El problema RLHF KL-restringido es matemáticamente equivalente.

3. **¿Por qué `log Z(x)` se cancela en DPO?**
   - Porque `Z(x)` depende solo del prompt `x`, no de la respuesta `y`. Al comparar dos respuestas al mismo prompt, ambas comparten el mismo `Z(x)`, que se va al restar.

4. **¿Por qué DPO es "IRL implícito"?**
   - Porque la política óptima de RLHF (forma Boltzmann) coincide con la solución MaxEnt-IRL con prior. Entrenar `π` con DPO = recuperar la `r` consistente con los pares de preferencia. Eso es IRL.

5. **¿Por qué Self-Rewarding es "IRL del juez interno"?**
   - Porque DPO recupera la reward consistente con los pares observados. En Self-Rewarding, los pares vienen del juez = mismo modelo. Por tanto la reward recuperada es la reward del juez interno.

---

## 12. Lecturas siguientes

Ahora estás equipado para los caps que más se apoyan en esto:

- **Cap 02 (DPO)** — la cancelación de `Z(x)` ahora debe ser cristalina.
- **Cap 05 (IRL)** — el puente MaxEnt-IRL ↔ DPO ahora es transparente.
- **Cap 06 (Self-Rewarding)** — las "tres lecturas teóricas" se sostienen.

---

## 13. Glosario

| Término | Una línea |
|---|---|
| Información `I(x)` | `−log p(x)`. Sorpresa de un evento. |
| Entropía `H(P)` | `E[−log P]`. Incertidumbre promedio. |
| Cross-entropy `H(P, Q)` | `E_P[−log Q]`. Costo de codificar P con Q. **= Loss MLE.** |
| KL divergence `KL(P\|\|Q)` | `H(P,Q) − H(P)`. Costo extra de usar Q en lugar de P. Asimétrica. |
| Forward KL `KL(P\|\|Q)` | "Q debe cubrir todos los modos de P" (mean-seeking). |
| Reverse KL `KL(Q\|\|P)` | "Q puede especializarse en un modo de P" (mode-seeking). **La usada en RLHF/DPO.** |
| Principio MaxEnt | Entre distribuciones consistentes con info, elige la de máxima entropía. |
| Boltzmann/Gibbs | `P(x) ∝ exp(−E(x)/T)`. Forma MaxEnt con restricciones de energía. |
| Función de partición `Z` | Normalización `Σ exp(−E/T)`. En LLMs, intratable directamente. |

---

## Lecturas relacionadas

- [00-fundamentos-rl.md](00-fundamentos-rl.md) — RL básico.
- [02-dpo.md](02-dpo.md) — derivación DPO; ahora será transparente.
- [05-irl.md](05-irl.md) — IRL y el puente con DPO.
