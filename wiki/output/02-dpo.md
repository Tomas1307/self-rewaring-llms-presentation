# 02 — DPO: Direct Preference Optimization

> **Prerequisitos**: [00-fundamentos-rl.md](00-fundamentos-rl.md), [01-rlhf.md](01-rlhf.md). Vas a usar policy, reward, KL divergence, modelo Bradley-Terry.

> **Lo que sabrás al final**: por qué DPO existe, la derivación completa paso a paso (sin saltos), por qué se llama "Your Language Model is Secretly a Reward Model", el rol de β y cómo se compara con PPO.

---

## 1. ¿Por qué existe DPO?

RLHF tiene cuatro piezas pesadas:

1. SFT (necesario).
2. Recopilar preferencias humanas.
3. **Entrenar reward model**.
4. **PPO (RL online)**.

Las dos últimas son las que duelen: requieren entrenar y mantener modelos extra, infraestructura RL, hiperparámetros sensibles.

**DPO observa**: si el objetivo de RLHF tiene una **solución cerrada** (analítica), tal vez podamos saltarnos las etapas 3 y 4 derivando una **pérdida supervisada** que optimiza directamente lo mismo. Esta intuición se concreta en el paper de Rafailov et al. 2023.

> 💡 **DPO en una frase**: "Lo que RLHF logra con reward model + PPO, DPO lo logra con una pérdida supervisada de clasificación binaria sobre pares de preferencia."

---

## 2. El plan de la derivación

Tres pasos:

1. Tomar el objetivo de RLHF (max reward con KL).
2. Resolver analíticamente: encontrar la fórmula cerrada de la política óptima `π*`.
3. Invertir la fórmula para expresar `r(x,y)` en términos de `π*` — y sustituir en la pérdida de Bradley-Terry para obtener una pérdida que depende **solo de la política**, no de un reward model entrenado separadamente.

Vamos paso a paso, con todas las cuentas explícitas.

---

## 3. Paso 1 — El objetivo de RLHF

Del cap 01 sabemos que RLHF optimiza:

```
J(π) = E_{x ~ D, y ~ π(·|x)} [ r(x, y) ]  −  β · KL( π(·|x) || π_ref(·|x) )
```

Para simplificar, **fijamos un prompt `x`** y trabajamos con la distribución condicional `π(y | x)`. El problema es:

> 📐 **Problema restringido por KL**:
> ```
> max_π   E_{y ~ π}[ r(x, y) ]  −  β · KL( π(·|x) || π_ref(·|x) )
> ```
> sujeto a que `π(·|x)` sea una distribución de probabilidad válida (i.e., `Σ_y π(y|x) = 1`).

> 💡 **¿Por qué resolver este problema?** No es que vayamos a resolverlo **directamente** (no podemos en práctica — el espacio `y` es infinito). Pero saber **la forma de la solución** nos va a permitir reformular el entrenamiento.

---

## 4. Paso 2 — Solución cerrada

Vamos a resolver el problema usando **multiplicadores de Lagrange** (técnica estándar de optimización con restricciones).

### 4.1. Reescribir KL

Recordando que `KL(π || π_ref) = E_{y ~ π}[ log(π(y|x) / π_ref(y|x)) ]`, el objetivo es:

```
J(π) = E_{y ~ π}[ r(x, y) − β · log(π(y|x) / π_ref(y|x)) ]
     = Σ_y π(y|x) · [ r(x, y) − β · log(π(y|x) / π_ref(y|x)) ]
```

Queremos maximizar esto sobre `π(·|x)` (que es un vector de probabilidades) con la restricción `Σ_y π(y|x) = 1`.

### 4.2. Lagrangiano

```
L(π, λ) = Σ_y π(y|x) · [ r(x, y) − β · log(π(y|x) / π_ref(y|x)) ]
         − λ · ( Σ_y π(y|x) − 1 )
```

### 4.3. Derivar respecto a `π(y|x)`

Para cada `y`:
```
∂L/∂π(y|x) = r(x, y) − β · log(π(y|x) / π_ref(y|x)) − β − λ
```

(Aplicando regla del producto sobre `π · log(π / π_ref)`: el `log(π/π_ref)` se queda y aparece un `+β` extra de la derivada de `π · log π = π · log π − π · log π_ref`.)

Igualando a 0:
```
r(x, y) − β · log(π(y|x) / π_ref(y|x)) − β − λ = 0
```

Despejando `log(π/π_ref)`:
```
log(π(y|x) / π_ref(y|x)) = (r(x, y) − β − λ) / β
                          = r(x, y) / β − 1 − λ/β
```

Tomando exponenciales:
```
π(y|x) / π_ref(y|x) = exp( r(x, y) / β − 1 − λ/β )
                    = exp(−1 − λ/β) · exp(r(x, y) / β)
```

Llamemos `C = exp(−1 − λ/β)` (constante respecto a `y`):
```
π(y|x) = C · π_ref(y|x) · exp( r(x, y) / β )
```

### 4.4. Determinar `C` con la restricción de normalización

`Σ_y π(y|x) = 1` →
```
1 = C · Σ_y π_ref(y|x) · exp(r(x, y) / β)
   = C · Z(x)
```

donde **`Z(x) = Σ_y π_ref(y|x) · exp(r(x, y) / β)`** es la **función de partición** (analogía con física estadística — es lo que normaliza una Boltzmann).

Entonces:
```
C = 1 / Z(x)
```

### 4.5. Resultado: la política óptima

> 📐 **La política óptima del problema RLHF KL-restringido**:
> ```
> π*(y|x) = (1 / Z(x)) · π_ref(y|x) · exp( r(x, y) / β )
> ```
>
> donde `Z(x) = Σ_y π_ref(y|x) · exp(r(x, y) / β)`.

> 💡 **Forma de Boltzmann**: la política óptima tiene **exactamente** la forma de una distribución de Boltzmann con "energía" `−r/β` y "prior" `π_ref`. Esta es la conexión con MaxEnt-IRL que veremos en el cap 05 (IRL).

### 4.6. El problema práctico

Esta fórmula es **teóricamente perfecta pero inútil computacionalmente**:
- `Z(x)` requiere sumar sobre **todas las respuestas posibles**. Para un LLM, el espacio de respuestas es prácticamente infinito.
- No podemos calcular `π*` directamente.

DPO viene de aquí: **no podemos calcular `π*`, pero podemos usar la ecuación para transformar el problema**.

---

## 5. Paso 3 — Invertir y sustituir

Aquí está el truco que hace DPO posible.

### 5.1. Invertir la ecuación

Reorganizamos `π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r/β)` para despejar `r`:

Tomando logaritmos:
```
log π*(y|x) = − log Z(x) + log π_ref(y|x) + r(x, y) / β
```

Despejando `r`:
```
r(x, y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

> 📐 **La reward implícita**:
> ```
> r(x, y) = β · log( π*(y|x) / π_ref(y|x) )  +  β · log Z(x)
> ```
>
> Léelo así: "La reward que el modelo está implícitamente optimizando es proporcional al **log-ratio** entre la política óptima y la referencia, más una constante que depende solo de `x`."

> 💡 **Esta es la frase del paper "Your Language Model is Secretly a Reward Model"**. La política y el reward son **la misma red**: `r(x,y) = β · log(π(y|x) / π_ref(y|x))` (hasta una constante en `x`).

### 5.2. Sustituir en Bradley-Terry

Del cap 01 sabemos que la probabilidad de que un humano prefiera `y_w` sobre `y_l` bajo Bradley-Terry es:
```
P(y_w ≻ y_l | x) = σ( r(x, y_w) − r(x, y_l) )
```

Sustituyendo la `r` implícita:
```
P(y_w ≻ y_l | x) = σ(
    [ β · log(π*(y_w|x) / π_ref(y_w|x)) + β · log Z(x) ]
  − [ β · log(π*(y_l|x) / π_ref(y_l|x)) + β · log Z(x) ]
)
```

> 💡 **¡Magia!** El término `β · log Z(x)` aparece dos veces con signos opuestos. **Se cancela.**

### 5.2.bis. La cancelación visualizada — el truco fundamental

> 📊 **Diagrama del milagro algebraico**:
>
> ```
>   r(y_w)  =  β · log[π*(y_w|x)/π_ref(y_w|x)]  +  β · log Z(x)
>                  ↑ depende de y_w                ↑
>                                                  ⚫ depende SOLO de x
>                                                  ⚫ NO depende de y_w
>
>   r(y_l)  =  β · log[π*(y_l|x)/π_ref(y_l|x)]  +  β · log Z(x)
>                  ↑ depende de y_l                ↑
>                                                  ⚫ depende SOLO de x
>                                                  ⚫ NO depende de y_l
>
>   ────────────────────── RESTAR ──────────────────────
>
>   r(y_w) − r(y_l)  =  β · log[π*(y_w|x)/π_ref(y_w|x)]
>                    −  β · log[π*(y_l|x)/π_ref(y_l|x)]
>                    +  β · log Z(x)  −  β · log Z(x)
>                                       ↑
>                                  ¡SE CANCELA!
>
>   ───────────────────── RESULTADO ─────────────────────
>
>   r(y_w) − r(y_l)  =  β · log[π_θ(y_w|x)/π_ref(y_w|x)]
>                    −  β · log[π_θ(y_l|x)/π_ref(y_l|x)]
>
>   (sustituimos π* por π_θ, la política parametrizada que entrenamos)
> ```

> 🎯 **Por qué importa**: `Z(x) = Σ_y π_ref(y|x)·exp(r(x,y)/β)` es la función de partición. Para un LLM, suma sobre TODAS las respuestas posibles → computacionalmente imposible. La cancelación es **lo que hace DPO factible**.
>
> Si no entendiste por qué `Z(x)` depende solo de `x` y no de `y`, ver [00b-informacion-y-entropia.md](00b-informacion-y-entropia.md) §6 y §7.

Queda:
```
P(y_w ≻ y_l | x) = σ(
    β · log(π*(y_w|x) / π_ref(y_w|x))
  − β · log(π*(y_l|x) / π_ref(y_l|x))
)
```

### 5.3. La pérdida DPO

Si queremos que `π_θ` se comporte como `π*`, maximizamos la log-verosimilitud del dataset bajo Bradley-Terry. Es decir, minimizamos:

> 📐 **Pérdida DPO**:
> ```
> L_DPO(π_θ; π_ref) = − E_{(x, y_w, y_l) ~ D} [
>     log σ(
>         β · log( π_θ(y_w | x) / π_ref(y_w | x) )
>       − β · log( π_θ(y_l | x) / π_ref(y_l | x) )
>     )
> ]
> ```

**Léeel así**: para cada triple `(x, y_w, y_l)`, computamos el log-ratio de `y_w` y de `y_l` bajo la política actual relativa a la referencia, restamos, multiplicamos por `β`, pasamos por sigmoide, tomamos log, negamos. Esa es la pérdida que minimizar.

---

## 6. ¿Qué hace realmente esta pérdida?

Definimos la **reward implícita** del modelo en cualquier momento:
```
r̂_θ(x, y) = β · log( π_θ(y|x) / π_ref(y|x) )
```

La pérdida DPO es:
```
L_DPO = − E[ log σ( r̂_θ(x, y_w) − r̂_θ(x, y_l) ) ]
```

Esto es **exactamente la pérdida cross-entropy de Bradley-Terry**, pero con la reward implícita en lugar de un reward model entrenado.

> 📊 **Visualización del flujo DPO**:
> ```
>   ┌──────────────────────────────────────────────────┐
>   │  Para cada par (x, y_w, y_l) en el dataset:      │
>   │                                                  │
>   │    π_θ(y_w|x) ─┐                                 │
>   │                ├─► r̂_θ(y_w) = β·log(π_θ/π_ref)  │
>   │    π_ref(y_w|x)┘                                 │
>   │                                                  │
>   │    π_θ(y_l|x) ─┐                                 │
>   │                ├─► r̂_θ(y_l) = β·log(π_θ/π_ref)  │
>   │    π_ref(y_l|x)┘                                 │
>   │                                                  │
>   │    diff = r̂_θ(y_w) − r̂_θ(y_l)                  │
>   │                                                  │
>   │    L = − log σ(diff)                             │
>   │                                                  │
>   │    Backprop → solo a través de π_θ              │
>   │    (π_ref está congelado)                       │
>   └──────────────────────────────────────────────────┘
>
>   Si diff > 0: σ(diff) > 0.5, L pequeña → "ya estoy bien"
>   Si diff < 0: σ(diff) < 0.5, L grande → "corrige fuerte"
>   Si diff ≈ 0: σ(diff) = 0.5, L = log 2 ≈ 0.69 → "señal moderada"
> ```

> 💡 **Una forma de verlo**: DPO está entrenando un **reward model implícito** (codificado en los pesos del LLM) **y** una política (los mismos pesos) **al mismo tiempo, en la misma pasada**.

### Intuición del gradiente

```
∇_θ L_DPO ∝  − σ( β · (r̂_θ(y_l) − r̂_θ(y_w)) )
              · β · ( ∇_θ log π_θ(y_w|x) − ∇_θ log π_θ(y_l|x) )
```

Tres casos:

| Situación | `σ(...)` | Magnitud del gradiente | Interpretación |
|---|---|---|---|
| `r̂(y_w) ≫ r̂(y_l)` | ~ 0 | Pequeña | "Ya estoy bien, ajusto poco." |
| `r̂(y_w) ≈ r̂(y_l)` | ~ 0.5 | Moderada | "Necesito separar más." |
| `r̂(y_w) ≪ r̂(y_l)` | ~ 1 | Grande | "Estoy muy mal, corrige fuerte." |

> ⚠️ **Punto crítico para Self-Rewarding**: si tras muchas iteraciones `r̂(y_w) ≈ r̂(y_l)` (porque ambas respuestas vienen del mismo modelo mejorado), el gradiente **tiende a cero**. El loop se queda sin señal. Esto es exactamente la **saturación** que Wang 2025 documenta (cap 12).

---

## 7. Ejemplo numérico

Supón:
- `β = 0.1`.
- `π_ref(y_w|x) = 0.001`, `π_ref(y_l|x) = 0.002`.
- `π_θ(y_w|x) = 0.005`, `π_θ(y_l|x) = 0.001`.

Reward implícita:
```
r̂_θ(y_w) = 0.1 · log(0.005 / 0.001) = 0.1 · log(5) = 0.1 · 1.609 = 0.161
r̂_θ(y_l) = 0.1 · log(0.001 / 0.002) = 0.1 · log(0.5) = 0.1 · (−0.693) = −0.069
```

Diferencia: `0.161 − (−0.069) = 0.230`.

Pérdida:
```
L_DPO = − log σ(0.230) = − log(0.557) = 0.585
```

Pérdida moderada. El modelo prefiere `y_w` sobre `y_l` pero con margen pequeño. El gradiente empujará a aumentar el margen.

---

## 8. Comparación lado-a-lado: RLHF (PPO) vs DPO

| Aspecto | RLHF con PPO | DPO |
|---|---|---|
| Reward model | Separado, entrenado | Implícito en `π_θ` |
| Sampling | On-policy (genera durante entrenamiento) | Offline (pares fijos) |
| Algoritmo | PPO (RL online) | Gradient descent supervisado |
| Modelos en memoria | 4 (policy, ref, RM, critic) | 2 (policy, ref) |
| Estabilidad | Sensible a hiperparams | Robusto |
| Costo | ~3-5× SFT | ~SFT |
| Hiperparámetros sensibles | Muchos | Solo β |
| Iterar | Difícil | Trivial (regenerar pares y entrenar otra vez) |

---

## 9. El rol de β en DPO

`β` controla cuánto puede divergir `π_θ` de `π_ref`. Misma intuición que en RLHF pero con efectos sutiles.

| β | Comportamiento | Cuándo usarlo |
|---|---|---|
| Alto (≥ 0.5) | Conservador. `π_θ ≈ π_ref`. | Si tu `π_ref` ya es muy bueno y solo quieres ajustes finos. |
| Medio (0.1) | Estándar. | **Yuan 2024 y Rafailov 2023 usan este valor.** |
| Bajo (≤ 0.05) | Agresivo. Drift fuerte. | Solo si tu `π_ref` está lejos del óptimo y quieres explorar más. |

> ⚠️ **β bajo aumenta el riesgo de reward hacking**, igual que en RLHF. En DPO se manifiesta como: el modelo asigna log-ratios enormes a respuestas con pequeños tics (uso de viñetas, ciertas palabras) sin mejorar la calidad real.

---

## 10. Limitaciones de DPO

### 10.1. Asume Bradley-Terry

Si las preferencias humanas son **intransitivas** (`A ≻ B`, `B ≻ C`, `C ≻ A`), Bradley-Terry no puede capturarlas. DNO (cap 08) resuelve esto generalizando a equilibrios de Nash.

### 10.2. Sobreajuste rápido

DPO puede **memorizar** los pares de preferencias. Soluciones: IPO (Azar 2023) añade regularización; otros métodos limitan el número de épocas.

### 10.3. Sin exploración

DPO usa pares **fijos**. No genera nuevas respuestas durante entrenamiento. Esto significa:
- No descubre comportamientos no presentes en el dataset.
- Limitado por la calidad y diversidad de los pares.

### 10.4. Dependencia de `π_ref`

Si `π_ref` es pobre, DPO hereda sus problemas. La reward implícita es **relativa** a la referencia.

### 10.5. Colapso bajo preferencias determinísticas

Si todas las preferencias son "casi-determinísticas" (`P(y_w ≻ y_l) ≈ 1` siempre), DPO empuja las probabilidades a extremos y puede colapsar. Azar 2023 (IPO) formaliza este fallo.

---

## 11. Variantes notables

- **IPO** (Azar 2023): añade regularización para prevenir colapso bajo preferencias determinísticas.
- **KTO** (Ethayarajh 2024): no requiere pares — solo etiquetas binarias por respuesta.
- **ORPO** (Hong 2024): combina SFT y alineamiento en una sola pérdida, elimina `π_ref`.
- **DNO** (Rosset 2024, cap 08): equilibrio de Nash, sin asumir Bradley-Terry.
- **SPPO** (Wu 2024, cap 08): juego de suma constante, garantías de convergencia.

---

## 12. ¿Por qué DPO es central para Self-Rewarding?

Self-Rewarding (cap 06) construye el loop:
```
M_t genera respuestas → M_t las juzga → pares (y_w, y_l) → DPO → M_{t+1}
```

DPO encaja porque:

1. **Es offline**: los pares se generan antes del entrenamiento, no durante.
2. **Es estable**: el loop no se rompe por inestabilidad de PPO.
3. **Es iterable**: cada iteración es una pasada de DPO bien definida.
4. **No necesita reward model entrenado**: el juez (LLM-as-a-Judge) ya provee los pares directamente.

Sin DPO, Self-Rewarding sería computacionalmente impráctico. Con DPO, es una iteración limpia.

---

## 13. Frase para fijar

> *"DPO toma el objetivo de RLHF (maximizar reward con KL), demuestra que tiene solución cerrada de forma Boltzmann, invierte esa fórmula para expresar la reward en términos del log-ratio política / referencia, sustituye en la pérdida Bradley-Terry, y ¡el factor de partición se cancela! El resultado es una pérdida supervisada que codifica simultáneamente política y reward. Por eso 'tu LLM es secretamente un reward model'."*

---

## 14. Qué viene

[03-rlaif.md](03-rlaif.md) — RLAIF reemplaza a los humanos por un LLM como anotador. Es el paso previo a Self-Rewarding.

[04-llm-as-judge.md](04-llm-as-judge.md) — el paradigma del juez LLM, que Self-Rewarding usará para generar sus propios pares.

---

## Lecturas relacionadas

- [[concepts/dpo]] — versión densa.
- [[summaries/rafailov-2023-dpo]] — paper original.
- [[summaries/azar-2023-ipo]] — IPO y el colapso de DPO.
- [[concepts/kl-beta]] — análisis del β (WIP).
- [[synthesis/pilar-3-dpo-en-loop]] — cómo DPO encaja en Self-Rewarding.
