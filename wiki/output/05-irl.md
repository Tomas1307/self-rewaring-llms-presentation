# 05 — Inverse Reinforcement Learning (IRL)

> **Prerequisitos**: [00-fundamentos-rl.md](00-fundamentos-rl.md), [02-dpo.md](02-dpo.md).

> **Lo que sabrás al final**: qué es IRL formalmente, los tres frameworks principales (Ng-Russell, MaxEnt, preference-based), por qué DPO "es" IRL implícito, y cómo este marco soporta el Pilar 1 de la presentación.

---

## 1. ¿Qué es IRL y por qué importa?

En **RL normal (forward)**:
```
Conocido: reward function r(s, a).
Aprendido: política óptima π* que maximiza E[Σ r].
```

En **Inverse RL (IRL)**:
```
Conocido: comportamiento de un experto (trayectorias τ_1, ..., τ_N).
Aprendido: reward function r tal que ese comportamiento es óptimo respecto a r.
```

> 💡 **La pregunta de IRL**: "Si veo a alguien actuar bien, ¿qué reward está optimizando implícitamente?"

### ¿Por qué nos importa para Self-Rewarding?

Self-Rewarding **infiere implícitamente qué es una "buena respuesta"** a través de la dinámica del juez interno + DPO. Si formalmente eso es IRL, entonces:

1. Self-Rewarding hereda **el marco teórico de IRL** (40 años de literatura).
2. La afirmación "el modelo aprende su propio reward" se vuelve **rigurosa**, no retórica.
3. Podemos identificar problemas conocidos en IRL que apliquen aquí.

Este es **el Pilar 1** de la presentación.

---

## 2. IRL clásico — Ng & Russell (2000)

### 2.1. Formulación

Dado:
- Un MDP `(S, A, P, γ)` (sin reward).
- Trayectorias de un experto `D = {τ_1, ..., τ_N}`.

Encontrar `r` tal que las trayectorias del experto sean óptimas:

> 📐 **IRL clásico**:
> ```
> Encontrar r tal que:
>   π^expert ∈ argmax_π  E_{τ ~ π}[ Σ_t γ^t · r(s_t, a_t) ]
> ```

Es decir: la `r` que recuperes debe ser consistente con que el experto **realmente sea óptimo** según ella.

### 2.2. El problema fundamental

**`r` no es única**. Hay muchas reward functions distintas para las cuales una misma política es óptima:
- `r(s, a) = 0` para todo `(s, a)` hace cualquier política óptima.
- Añadir una constante a `r` no cambia la política óptima.
- Cualquier transformación monotónica preserva el orden.

Esto se llama **degeneración de IRL** y motivó muchas variantes posteriores.

### 2.3. Tres formas de "regularizar"

Para escoger una `r` específica entre todas las consistentes:

1. **Margen máximo**: elegir `r` que haga la política experta **claramente mejor** que cualquier otra.
2. **MaxEnt**: elegir `r` que produzca una distribución sobre trayectorias de máxima entropía consistente con las features del experto.
3. **Sparse rewards**: penalizar `r` no cero (escoger la `r` más simple).

La opción 2 (MaxEnt) es la que conecta con LLMs.

---

## 3. MaxEnt-IRL (Ziebart 2008) — el puente con DPO

### 3.1. La idea

Entre todas las distribuciones sobre trayectorias consistentes con las **expectativas de features** del experto, elige la de **máxima entropía**.

> 📐 **Definición de MaxEnt-IRL**:
> ```
> max_P  H(P)
> sujeto a:
>   E_{τ ~ P}[ φ(τ) ] = E_{τ ~ expert}[ φ(τ) ]   (matching de features)
>   Σ_τ P(τ) = 1                                   (normalización)
> ```
> donde `H(P) = − Σ P(τ) log P(τ)` es la entropía.

**Léelo así**: "encuentra la distribución de trayectorias que matchea al experto en las features promedio, pero entre todas las que cumplen eso, la más 'incierta' (menos sesgada hacia trayectorias específicas)".

### 3.2. La solución cerrada

Por argumentos de optimización (multiplicadores de Lagrange):

> 📐 **Distribución MaxEnt**:
> ```
> P(τ) = (1 / Z) · exp( w^T · φ(τ) )
> ```
> donde `w` son los multiplicadores y `Z` la función de partición.

**Esta es la distribución de Boltzmann**. Cada trayectoria tiene "energía" `−w^T φ(τ)` y la distribución es la que maximiza entropía sujeta a las restricciones.

> 💡 **Conexión clave**: si interpretamos `w^T φ(τ)` como la **reward acumulada** de la trayectoria, entonces:
> ```
> P(τ) ∝ exp( reward total de τ )
> ```
> Las trayectorias **mejores** (más reward) son **más probables**. La constante `w^T` actúa como reward lineal en features.

### 3.3. La conexión con LLMs (la clave para DPO)

Recordemos del cap 02 (DPO §4.5) que la política óptima del problema RLHF KL-restringido es:
```
π*(y|x) = (1 / Z(x)) · π_ref(y|x) · exp( r(x, y) / β )
```

> 📊 **El puente formal MaxEnt-IRL ↔ DPO (lado a lado)**:
>
> ```
>   ┌─────────────────────────┐        ┌─────────────────────────┐
>   │   MaxEnt-IRL            │        │   RLHF KL-óptimo        │
>   │   (Ziebart 2008)        │        │   (Rafailov 2023)       │
>   ├─────────────────────────┤        ├─────────────────────────┤
>   │                         │        │                         │
>   │            1            │        │              1          │
>   │  P(τ) =  ─── · exp(...) │   ═══  │  π*(y|x) = ─── · ...    │
>   │           Z             │        │            Z(x)         │
>   │                         │        │                         │
>   │  ...exp(w^T · φ(τ))     │        │  ...π_ref(y|x)·         │
>   │                         │        │    exp(r(x,y) / β)      │
>   │                         │        │                         │
>   ├─────────────────────────┤        ├─────────────────────────┤
>   │  w^T φ(τ)               │   ═══  │  r(x, y) / β            │
>   │  ↑                      │        │  ↑                      │
>   │  Reward acumulada       │        │  Reward / temperatura   │
>   │  lineal en features     │        │                         │
>   │                         │        │                         │
>   │  Sin prior              │   ≠    │  Con prior π_ref        │
>   │  (entropía absoluta)    │        │  (entropía relativa     │
>   │                         │        │   a π_ref)              │
>   │                         │        │                         │
>   │  Z = Σ_τ exp(w^T φ(τ))  │   ═══  │  Z(x) = Σ_y π_ref(y|x) ·│
>   │                         │        │    exp(r(x,y)/β)        │
>   └─────────────────────────┘        └─────────────────────────┘
>
>                           ═══
>              MISMA FORMA EXPONENCIAL DE BOLTZMANN
>                           ═══
> ```

| MaxEnt-IRL (Ziebart) | RLHF KL-óptimo (Rafailov) |
|---|---|
| `P(τ) = (1/Z) · exp(w^T φ(τ))` | `π*(y\|x) = (1/Z(x)) · π_ref(y\|x) · exp(r(x,y)/β)` |
| Sin prior | Con prior `π_ref` |
| `w^T φ` es reward lineal | `r(x,y)/β` es reward sobre tokens |

**La forma es idéntica.** MaxEnt-IRL con prior `π_ref` = solución RLHF KL-restringida.

> 💡 **Frase consolidada por la literatura** (Lambert RLHF Book, cap 5):
> *"Reward modeling in RLHF can be viewed as a kind of inverse RL."*

> 📊 **Por qué esta equivalencia existe** (en una imagen):
> ```
>   ┌──────────────────────────────────────────────────────┐
>   │  Problema MaxEnt:                                    │
>   │    max  H(P)                                         │
>   │    s.t. E_P[features] = expert_features              │
>   │                                                      │
>   │           ⇓  (Lagrange + KKT)                        │
>   │                                                      │
>   │    P(τ) ∝ exp(w^T · φ(τ))    ←─ Boltzmann            │
>   └──────────────────────────────────────────────────────┘
>
>   ┌──────────────────────────────────────────────────────┐
>   │  Problema RLHF KL-restringido:                       │
>   │    max  E_π[r(x,y)] − β·KL(π || π_ref)               │
>   │                                                      │
>   │           ⇓  (Lagrange + KKT, ver cap 02 §4)         │
>   │                                                      │
>   │    π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β) ← Boltzmann │
>   └──────────────────────────────────────────────────────┘
>
>   Ambos problemas son CASOS DEL MISMO TEMPLATE:
>   "Maximizar entropía sujeto a restricciones lineales" →
>   "Solución exponencial (Boltzmann/Gibbs)".
>
>   Por eso la forma es idéntica. No es coincidencia.
> ```

---

## 4. La cadena lógica completa: Self-Rewarding como IRL implícito

Aquí está el **argumento del Pilar 1** paso a paso:

### Paso 1 — RLHF KL-restringido tiene solución cerrada de forma Boltzmann

Demostrado en cap 02:
```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```

### Paso 2 — Esta solución es estructuralmente MaxEnt-IRL con prior

Ver §3.3 arriba. La equivalencia formal es exacta.

### Paso 3 — DPO invierte la ecuación

Cap 02 §5: despejando `r`:
```
r(x,y) = β · log( π(y|x) / π_ref(y|x) ) + β · log Z(x)
```

DPO no entrena `r` explícitamente; entrena `π` tal que su **reward implícita** explique los pares de preferencia.

> 💡 **Esto es IRL**: dado un comportamiento (los pares de preferencia), recupera la reward implícita (`β · log(π/π_ref)`) que lo justifica.

### Paso 4 — Self-Rewarding cierra el círculo

En RLHF clásico, los pares vienen de **humanos** → DPO recupera `r` que explica preferencias humanas.

En Self-Rewarding (cap 06), los pares vienen del **mismo modelo** actuando como juez → DPO recupera `r` que explica preferencias del juez interno.

> 📐 **Conclusión formal del Pilar 1**:
>
> *"Self-Rewarding no es IRL en el sentido estricto de Ng-Russell (no hay experto externo), pero ejecuta una **recuperación implícita de recompensa estructuralmente equivalente al óptimo MaxEnt-IRL con prior `π_ref`**, donde el rol de experto lo cumple el propio modelo en su modo juez."*

---

## 5. Validación empírica: Joselowitz et al. 2025

¿La reward implícita "existe" como objeto real, o es solo retórica matemática?

**Joselowitz et al. (2025, arXiv:2410.12491)** lo prueba empíricamente:

1. Toman un LLM ya alineado con RLHF (e.g., Llama-2-Chat).
2. Aplican IRL (versión MaxEnt) al comportamiento del modelo.
3. **Recuperan una `r̂`** consistente con esa política.
4. Evalúan: `r̂` predice preferencias humanas held-out con **hasta 85% de precisión**.

> 💡 **Lectura**: la dualidad MaxEnt-IRL ↔ DPO no es solo elegancia matemática. La reward implícita **existe**, **es recuperable**, **es semánticamente significativa**.

Esto valida el Pilar 1 con datos.

---

## 6. Preference-based IRL — el subcampo más relevante

Una rama de IRL trabaja directamente con **preferencias** en lugar de trayectorias.

### Sadigh et al. (2017) — Active Preference Learning

Aplicado a robots: en vez de demostraciones, el robot pregunta al humano "¿prefieres esta trayectoria o esta otra?". Cada par de preferencias se usa para aprender la reward.

Esta es **conceptualmente** la misma estructura que RLHF, anterior a LLMs.

### Wirth et al. (2017) — Survey de Preference-Based RL

JMLR. Survey clásica que enmarca PbRL (preference-based RL) como puente entre IRL clásico y la línea moderna de aprendizaje desde preferencias.

> 💡 **Posición histórica**: RLHF (Christiano 2017) y luego DPO (2023) son **PbRL aplicado a LLMs**. Toda la maquinaria existía antes; lo nuevo es la escala y el dominio.

### Hejna & Sadigh (2023) — Inverse Preference Learning

Formaliza "PbRL sin reward function explícito". Da garantías y muestra equivalencias con DPO en cierto régimen.

---

## 7. Caveats honestos para la defensa

Cosas a admitir en la presentación:

### 7.1. No es IRL estricto

El "experto" en Self-Rewarding **no es externo**. Esto rompe la justificación clásica de IRL (que la política observada sea efectivamente óptima respecto a alguna reward verdadera). Tienes que decir "IRL **funcional** o **implícito**, no estricto".

### 7.2. El experto interno no es óptimo

`M_t` mejora con `t`, pero arrastra sesgos del modelo base. Su política `π_J` (modo juez) no es necesariamente la "verdadera política óptima a recuperar".

### 7.3. El experto interno se mueve

Cada iteración cambia el experto. Esto **rompe la suposición de estacionalidad** de IRL clásico. La reward recuperada en iteración `t` puede no ser la misma que en `t+1`.

### 7.4. El sesgo de auto-preferencia distorsiona la `r` recuperada

Como vimos en cap 04, el juez prefiere outputs estilísticamente familiares. La `r` que DPO recupera puede ser **`r̂ = calidad real + sesgo estilístico`**. Distinguir las dos componentes es un problema abierto.

---

## 8. ¿Por qué importa esta conexión para la presentación?

Sin el marco de IRL, Self-Rewarding parece **un truco de ingeniería**: "el modelo se evalúa, hace DPO, se mejora". Con IRL:

1. Se vuelve **una instancia de un problema clásico** (recuperación de reward desde comportamiento).
2. Adquiere **soporte teórico** (Ziebart 2008, Rafailov 2023, Joselowitz 2025).
3. Permite **identificar problemas conocidos** (degeneración, multimodalidad de la `r` recuperada, sesgos).
4. **Conecta con el curso de RL** del que es la presentación. IRL es un tema clásico del syllabus.

---

## 9. Frase para defensa del Pilar 1

> *"DPO, derivado del objetivo RLHF KL-restringido, codifica implícitamente una reward `r̂ = β·log(π/π_ref)`. Esta forma es estructuralmente equivalente a MaxEnt-IRL con prior. Entrenar con DPO es por tanto recuperar una reward implícita — IRL funcional. Self-Rewarding extiende esto: la reward recuperada es la del propio modelo en su modo juez. Esto está respaldado teóricamente por Ziebart 2008 y Rafailov 2023, y empíricamente por Joselowitz 2025, que muestra que IRL aplicado a LLMs RLHF recupera `r̂` con 85% de precisión sobre preferencias humanas."*

---

## 10. Tabla resumen del Pilar 1

| Paso | Resultado | Fuente |
|---|---|---|
| 1 | RLHF KL-restringido tiene solución de forma Boltzmann | Rafailov 2023 |
| 2 | Esa forma = MaxEnt-IRL con prior `π_ref` | Ziebart 2008 + Rafailov 2023 |
| 3 | DPO invierte la ecuación: política → reward implícita | Rafailov 2023 ec. (6) |
| 4 | Self-Rewarding aplica DPO a pares generados internamente | Yuan 2024 |
| 5 | ∴ Self-Rewarding = IRL funcional del juez interno | Síntesis de la presentación |
| 6 | IRL aplicado a LLMs RLHF empíricamente recupera `r̂` con 85% acc | Joselowitz 2025 |

---

## 11. Qué viene

[06-self-rewarding.md](06-self-rewarding.md) — Ahora sí, **el paper ancla**. Con todo lo anterior tienes el contexto completo.

---

## Lecturas relacionadas

- [[concepts/irl]] — IRL clásico denso.
- [[summaries/ziebart-2008-maxent-irl]] — MaxEnt-IRL original.
- [[summaries/rafailov-2023-dpo]] — DPO.
- [[summaries/joselowitz-2025-irl-llm]] — validación empírica.
- [[summaries/sun-vanderschaar-2024-irl-llm]] — tutorial IRL ↔ LLM.
- [[synthesis/pilar-1-irl-implicito]] — argumentación completa para la presentación.
