---
name: preference-pair
description: Modelo Bradley-Terry y construcción de pares de preferencia (x, y_w, y_l) — núcleo del dataset que DPO consume.
metadata:
  type: concept
  tags: [bradley-terry, preferencias, dataset, dpo, rlhf]
  last_updated: 2026-05-12
---

# Pares de preferencia

> El "átomo" del entrenamiento basado en preferencias: tripletas `(x, y_w, y_l)` donde `y_w` (winner / chosen) es preferida sobre `y_l` (loser / rejected) dado el prompt `x`. Modelo subyacente: Bradley-Terry.

## 1. La tripleta

```
(x, y_w, y_l)
   │   │   │
   │   │   └── respuesta rechazada (loser / rejected)
   │   └────── respuesta preferida (winner / chosen)
   └────────── prompt / contexto
```

**Convención**: `y_w` es siempre la preferida según el juez (humano, LLM-judge, ranking interno). La pérdida DPO maximiza la probabilidad de `y_w` relativa a `y_l`.

## 2. Modelo de Bradley-Terry

Bradley-Terry (1952) modela preferencias pairwise como:
```
P(y_w ≻ y_l | x) = exp(r(x, y_w)) / [ exp(r(x, y_w)) + exp(r(x, y_l)) ]
                 = σ( r(x, y_w) − r(x, y_l) )
```

donde `r` es una **función de utilidad** (score) sobre respuestas, y `σ` es la sigmoide logística.

**Propiedades**:
- Solo importa la **diferencia** `r(y_w) − r(y_l)`, no los valores absolutos.
- Asume **transitividad estocástica**: si `y_w ≻ y_l` y `y_l ≻ y_z`, entonces `P(y_w ≻ y_z) ≥ max(P(y_w ≻ y_l), P(y_l ≻ y_z))`.
- Asume **independencia de alternativas irrelevantes**: la preferencia entre `y_w` y `y_l` no depende de otras opciones.

## 3. ¿Cómo se construyen los pares?

Tres fuentes principales en la literatura:

### 3.1. Anotación humana (RLHF clásico)

[[summaries/ouyang-2022-instructgpt]]: humanos rankean N=4-9 respuestas; se generan `C(N,2)` pares por prompt. Ejemplo InstructGPT: ~33k prompts, ~70k pares.

**Costo**: alto. Tiempo de anotador limita escala.

### 3.2. AI feedback (RLAIF)

[[summaries/bai-2022-constitutional-ai]] y Lee 2023: un LLM externo (típicamente GPT-4) genera las preferencias. Coste ~10× menor que humano. Calidad comparable según Lee 2023.

### 3.3. Self-feedback (Self-Rewarding)

[[summaries/yuan-2024-self-rewarding]]: el **mismo modelo** entrenado genera las preferencias vía [[concepts/llm-as-a-judge]].

Algoritmo concreto:
```
1. Para cada x: muestrear N=4 respuestas {y_1, ..., y_N}.
2. Para cada y_i: obtener s_i = score(y_i | x) vía juez interno.
3. y_w ← argmax_i s_i
4. y_l ← argmin_i s_i
5. Si s_w > s_l: agregar (x, y_w, y_l).
   Si s_w = s_l: descartar (sin señal de gradiente).
```

**Importante**: solo se conserva un par por prompt (max, min). Las otras `C(N,2)−1` combinaciones se descartan. Trade-off: menos pares pero más distinguibles.

## 4. Por qué importa la calidad de los pares

La pérdida DPO depende del **gap representacional** `r̂(y_w) − r̂(y_l)`:

```
∇ L_DPO ∝ σ( β·[r̂(y_w) − r̂(y_l)] ) · ∇[r̂(y_w) − r̂(y_l)]
```

- **Par con gap grande**: `σ → 1`, gradiente pequeño, señal "ya entendí, ajusto poco".
- **Par con gap mediano**: gradiente útil.
- **Par con gap cero**: gradiente nulo, sin señal.
- **Par invertido** (mal anotado): gradiente fuerte en dirección equivocada.

**Implicación para Self-Rewarding**: la saturación documentada por [[summaries/wang-2025-temporal-sr]] es exactamente este problema. A medida que el modelo mejora, todos los `y_i` se vuelven buenos → `s_w ≈ s_l` → gap → 0 → loop se anula.

## 5. Modos de fallo del dataset de pares

### 5.1. Pares con gap nulo

Si el juez asigna el mismo score, no hay señal. Yuan 2024 los descarta.

### 5.2. Pares ruidosos (mal anotados)

Especialmente problemático con jueces débiles. El paper de [[summaries/azar-2023-ipo]] muestra que el ruido en preferencias hace que DPO sobre-ajuste. IPO regulariza mejor en este régimen.

### 5.3. Pares estilísticamente sesgados

Si el juez prefiere respuestas largas, todos los `y_w` son largos. El modelo aprende "ser largo" como proxy de calidad. Es el problema de **length inflation** documentado en Yuan 2024 y mitigado en Wu 2024 (Meta-Rewarding) con length-control.

### 5.4. Pares cuasi-deterministas

Si `P(y_w ≻ y_l) → 1` (es decir, `r(y_w) − r(y_l)` enorme), la pérdida DPO `−log σ(β·diff)` diverge — fuerza `π_θ` arbitrariamente lejos de `π_ref`. IPO ([[summaries/azar-2023-ipo]]) lo soluciona con regularización cuadrática.

## 6. Variantes en construcción de pares

- **Best-of-N + worst-of-N** (Yuan 2024): preserva contraste máximo, pero solo 1 par por prompt.
- **All-pairs**: usar todas las `C(N,2)` combinaciones. Más datos, pero más pares con gap pequeño.
- **Adjacent pairs**: solo pares `(y_i, y_{i+1})` ordenados por score. Útil cuando hay ranking explícito.
- **Anchored Rejection** (Wang 2025): `y_l` proviene de `M_0`, no del modelo actual. Preserva gap iter sobre iter.
- **Future-Guided Chosen** (Wang 2025): `y_w` proviene de un proxy de `M_{t+1}`. Agranda gap.

## 7. Sin pares — alternativas

- **KTO** (Ethayarajh 2024): solo etiquetas binarias deseable/no-deseable por respuesta. No requiere pares.
- **ORPO** (Hong 2024): combina SFT y alineación; necesita pares pero elimina `π_ref`.
- **Online RL con reward model** (PPO clásico): genera respuestas on-policy y las puntúa con `r̂`, sin construir pares explícitos.

## 8. Implicaciones para la presentación

Una slide sobre "construcción de pares" en Self-Rewarding puede mostrar:

1. La tripleta `(x, y_w, y_l)`.
2. El score aditivo (rúbrica de 5 puntos).
3. La regla argmax/argmin.
4. **El problema**: si todas las respuestas son buenas, los pares se vuelven uninformativos → Wang 2025.

Esto conecta limpiamente con [[synthesis/diagnostico-fallos]] §2.1.

## 9. Lecturas relacionadas

- [[concepts/dpo]] — pérdida que consume estos pares.
- [[concepts/llm-as-a-judge]] — genera los scores que producen los pares.
- [[concepts/self-rewarding-loop]] — el algoritmo donde se construyen.
- [[summaries/yuan-2024-self-rewarding]] — construcción concreta en el paper ancla.
- [[summaries/wang-2025-temporal-sr]] — diagnóstico de la saturación de pares.
- [[summaries/azar-2023-ipo]] — qué pasa con pares cuasi-deterministas.
