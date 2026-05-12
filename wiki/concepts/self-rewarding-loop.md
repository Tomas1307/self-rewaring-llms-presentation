---
name: self-rewarding-loop
description: Algoritmo completo del loop iterativo de Self-Rewarding — pseudocódigo, hiperparámetros, dinámica de las iteraciones.
metadata:
  type: concept
  tags: [self-rewarding, loop, algoritmo, pseudocodigo, dpo]
  last_updated: 2026-05-12
---

# El loop de Self-Rewarding

> Algoritmo iterativo de auto-mejora donde un único modelo cumple los roles de generador y juez, optimizado vía DPO sobre pares auto-generados. Fuente: [[summaries/yuan-2024-self-rewarding]].

## 1. Esquema general

```
M_0 (modelo base, p.ej. Llama-2-70B)
  │
  ▼
SFT con IFT (instrucción) + EFT (evaluación)
  │
  ▼
M_1 (modelo semilla)
  │
  ▼  iteración 1
[Generación → Auto-evaluación → Construcción de pares → DPO]
  │
  ▼
M_2
  │
  ▼  iteración 2
[mismo loop]
  │
  ▼
M_3
  │
  ▼  iteración 3
[mismo loop]
  │
  ▼
M_4
```

Yuan 2024 evalúa hasta `M_3`. Saturación documentada más allá ([[synthesis/diagnostico-fallos]] §2.1).

## 2. Pseudocódigo de una iteración

```
INPUT:
  M_t            : modelo actual
  X_t            : pool de prompts (auto-generados o tomados de un dataset)
  N              : respuestas por prompt (Yuan 2024: N = 4)
  T, p           : temperatura, top-p (Yuan 2024: T=0.7, p=0.9)
  K              : número de evaluaciones por respuesta (Yuan 2024: K = 3)
  β              : coeficiente KL para DPO (Yuan 2024: β = 0.1)

OUTPUT:
  M_{t+1}        : modelo mejorado

# Paso 1 — Generación
AIFT(M_t) ← ∅
for x in X_t:
    y_1, ..., y_N ~ M_t(· | x, prompt_generación)  # sampling con T, p

    # Paso 2 — Auto-evaluación
    for i in 1..N:
        scores = []
        for k in 1..K:
            s_k ← M_t(score | x, y_i, prompt_juez)  # rúbrica aditiva 0..5
            scores.append(s_k)
        s_i ← mean(scores)

    # Paso 3 — Construcción de pares
    y_w ← argmax_i s_i
    y_l ← argmin_i s_i
    if s_{y_w} > s_{y_l}:
        AIFT(M_t).add( (x, y_w, y_l) )
    # else: descartar (sin señal de aprendizaje)

# Paso 4 — DPO update
M_{t+1} ← copy(M_t)
π_ref ← M_t  # congelado para esta iteración
for epoch in 1..E:
    for (x, y_w, y_l) in batches(AIFT(M_t)):
        compute L_DPO(M_{t+1}; π_ref, (x, y_w, y_l), β)
        backprop, update M_{t+1}

return M_{t+1}
```

## 3. Detalles importantes

### 3.1. Generación de prompts (`X_t`)

Yuan 2024 usa **self-instruction creation**: el modelo `M_t` genera nuevos prompts a partir de demos few-shot (8-shot del propio dataset IFT). Esto evita depender de prompts externos y permite que el pool de entrenamiento crezca con el modelo.

Alternativa común: tomar prompts de datasets existentes (UltraFeedback, AlpacaEval-train, etc.).

### 3.2. Rúbrica del juez (LLM-as-a-Judge)

Rúbrica aditiva de 5 criterios binarios:

| # | Criterio | Pregunta |
|---|---|---|
| 1 | Relevancia | ¿La respuesta aborda lo que el prompt pide? |
| 2 | Cobertura | ¿Cubre los puntos principales sin omitir? |
| 3 | Utilidad | ¿Sería útil para el usuario? |
| 4 | Claridad | ¿Está bien escrita y organizada? |
| 5 | Experticia | ¿Demuestra conocimiento profundo del tema? |

Score final: `s ∈ {0, 1, 2, 3, 4, 5}`. Ver [[concepts/llm-as-a-judge]] para el prompt completo.

### 3.3. Construcción de pares — detalles

- Si `s_max = s_min`, descartar (sin señal).
- Yuan 2024 promedia `K=3` evaluaciones independientes por respuesta para reducir varianza.
- Tamaños empíricos: AIFT(M_1) ≈ 4000 pares; AIFT(M_2) ≈ 7000 pares.

### 3.4. DPO update — π_ref se mueve

**Crítico**: `π_ref = M_t` (no `M_0`). Cada iteración la referencia es el modelo de la iteración anterior. Esto significa:
- Cada DPO restringe `M_{t+1}` cerca de `M_t`, no de `M_0`.
- El KL acumulado `KL(M_t ‖ M_0)` puede crecer sin cota con `t`.
- Esto explica el drift acumulativo documentado por [[summaries/wang-2025-temporal-sr]].

Variante alternativa propuesta por Wang 2025 (*Anchored Rejection*): mantener `y_l` derivado de `M_0` para preservar el gap representacional.

## 4. La "doble mejora" emergente

Yuan 2024 reporta que `M_{t+1}` mejora simultáneamente:

| Rol | Métrica | Mejora |
|---|---|---|
| Actor | AlpacaEval 2.0 win rate vs GPT-4 Turbo | 9.94 → 15.38 → 20.44 (3 iter) |
| Judge | Correlación de scores con humanos | Monotónica iter sobre iter |

**Razón**: las representaciones internas para generar bien y para evaluar bien **comparten parámetros θ**. Mejor representación de calidad ⇒ mejores generaciones y mejores juicios.

[[summaries/wu-2024-meta-rewarding]] explota esto desacoplando los canales: entrena el juez explícitamente vía meta-judge, obteniendo mejoras más sostenidas y separables.

## 5. Hiperparámetros consolidados (Yuan 2024)

| Parámetro | Valor |
|---|---|
| Modelo base | Llama-2-70B |
| SFT LR inicial | 5.5e-6 |
| SFT LR decay | cosine → 1.1e-6 |
| DPO LR inicial | 1e-6 |
| DPO LR decay | → 1e-7 |
| DPO β | **0.1** |
| Batch size | 16 |
| Dropout | 0.1 |
| Early stopping | cada 200 steps con Claude 2 sobre 253 ejemplos |
| Temperatura generación | 0.7 |
| Top-p | 0.9 |
| Respuestas por prompt (N) | 4 |
| Evaluaciones por respuesta (K) | 3 |

## 6. Variaciones publicadas posteriores

- **[[summaries/wu-2024-meta-rewarding]]** — agrega un meta-judge (tercer rol).
- **[[summaries/wang-2025-temporal-sr]]** — Anchored Rejection + Future-Guided Chosen para evitar colapso.
- **Process-SRLM** (Zhang 2025, arXiv:2503.03746) — juicio step-wise para razonamiento matemático.
- **SCIR** (Zhou 2025, arXiv:2502.08922) — filtra pares con inconsistencia entre rewards internos.

## 7. Modos de fallo del loop

Ver [[synthesis/diagnostico-fallos]] para la lista completa. Los principales:

1. **Saturación del gap representacional** (Wang 2025).
2. **Inconsistencia entre rewards internos** (Zhou 2025).
3. **Falla en dominios verificables** (Zhang 2025, Shafayat 2025).
4. **Self-preference bias amplificado** (Zheng 2023, arXiv:2410.21819).
5. **Degeneración del pensamiento (DoT)** (Liang 2024).

## 8. Lecturas relacionadas

- [[summaries/yuan-2024-self-rewarding]] — paper original.
- [[concepts/dpo]] — pérdida usada en el paso 4.
- [[concepts/llm-as-a-judge]] — paradigma del juez interno.
- [[concepts/preference-pair]] — construcción de pares en el paso 3.
- [[synthesis/pilar-3-dpo-en-loop]] — análisis del rol de DPO.
- [[synthesis/diagnostico-fallos]] — modos de fallo del loop.
