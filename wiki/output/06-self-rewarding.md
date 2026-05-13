# 06 — Self-Rewarding Language Models (Yuan et al. 2024) — **EL PAPER ANCLA**

> **Prerequisitos**: capítulos 00-05.

> **Lo que sabrás al final**: cómo funciona el loop iterativo paso a paso, el formato exacto de los pares de preferencia, los resultados empíricos clave, las tres lecturas teóricas (los pilares), y exactamente qué defender en la presentación.

---

## 1. La hipótesis central del paper

> 📐 **Hipótesis Yuan 2024 (en sus palabras)**:
> *"Para lograr agentes superhumanos, necesitamos feedback superhumano. Los humanos están limitados, y los reward models congelados se vuelven obsoletos cuando el LLM mejora. La solución: que el modelo se evalúe a sí mismo y use esa evaluación para mejorar iterativamente."*

Tres afirmaciones:

1. **Los humanos son cuello de botella**. Para tareas más allá del humano promedio, RLHF no funciona porque los humanos no pueden anotar bien.
2. **Reward models congelados se vuelven obsoletos**. Una vez entrenado, `r̂` no aprende — pero la política sí evoluciona → distribution shift.
3. **El loop puede cerrarse**. Si el modelo evalúa sus propios outputs, **el juez mejora junto con la política**.

---

## 2. El loop iterativo en cuatro pasos

### Vista general

```
M_0 (inicial)
    ↓
[Paso 1] Self-Instruction: generar N respuestas por prompt
[Paso 2] Self-Evaluation: el mismo M_0 las juzga
[Paso 3] Construcción de pares: max-score → y_w, min-score → y_l
[Paso 4] DPO sobre los pares → M_1
    ↓
Repetir con M_1, M_2, ...
```

> 📊 **El loop completo en un diagrama (para fijar)**:
>
> ```
>                  ╔══════════════════════╗
>                  ║   ITERACIÓN t        ║
>                  ║   Modelo actual: M_t ║
>                  ╚══════════════════════╝
>                            │
>                            ↓
>          ┌──────────────────────────────────────┐
>          │  PASO 1: SELF-INSTRUCTION             │
>          │                                       │
>          │  M_t genera prompts nuevos x_i        │
>          │  (few-shot prompting con seeds)       │
>          │                                       │
>          │  Para cada x_i, generar N respuestas: │
>          │  y_{i,1}, y_{i,2}, y_{i,3}, y_{i,4}   │
>          │  (sampling con T=0.7)                 │
>          └──────────────────────────────────────┘
>                            │
>                            ↓
>          ┌──────────────────────────────────────┐
>          │  PASO 2: SELF-EVALUATION              │
>          │                                       │
>          │  EL MISMO M_t actúa como juez         │
>          │  (LLM-as-a-Judge, rúbrica 5 puntos)   │
>          │                                       │
>          │  Asigna score s_{i,j} ∈ {0..5}        │
>          │  a cada respuesta y_{i,j}             │
>          └──────────────────────────────────────┘
>                            │
>                            ↓
>          ┌──────────────────────────────────────┐
>          │  PASO 3: CONSTRUCCIÓN DE PARES        │
>          │                                       │
>          │  Para cada x_i:                       │
>          │    y_w = respuesta con score máximo   │
>          │    y_l = respuesta con score mínimo   │
>          │    si scores iguales → descartar      │
>          │                                       │
>          │  Dataset D_t = {(x_i, y_w, y_l)}      │
>          └──────────────────────────────────────┘
>                            │
>                            ↓
>          ┌──────────────────────────────────────┐
>          │  PASO 4: DPO                          │
>          │                                       │
>          │  L_DPO sobre D_t:                     │
>          │    π_θ inicializada en M_t            │
>          │    π_ref = M_t (congelado)            │
>          │    β = 0.1                            │
>          │                                       │
>          │  Backprop → actualizar pesos          │
>          └──────────────────────────────────────┘
>                            │
>                            ↓
>                  ╔══════════════════════╗
>                  ║  M_{t+1} (mejorado)  ║
>                  ║  - Mejor generador   ║
>                  ║  - Mejor juez        ║◀── ¡KEY INSIGHT!
>                  ╚══════════════════════╝
>                            │
>                            ↓
>                    ¿más iteraciones?
>                            │
>                       sí ──┴── no
>                       ↓        ↓
>                    REPETIR   FIN
> ```

> 💡 **El diagrama clave** que debes poder dibujar de memoria en la presentación. Captura:
> 1. **El modelo es ambos**: generador (paso 1) Y juez (paso 2).
> 2. **El producto es un dataset DPO**: pares de preferencia auto-generados.
> 3. **El output es un nuevo modelo** que reemplaza el inicial.
> 4. **La iteración cierra el loop**: `M_t → M_{t+1} → M_{t+2}...`

### Paso 1 — Self-Instruction Creation

> 📐 **Setup**:
> - Partir de `M_0` = modelo SFT inicial.
> - Tener un conjunto de prompts seed (puede ser pequeño, ej. 3-5K).
> - Generar **prompts nuevos** vía few-shot prompting al propio modelo:
>   ```
>   "Aquí hay 3 ejemplos de prompts de instrucción.
>    Genera un cuarto prompt en el mismo estilo."
>   ```
> - Para cada prompt nuevo, generar **N respuestas** con sampling de temperatura (Yuan usa N=4, T~0.7).

Output del paso: `{(x_i, y_{i,1}, ..., y_{i,N}) : i = 1, ..., M}`.

> 💡 **Por qué generar prompts**: Yuan 2024 muestra que generar prompts diversos amplifica la mejora. Si solo usas los prompts del SFT, te quedas en una región pequeña del espacio. Con prompts generados, exploras más.

### Paso 2 — LLM-as-a-Judge Self-Evaluation

Para cada respuesta `y_{i,j}`:
- Pasar al juez (el mismo modelo `M_t` con prompt de evaluación).
- Rúbrica aditiva de 5 puntos (ver cap 04 §2.3).
- Obtener `s_{i,j} ∈ {0, 1, 2, 3, 4, 5}` (o continuo si usas score esperado).

> 💡 **Crítico**: el modelo **se evalúa a sí mismo**. No hay juez externo. La rúbrica aditiva ayuda a estructurar la decisión.

### Paso 3 — Construcción de pares de preferencia

Para cada prompt `x_i`:
```
y_w = argmax_j s_{i,j}
y_l = argmin_j s_{i,j}
```

Si `y_w` y `y_l` tienen el **mismo score**, descartar el prompt (no hay señal).

Output: dataset DPO `D_t = {(x_i, y_w, y_l)}`.

### Paso 4 — Entrenamiento con DPO

Aplicar la pérdida DPO (cap 02 §5.3) sobre `D_t`:
```
L_DPO(θ; π_ref=M_t) = − E_{(x, y_w, y_l) ~ D_t} [
    log σ( β · log(π_θ(y_w|x)/M_t(y_w|x)) − β · log(π_θ(y_l|x)/M_t(y_l|x)) )
]
```

> ⚠️ **Detalle sutil**: en el paper original, `π_ref` para la iteración `t→t+1` es `M_t` (el modelo de la iteración anterior). No es el SFT original. Esto significa que `π_ref` se actualiza cada iteración. Hay variaciones en la literatura sobre este punto.

Después de entrenar:
```
M_{t+1} = π_θ (resultado de DPO)
```

Vuelta al paso 1 con `M_{t+1}`.

---

## 3. La crucial observación: el juez también mejora

Aquí está el **hallazgo central** del paper, y la fuente de la pregunta sutil más importante.

### 3.1. La observación

Cuando aplicas DPO sobre los pares generados por el juez de `M_t`, los pesos del modelo se actualizan. **Pero esos mismos pesos producen los juicios**. Por tanto:

> 💡 **El juez `M_{t+1}` es estructuralmente distinto del juez `M_t`**, y empíricamente Yuan 2024 muestra que **`M_{t+1}` juzga mejor que `M_t`** según métricas de acuerdo con humanos.

Esto es **lo que diferencia Self-Rewarding de RLAIF clásico**. En RLAIF, el juez es congelado → la calidad del sistema está limitada por la calidad inicial del juez. En Self-Rewarding, el juez evoluciona → ¿en principio, sin límite?

(Spoiler del cap 12: no, en práctica satura en 3 iteraciones. Pero la idea es notable.)

### 3.2. Pregunta sutil: ¿por qué mejora el juez si DPO solo entrena generación?

Este es **el punto más confuso** del paper para alguien que viene de NLP. La pérdida DPO es:

```
L_DPO = − E[ log σ( β · log(π_θ(y_w|x)/π_ref(y_w|x)) − β · log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

Esta pérdida **solo se calcula sobre respuestas**, no sobre juicios. Solo empuja `π_θ(y_w|x)` arriba y `π_θ(y_l|x)` abajo. **¿Por qué entonces mejora el juicio?**

### 3.3. La respuesta — tres componentes

> 📊 **Diagrama del mecanismo**:
>
> ```
>   ┌──────────────────────────────────────────────────────┐
>   │  EL LLM ES UNA SOLA RED DE PESOS θ                   │
>   │                                                       │
>   │     Pesos θ ⟶  capacidades subyacentes:               │
>   │                  - comprensión semántica              │
>   │                  - razonamiento                       │
>   │                  - conocimiento del mundo             │
>   │                  - habilidad de seguir instrucciones  │
>   │                                                       │
>   └──────────────────────────────────────────────────────┘
>                            │
>                ┌───────────┴───────────┐
>                │                       │
>                ↓                       ↓
>    ┌──────────────────┐    ┌──────────────────┐
>    │  Modo GENERADOR   │    │  Modo JUEZ       │
>    │                   │    │                   │
>    │  Prompt: "Pregún- │    │  Prompt: "Evalúa │
>    │  tame algo"       │    │  esta respuesta" │
>    │                   │    │                   │
>    │  Forward pass con │    │  Forward pass con│
>    │  los mismos pesos │    │  los mismos pesos│
>    │  θ                │    │  θ               │
>    │                   │    │                   │
>    │  Output: y        │    │  Output: score   │
>    └──────────────────┘    └──────────────────┘
> ```

**Componente 1 — Capabilities share**:

Generar bien y juzgar bien comparten **las mismas capacidades subyacentes**:
- Para escribir buena respuesta a "explica un transformer", necesitas entender qué es un transformer.
- Para juzgar si una respuesta sobre transformers es buena, necesitas **el mismo conocimiento**.

Cuando DPO mejora estas capacidades (al empujar `π_θ` hacia respuestas mejor scoreadas), las capacidades **se elevan** y eso beneficia **ambos modos**.

**Componente 2 — Better reasoning → better judgment**:

El prompt del juez típicamente incluye **chain-of-thought**: "Razona primero, luego da el score". El razonamiento es el mismo tipo de razonamiento que el modelo usa al generar respuestas. Si el modelo razona mejor en general (efecto colateral de DPO), también razona mejor al evaluar.

**Componente 3 — Consistency**:

Cuando DPO empuja `π_θ` a producir más `y_w` (respuestas de score alto) y menos `y_l` (score bajo), el modelo aprende **implícitamente** qué características hacen que `y_w` reciba alto score. Esa misma información — qué hace que una respuesta sea buena — es **exactamente lo que necesitas para juzgar**.

### 3.4. Visualizado: por qué los pesos compartidos importan

> 📊 **El efecto del update DPO sobre los dos modos**:
>
> ```
>   ANTES de DPO (iteración t):
>     ┌─────────────────────────────────────────────────────┐
>     │ Pesos θ_t producen:                                  │
>     │                                                       │
>     │  Modo generador:                                      │
>     │    π_{θ_t}(y_w | x) = 0.3                            │
>     │    π_{θ_t}(y_l | x) = 0.4                            │
>     │                                                       │
>     │  Modo juez:                                           │
>     │    Score(y_w) según θ_t = 4/5                        │
>     │    Score(y_l) según θ_t = 3/5                        │
>     │    (juez ya tiene una pista, pero acuerdo con humano = 65%)│
>     └─────────────────────────────────────────────────────┘
>
>   Aplicamos DPO. Gradiente:
>     - Sube π_θ(y_w|x), baja π_θ(y_l|x)
>     - Pero θ_t es una red enorme con miles de parámetros compartidos
>     - El gradiente toca pesos que afectan AMBOS modos
>
>   DESPUÉS de DPO (iteración t+1):
>     ┌─────────────────────────────────────────────────────┐
>     │ Pesos θ_{t+1} producen:                              │
>     │                                                       │
>     │  Modo generador:                                      │
>     │    π_{θ_{t+1}}(y_w | x) = 0.5  ← subió               │
>     │    π_{θ_{t+1}}(y_l | x) = 0.2  ← bajó                │
>     │                                                       │
>     │  Modo juez:                                           │
>     │    Score(y_w) según θ_{t+1} = 4.5/5  ← más confiado   │
>     │    Score(y_l) según θ_{t+1} = 2/5    ← más confiado   │
>     │    (acuerdo con humano = 68%, MEJORÓ)                │
>     └─────────────────────────────────────────────────────┘
> ```

### 3.5. Datos empíricos del paper

Yuan 2024 reporta el acuerdo del juez con anotadores humanos sobre un set de validación:

| Iter | Acuerdo con humanos |
|---|---|
| M_0 | 65.1% |
| M_1 | 68.3% |
| M_2 | 70.2% |
| M_3 | (no reportado en detalle, pero satura) |

**Importante**: la mejora del juez es **modesta** comparada con la del generador (AlpacaEval va de 9.94% a 20.44%). Esto sugiere que la mejora colateral existe pero es **limitada**.

### 3.6. ¿Por qué entonces satura?

Si el juez mejora con cada iteración, ¿por qué no continúa mejorando indefinidamente?

Dos razones:

1. **La mejora del juez es indirecta**. DPO no apunta a mejorarla — solo emerge por capabilities share. La señal de gradiente sobre el juez es **diluida**.

2. **El juez también empieza a sufrir self-preference bias**. A medida que `M_t` evoluciona, su distribución preferida se desplaza, y el juez prefiere outputs de esa distribución (que ahora son también los que `M_t` genera). El feedback loop **se cierra sobre sí mismo**.

Esta es exactamente la motivación de **Meta-Rewarding** (cap 07): si el problema es que el juez no se entrena explícitamente, **entrenémoslo explícitamente** introduciendo el meta-juez.

### 3.7. Frase para defensa

> *"DPO solo entrena el modo generador en su pérdida explícita. Pero el LLM es una sola red de pesos: las capacidades subyacentes — razonamiento, comprensión, conocimiento — son compartidas entre 'generar' y 'juzgar'. Cuando DPO mejora estas capacidades vía el gradiente, ambos modos se benefician. Yuan 2024 lo mide: el acuerdo del juez con humanos crece de 65% a 70% en tres iteraciones. La mejora colateral es real pero modesta — esto es precisamente lo que motiva Meta-Rewarding, que entrena el juicio explícitamente."*

---

## 4. Datos concretos del experimento principal

| Setup | Valor |
|---|---|
| Modelo base | Llama 2 70B |
| SFT inicial | Sobre OpenAssistant + EFT (Evaluation Fine-Tuning) seed |
| Prompts seed | 3.2K de instrucciones + 1.4K de evaluaciones |
| Prompts generados por iter | ~5K nuevos |
| N (respuestas por prompt) | 4 |
| Temperatura sampling | 0.7 |
| β DPO | 0.1 |
| LR DPO | 1e-6 → 1e-7 con decay |
| Batch size | 16 |
| Iteraciones evaluadas | M_0 → M_1 → M_2 → M_3 |

> ⚠️ **EFT (Evaluation Fine-Tuning)** es importante: además del SFT estándar, el modelo se entrena específicamente en la **tarea de evaluación** (usar la rúbrica 5-puntos correctamente). Sin esto, el juez inicial es demasiado pobre.

---

## 5. Resultados empíricos

### 5.1. AlpacaEval 2.0 — el headline

| Modelo | Win rate vs GPT-4 0314 (length-controlled) |
|---|---|
| M_0 (SFT inicial) | 9.94% |
| M_1 (iter 1) | 15.38% |
| M_2 (iter 2) | 20.44% |
| M_3 (iter 3) | **20.44%** (no mejora) |
| Claude 2 | 17.19% |
| Gemini Pro | 16.85% |
| GPT-4 0613 | 15.76% |
| GPT-4 1106 | 50.00% (referencia) |

> 💡 **El claim "supera a Claude 2, Gemini Pro, GPT-4 0613"** es real. Pero hay que matizarlo: supera a esas versiones **específicas** en AlpacaEval 2.0 — un benchmark concreto que privilegia ciertos estilos. No los supera en capacidad general.

### 5.2. La capacidad de juicio también mejora

Yuan reporta acuerdo del juez con anotadores humanos en RewardBench-like benchmarks:

| Iter | Acuerdo con humanos en juicios |
|---|---|
| M_0 | 65.1% |
| M_1 | 68.3% |
| M_2 | 70.2% |

> 💡 **Este es el resultado más conceptualmente importante**. No solo el generador mejora — el juez también. La calidad de la señal de entrenamiento crece junto con la del modelo.

### 5.3. Saturación a 3 iteraciones

`M_3` no supera a `M_2`. **Esto no se enfatiza en el paper original**, pero las críticas posteriores (cap 12) muestran que es el **modo de fallo principal** del método.

---

## 6. Las tres lecturas teóricas (los tres pilares)

Yuan 2024 propone Self-Rewarding como instancia de tres frameworks distintos.

### 6.1. Pilar 1 — Inverse RL implícito

Ya desarrollado en cap 05. **Resumen**: DPO recupera una reward implícita; Self-Rewarding aplica DPO sobre pares generados por el juez interno; por tanto Self-Rewarding **infiere implícitamente** la reward del juez. Esto es IRL funcional.

**Defendible con**: Ziebart 2008 + Rafailov 2023 (formal), Joselowitz 2025 (empírico).

### 6.2. Pilar 2 — Cooperative Multi-Agent RL

Self-Rewarding opera con **dos agentes cooperativos** (mismo modelo, distintos roles):

- **Generador**: produce respuestas. Objetivo: maximizar score del juez.
- **Juez**: evalúa. Objetivo: producir señales útiles para entrenamiento.

La cooperación emerge porque:
- Mejor juez → señales más precisas → mejor generador.
- Mejor generador → respuestas más informativas → juez mejor calibrado.

> 💡 **Esto NO es multi-agent RL formal** (los agentes no comparten un environment con MDP joint). Es **estructural-cooperativo**: dos roles que cooperan en un objetivo común (mejorar el modelo).

**Conexión con Hierarchical RL**: Meta-Rewarding (cap 07) añade un tercer agente (meta-juez) que opera a un nivel jerárquico superior. Esto **sí** se parece más a MARL.

### 6.3. Pilar 3 — DPO en el loop

DPO no es solo "la técnica de optimización". Es **lo que hace posible el loop iterativo**:

1. **Offline**: pares fijos → no requiere generar durante entrenamiento.
2. **Estable**: una pasada DPO es predecible.
3. **Iterable**: la output del DPO es una política bien definida que sirve como input para la siguiente iteración.

Sin DPO, Self-Rewarding sería impráctico (PPO en loop iterativo es inestable y costoso).

---

## 7. ¿Qué hace Self-Rewarding "nuevo"?

Tres cosas, ya mencionadas en cap 03 pero las reitero:

1. **Formaliza** que el juez puede ser el mismo modelo.
2. **Iteriza** explícitamente: define la secuencia `M_0 → M_1 → M_2`.
3. **Muestra empíricamente** que el juez también mejora con cada iteración.

> ⚠️ **Lo que NO inventa**:
> - LLM-as-a-Judge: ya existía (Zheng 2023).
> - Auto-evaluación: ya existía (Constitutional AI 2022).
> - DPO: ya existía (Rafailov 2023).
> - "Auto-mejora": ya existía conceptualmente (STaR, Self-Refine, Self-Improve).
>
> Self-Rewarding es **un ensamblaje novedoso** de piezas existentes que produce un loop limpio.

---

## 8. Pseudocódigo completo

```python
# Inicialización
M = SFT_with_EFT(base_model, seed_data)
prompts_seed = load_prompts()

for t in range(T):  # T = 3 en Yuan 2024
    # === Paso 1: Self-Instruction ===
    prompts_new = []
    for _ in range(num_new_prompts):
        x = M.generate_with_few_shot(prompts_seed)
        prompts_new.append(x)
    
    # === Paso 2: Generación + Self-Evaluation ===
    pairs = []
    for x in prompts_new:
        responses = [M.generate(x, temperature=0.7) for _ in range(N=4)]
        scores = [judge_score(M, x, y) for y in responses]
        
        y_w = responses[argmax(scores)]
        y_l = responses[argmin(scores)]
        
        if scores[argmax(scores)] > scores[argmin(scores)]:  # ojo: margen real
            pairs.append((x, y_w, y_l))
    
    # === Paso 3: Construcción dataset ===
    D_t = pairs
    
    # === Paso 4: DPO ===
    M_new = train_DPO(
        policy=M,
        ref=M,           # ref es el modelo de la iter anterior
        dataset=D_t,
        beta=0.1,
        lr=1e-6
    )
    
    M = M_new

def judge_score(model, x, y):
    """LLM-as-a-Judge con rúbrica aditiva 5-puntos."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=x, response=y)
    output = model.generate(prompt)
    score = parse_score(output)  # extracts "Score: N" from output
    return score
```

---

## 9. ¿Qué papel juega EFT?

**EFT (Evaluation Fine-Tuning)** es un detalle crítico que el paper menciona pero que se pierde en resúmenes superficiales.

**Sin EFT**: el modelo SFT base no sabe usar bien la rúbrica de 5 puntos. Su juez es ruidoso. Los pares generados son malos. DPO no logra nada.

**Con EFT**: durante el SFT inicial, el modelo se entrena específicamente en:
1. Producir scores según la rúbrica aditiva.
2. Justificar el score con razonamiento.

Esto requiere un dataset adicional de ~1.4K ejemplos `(prompt, respuesta, score_correcto, justificación)`.

> 💡 **Lectura general**: EFT prepara al modelo para tener un juez decente desde el inicio. Es **el bootstrap del juez**.

---

## 10. Sesgos y problemas (preview del cap 12)

**Self-preference bias amplificado**: el juez prefiere outputs similares a los que él mismo produciría. Como juez = generador, esto se amplifica al máximo. Resultado: parte de la mejora reportada podría ser **inflación por auto-preferencia**, no calidad real.

**Saturación**: ya mencionada. `M_3 ≈ M_2`. Yuan 2024 no enfatiza esto pero está en sus números.

**Falla en razonamiento técnico**: AlpacaEval evalúa instruction-following, no matemáticas. Self-Rewarding ingenuo **degrada** en math (cap 09).

**Drift acumulativo**: `KL(M_t || M_0)` crece sin cota. El modelo se aleja de la inicialización; en algún punto puede degradar capacidades preentrenadas.

---

## 11. Reproducibilidad

Yuan 2024 publica:
- Receta completa (hiperparámetros).
- Datos seed.
- Prompts del juez.

**No publica**: pesos finales de `M_t` (es Llama 2 70B fine-tuneado por Meta, restricciones de licencia).

Reproducciones independientes:
- **Reproducciones exitosas a menor escala** (7B) confirman la mejora, aunque más modesta.
- **CREAM (Wang Z. 2024)** muestra que en modelos pequeños hay **regresión** sin regularización.
- **SPPO (Wu 2024)** hace head-to-head Self-Rewarding vs SPPO vs SPIN sobre Mistral-7B-Instruct-v0.2 con setup uniforme; Self-Rewarding compite pero no domina.

---

## 12. Argumento conjunto para defender

Los tres pilares juntos te dan **el speech maestro**:

> *"Self-Rewarding es la intersección de tres temas del curso. **Primero, Inverse RL implícito**: DPO, derivado del óptimo KL-RLHF en forma MaxEnt-Boltzmann, codifica una reward implícita; Self-Rewarding entrena esa reward usando pares generados por el propio modelo como juez. **Segundo, Cooperative Multi-Agent**: el modelo opera con dos roles cooperativos (generador y juez) que se mejoran mutuamente. **Tercero, DPO en el loop**: la simplicidad y estabilidad de DPO son lo que hace posible la iteración limpia que distingue este método de RLAIF clásico. Empíricamente, tres iteraciones sobre Llama 2 70B superan a GPT-4 0613, Claude 2 y Gemini Pro en AlpacaEval 2.0 — sin un solo nuevo anotador humano. Sus límites — saturación a 3 iter, self-preference bias, falla en razonamiento técnico — son extensamente documentados y abordados por la familia de métodos posteriores."*

---

## 13. Qué viene

[07-meta-rewarding.md](07-meta-rewarding.md) — Wu et al. 2024. La extensión inmediata: ¿cómo evitar que el juez se sature?

---

## Lecturas relacionadas

- [[summaries/yuan-2024-self-rewarding]] — resumen denso.
- [[concepts/self-rewarding-loop]] — el algoritmo en notación compacta.
- [[synthesis/pilar-1-irl-implicito]] — argumentación Pilar 1.
- [[synthesis/pilar-2-multi-agente-cooperativo]] — argumentación Pilar 2.
- [[synthesis/pilar-3-dpo-en-loop]] — argumentación Pilar 3.
- [[synthesis/diagnostico-fallos]] — anticipo de las críticas.
