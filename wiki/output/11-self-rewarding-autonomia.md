# 11 — Autonomía al límite: Self-Taught Evaluators, RLSR, Multi-Agent Evolve

> **Prerequisitos**: capítulos 06, 07, 08.

> **Lo que sabrás al final**: tres métodos que llevan la auto-mejora más allá de "evaluar respuestas": el juez se auto-entrena (Self-Taught Evaluators), el modelo genera sus propias tareas (RLSR), múltiples instancias co-evolucionan (Multi-Agent Evolve).

---

## 1. La frontera de la autonomía

Self-Rewarding ya elimina anotadores humanos. Pero hay **niveles de autonomía más profundos** que la comunidad está explorando:

| Nivel | Lo que se automatiza | Métodos |
|---|---|---|
| 0 | Nada — RLHF clásico | RLHF |
| 1 | Anotador → LLM externo | RLAIF, CAI |
| 2 | Juez = mismo modelo (iterativo) | Self-Rewarding, Meta-Rewarding |
| **3** | **Juez se auto-entrena explícitamente** | **Self-Taught Evaluators** |
| **4** | **Modelo genera sus propias tareas (currículo)** | **RLSR** |
| **5** | **Múltiples instancias co-evolucionan** | **Multi-Agent Evolve** |

Estos tres métodos están en el extremo de "agentes autónomos en formación".

---

## 2. Self-Taught Evaluators (Tianlu Wang et al. 2024, Meta FAIR)

### 2.1. La idea

¿Qué pasa si entrenamos **solo al juez**, sin humanos? Self-Taught Evaluators construye un LLM-as-a-Judge **partiendo de cero** (sin etiquetas humanas).

### 2.2. Algoritmo

```
Inicialización: juez_0 = LLM base + prompt de evaluación.

Iteración t:
  1. Tomar prompts no anotados.
  2. Generar pares de respuestas contrastantes sintéticamente:
     - y_w: respuesta de calidad del modelo (potencialmente buena).
     - y_l: respuesta degradada (instruir al modelo a producir errores).
  3. juez_t produce un juicio sobre cada par.
  4. Entrenar juez_{t+1} en (par, juicio_t) — usando el propio juicio
     como señal supervisada.
  5. Repetir.
```

### 2.3. Por qué funciona

**Truco crítico**: las degradaciones sintéticas (`y_l`) están **diseñadas para ser claramente peores**. El juez inicial tiene **alta confianza** en los pares donde hay degradación obvia. Esos pares con juicios confiables se usan como **dataset supervisado** para entrenar al juez.

A medida que el juez mejora, puede distinguir pares más sutiles, y se enriquece el dataset.

### 2.4. Resultados

Llama3-70B-Instruct como juez:
- Baseline: 75.4 en RewardBench.
- Tras Self-Taught Evaluators (varias iter): **88.3** (88.7 con majority voting).
- Supera a GPT-4 como evaluador, **sin etiquetas humanas**.

### 2.5. Relevancia

Confirma una tesis profunda de Self-Rewarding: **el juez puede mejorar autónomamente**. Self-Taught Evaluators lo lleva al extremo — no entrena la política, solo el juez. El resultado es un juez que supera a GPT-4 sin haber visto preferencias humanas.

> 💡 **Esto valida indirectamente** el componente "juez mejora" de Self-Rewarding (Yuan 2024). Si el juez puede entrenarse autónomamente solo, entonces el juez puede mejorar en cualquier loop que lo incluya.

---

## 3. RLSR — Reinforcement Learning from Self Reward (2025)

### 3.1. El paso conceptual

RLSR observa: **un modelo puede no solo evaluar sus respuestas, también generar las preguntas óptimas para practicar**.

> 💡 **Analogía**: como un estudiante que **identifica sus propias debilidades** y se diseña ejercicios para superarlas. Es el modelo aprendiendo a **automejorarse curricularmente**.

### 3.2. Algoritmo (alto nivel)

```
Ciclo:
  1. Modelo identifica sus dominios débiles (e.g., problemas que falla).
  2. Modelo genera tareas sintéticas dirigidas a esos dominios.
  3. Modelo se auto-evalúa en esas tareas.
  4. DPO/RL sobre los resultados.
  5. Revisar qué mejoró, qué sigue débil → iterar.
```

### 3.3. Por qué importa

Self-Rewarding asume que **los prompts vienen de algún lado** (dataset, generación con few-shot). RLSR cierra el último gap: **incluso los prompts son generados por el modelo según sus necesidades**.

Esto se parece a:
- **Práctica deliberada** (Ericsson) aplicada a LLMs.
- **Active learning** donde el modelo elige sus propios ejemplos.
- **Curriculum learning automático**.

### 3.4. Limitaciones

- **Definir "debilidad"**: requiere que el modelo se auto-evalúe **correctamente**. Si ya falla la auto-evaluación, el ciclo se rompe.
- **Diversidad de tareas**: el modelo puede generar tareas **redundantes** o **estilísticamente similares**, no realmente diversas.
- **Aún experimental**: menos validado empíricamente que Self-Rewarding.

### 3.5. Implicación filosófica

RLSR apunta hacia agentes que **se auto-dirigen** en aprendizaje continuo. Esta dirección **plantea preguntas serias de safety/alignment** (cap 12).

---

## 4. Multi-Agent Evolve (2025)

### 4.1. La idea

Más allá de Self-Rewarding (un modelo, dos roles) o Meta-Rewarding (un modelo, tres roles), Multi-Agent Evolve usa **múltiples instancias separadas** del LLM, cada una con un rol cooperativo.

### 4.2. Roles

- **Generator-agent**: produce respuestas.
- **Judge-agent**: evalúa respuestas.
- **Curriculum-agent**: ajusta la dificultad de los prompts.

Cada agente es una instancia del LLM (mismo peso base inicial, pero entrenados separadamente con datos específicos a su rol).

### 4.3. Las tres rewards

```
r_total = α · r_judge + β · r_difficulty + γ · r_format
```

donde:
- `r_judge`: producido por el judge-agent.
- `r_difficulty`: producido por el curriculum-agent (premia tareas en el "zone of proximal development").
- `r_format`: estilo, longitud, formato.

### 4.4. Por qué es notable

**Domain-agnostic**: funciona en math, código, razonamiento, knowledge general. No requiere verificador externo ni etiquetas humanas.

### 4.5. Resultados

Sobre Qwen2.5-3B-Instruct, mejora consistente en math/code/reasoning/QA. Llamada de atención: funciona en **modelos pequeños** (3B) donde Self-Rewarding ingenuo regresaría (cap 08 §6 — CREAM).

### 4.6. Conexión con MARL formal

Multi-Agent Evolve es **el método más cercano** a Multi-Agent Reinforcement Learning formal en esta familia:
- Agentes separados con políticas distintas.
- Cooperación explícita.
- Rewards compartidas.

Esto refuerza el **Pilar 2** (Cooperative MARL) de la presentación. Self-Rewarding es estructural-cooperativo (un solo modelo, dos roles). Multi-Agent Evolve es **funcional-cooperativo** (múltiples instancias, roles separados).

---

## 5. Comparación de los tres métodos

| Método | Quién se mejora | Necesita prompts | Riesgo principal |
|---|---|---|---|
| **Self-Taught Evaluators** | Solo el juez | Sí (no anotados) | Sesgos del juez se cristalizan |
| **RLSR** | Política + currículo | **No** (el modelo los genera) | Tareas redundantes/limitadas |
| **Multi-Agent Evolve** | Política + juez + currículo | Iniciales mínimos | Coordinación multi-agente compleja |

---

## 6. ¿Qué dice esto sobre la dirección del campo?

Estos métodos apuntan hacia **una visión específica de IA**: agentes que aprenden continuamente sin supervisión humana post-inicialización.

Implicaciones:

### 6.1. Promesa

- Escalabilidad infinita.
- Costo bajo de iteración.
- Posibilidad de superar el nivel humano en dominios verificables.

### 6.2. Riesgos

- **Drift sin control**: sin ancla humana, el sistema puede derivar a comportamientos indeseables.
- **Optimización de objetivos mal especificados**: el modelo optimiza lo que él mismo evalúa, no necesariamente lo que es "correcto".
- **Safety/Alignment**: si el sistema se auto-mejora indefinidamente sin supervisión, **¿quién garantiza que esté alineado con valores humanos?**

> ⚠️ **Pregunta abierta importante**: ¿cómo garantizamos que un modelo que se auto-supervisa no drift hacia comportamientos indeseables? Constitutional AI (cap 03) ofrece una respuesta parcial (principios explícitos), pero **la cuestión sigue abierta**. Esto se discute en el cap 12.

---

## 7. Frase para defensa

> *"La frontera de Self-Rewarding empuja la autonomía cada vez más lejos: Self-Taught Evaluators muestra que el juez puede entrenarse autónomamente (de 75.4 a 88.3 en RewardBench, superando GPT-4 sin etiquetas humanas). RLSR cierra el último gap haciendo que el modelo también genere sus propias tareas de práctica. Multi-Agent Evolve usa múltiples instancias cooperando, acercándose a Multi-Agent RL formal y abriendo el espacio para modelos que aprenden curricularmente. Esto plantea preguntas serias de safety: si el sistema se auto-mejora sin supervisión humana, ¿cómo garantizamos alineamiento con valores humanos?"*

---

## 8. Qué viene

[12-fallos-y-criticas.md](12-fallos-y-criticas.md) — la crítica empírica organizada. Lo que **NO** funciona y por qué.

---

## Lecturas relacionadas

- Entradas Self-Taught Evaluators, RLSR, Multi-Agent Evolve en [[sources]].
- [[synthesis/pilar-2-multi-agente-cooperativo]] — framing cooperativo.
