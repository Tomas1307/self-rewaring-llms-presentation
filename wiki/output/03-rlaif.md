# 03 — RLAIF: Reinforcement Learning from AI Feedback (y Constitutional AI)

> **Prerequisitos**: [01-rlhf.md](01-rlhf.md), [02-dpo.md](02-dpo.md).

> **Lo que sabrás al final**: por qué se inventó RLAIF, cómo funciona el pipeline, qué es Constitutional AI y por qué importa para Self-Rewarding, en qué difiere de Self-Rewarding (la frontera es más fina de lo que parece).

---

## 1. El problema que RLAIF resuelve

RLHF tiene un problema concreto: **los humanos son caros y lentos**. Cada round de mejora del modelo requiere miles de anotaciones nuevas. Multiplica por iteraciones y semanas-mes de espera, y tienes una limitación práctica grave.

Hay un segundo problema más conceptual: **los humanos no escalan a tareas superhumanas**. Si tu modelo es mejor que un humano promedio en una tarea, los anotadores ya no pueden evaluar bien.

**RLAIF resuelve esto reemplazando al humano por un LLM**.

> 💡 **Idea clave**: dado que los LLMs grandes ya son razonablemente buenos juzgando texto, podemos usarlos como "anotadores automáticos" en lugar de humanos. Esto convierte cada anotación en una llamada de API (o un forward pass), en vez de un humano frente a una pantalla por minutos.

---

## 2. RLAIF en una frase

> 📐 **RLAIF en una línea**:
> "El mismo pipeline de RLHF, pero las preferencias para entrenar el reward model las genera un **LLM juez** en lugar de humanos."

---

## 3. Pipeline canónico

### Versión "RM intermedio" (la clásica)

```
[Modelo SFT]
       ↓
   Para cada prompt: generar K respuestas con π_SFT
       ↓
   LLM juez (modelo externo) compara y produce pares (y_w, y_l)
       ↓
   Con esos pares: entrenar Reward Model r̂
       ↓
   PPO con r̂ (igual que RLHF clásico)
       ↓
[Modelo alineado]
```

Cambios respecto a RLHF:
- **Anotadores humanos → LLM juez** (típicamente más grande o más capaz que el modelo a entrenar).
- Todo lo demás (RM, PPO, KL) **igual**.

### Versión "direct RLAIF"

Salta el reward model:

```
[Modelo SFT]
       ↓
   Generar respuestas
       ↓
   LLM juez asigna directamente un score numérico s(x, y)
       ↓
   PPO usando s(x, y) como reward (sin entrenar RM separado)
```

Esto ahorra una etapa y un modelo en memoria. Es la versión más común en práctica reciente.

### Variante "RLAIF + DPO"

Lo más simple: el juez produce pares, los usas directamente con DPO. Sin reward model ni PPO. Esta es estructuralmente lo que hace Self-Rewarding, pero con juez externo.

---

## 4. Diseño del juez LLM

El juez recibe un **prompt estructurado** con la rúbrica. Forma típica:

```
System: Eres un evaluador imparcial. Evalúa las respuestas según
los siguientes criterios:
- Utilidad: ¿la respuesta atiende la pregunta del usuario?
- Honestidad: ¿es factualmente correcta?
- Seguridad: ¿evita contenido dañino?

User prompt: [el prompt original]

Respuesta A: [y_A]
Respuesta B: [y_B]

Tarea: razona brevemente y luego responde con "A" o "B" según
cuál sea mejor.

Output:
- Razonamiento: ...
- Veredicto: A | B
```

Hay muchos detalles de ingeniería del prompt: ordenación aleatoria de A/B (para evitar **position bias**), uso de chain-of-thought antes del veredicto, prompt en formato XML para parsear de forma fiable, etc.

Veremos el paradigma del juez en detalle en [04-llm-as-judge.md](04-llm-as-judge.md).

---

## 5. Constitutional AI — la implementación más influyente

**Constitutional AI (CAI)**, de Anthropic (Bai et al. 2022), fue **la prueba de concepto** que mostró que RLAIF funciona a escala. Es el método con el que se entrena Claude.

### 5.1. La intuición

RLHF tiene tres problemas que CAI aborda:

1. **Preferencias humanas implícitas y opacas**: los anotadores aplican criterios propios sin documentar.
2. **Costo de anotación**.
3. **Sesgos no controlados**.

CAI dice: **hagamos los criterios explícitos**, escribámoslos como una "constitución", y usemos al LLM mismo para aplicarlos.

### 5.2. Las dos fases

#### Fase SL (Supervised Learning) — auto-crítica y auto-revisión

```
Para cada prompt potencialmente problemático:
  1. Modelo helpful genera respuesta inicial (puede ser dañina).
  2. Seleccionar un principio aleatorio de la constitución.
  3. Modelo se auto-critica: "Identifica formas en las que tu respuesta
     viola el principio: [principio]".
  4. Modelo se auto-revisa: "Reescribe eliminando los problemas".
  5. Repetir 2-4 varias veces con principios distintos.
  6. La respuesta final revisada se usa como demostración para SFT.
```

**Output**: dataset `(prompt, respuesta_revisada)` que fine-tunea el modelo. Llamemos al resultado `π_SFT-CAI`.

#### Fase RL (RLAIF propiamente)

Sobre `π_SFT-CAI`:

```
Para cada prompt:
  1. Generar pares (y_A, y_B).
  2. El modelo (prompted con un principio aleatorio) evalúa
     cuál de las dos cumple mejor el principio.
  3. Acumular preferencias AI-generadas.
  4. Entrenar reward model (o usar DPO directamente).
  5. Aplicar RL.
```

### 5.3. La constitución

Lista de principios en lenguaje natural. Ejemplos (sintetizados del paper):

- *"Selecciona la respuesta que sea menos dañina, ofensiva o tóxica."*
- *"¿Qué respuesta evita más explícitamente apoyar contenido ilegal, no ético o inmoral?"*
- *"Selecciona la respuesta que un sabio, ético y sensato consejero daría."*
- *"Compara las respuestas en términos de ayuda en una situación de crisis emocional."*

Decenas de principios. **En cada paso de auto-crítica o evaluación se selecciona uno al azar** para evitar sobreajustar a un principio único.

> 💡 **Propiedad crítica**: la constitución es **auditable y editable**. Si Anthropic decide cambiar un principio, edita la lista y re-entrena. Contraste con RLHF, donde cambiar la "política implícita" requiere re-anotar miles de comparaciones.

### 5.4. Por qué funciona

Tres condiciones que evitan que CAI sea solo "el modelo se evalúa a sí mismo a ciegas":

1. **Principios explícitos**: el juez aplica un criterio nombrado, no su distribución preferida.
2. **Principio aleatorio por paso**: distintos principios capturan distintas dimensiones de calidad → el sesgo se promedia.
3. **Modelo helpful preexistente**: el que critica ya es razonable. No es un modelo random.

---

## 6. Ventajas y limitaciones

### Ventajas

- **Escalabilidad masiva**: 0 costo de anotación humana en cada round.
- **Consistencia**: el juez aplica los mismos criterios siempre.
- **Velocidad**: ciclos de iteración órdenes de magnitud más rápidos que RLHF.
- **Trazabilidad** (en CAI): principios explícitos auditables.

### Limitaciones

#### Sesgos del juez

El LLM juez tiene sesgos sistemáticos:
- **Self-enhancement**: prefiere outputs estilísticamente similares a los que él mismo produciría.
- **Length bias**: prefiere respuestas largas.
- **Position bias**: en pairwise, prefiere consistentemente A o B según el orden.

Detalle en [04-llm-as-judge.md](04-llm-as-judge.md).

#### Amplificación de errores

Si el juez tiene una limitación sistemática (e.g., entiende mal una categoría), esa limitación se **propaga** al modelo entrenado. Y como el juez es congelado, no aprende a corregirse.

#### Dependencia de un modelo capaz

Necesitas un juez **bueno**. Si tu juez es pequeño/débil, los pares serán ruidosos y el modelo entrenado será pobre.

---

## 7. La relación con Self-Rewarding (importante)

Aquí hay una pregunta sutil que la mayoría de presentaciones se saltan, pero **deberías saber para defender bien**.

### 7.1. La frontera entre RLAIF y Self-Rewarding

Self-Rewarding (Yuan 2024, cap 06) es **estructuralmente** un caso de RLAIF donde:
- El juez es **el mismo modelo** que se entrena.
- El loop se itera formalmente: `M_0 → M_1 → M_2 → ...`.
- Cada iteración usa DPO sobre pares generados por el modelo actual.

Pero **¿la idea de "el juez es el mismo modelo"** es nueva de Yuan 2024?

**No.** Lee et al. 2024 (RLAIF sistemático) ya reportaba que **RLAIF logra auto-mejora cuando el LLM juez es del mismo tamaño que la política, o incluso el mismo checkpoint**. La frontera entre RLAIF y Self-Rewarding **no es nítida**.

### 7.2. Qué aporta Self-Rewarding sobre RLAIF

Tres cosas:

1. **Formalización**: hace explícito que el juez puede ser el mismo modelo.
2. **Iteración**: define un loop formal `M_t → M_{t+1}` con notación clara.
3. **Evidencia empírica**: muestra que iterar 3 veces sobre Llama 2 70B supera a GPT-4 0613, Claude 2, Gemini Pro en AlpacaEval 2.0.

> ⚠️ **Para defensa académica honesta**: NO digas "Self-Rewarding inventó la auto-mejora con juez = modelo". Dilo así:
>
> *"Self-Rewarding **formaliza e iteriza** una idea ya presente en RLAIF (Lee 2024), añadiendo evidencia empírica a gran escala y el componente clave de que el juez también mejora con cada iteración."*

---

## 8. Constitutional AI y Self-Rewarding: ancestro y descendiente

Estructuralmente:

| | Constitutional AI | Self-Rewarding |
|---|---|---|
| Origen del criterio | Constitución explícita escrita por humanos | Rúbrica del prompt LLM-as-a-Judge |
| Diversidad de criterios | Decenas de principios randomizados | Rúbrica de 5 puntos fija |
| Iteración del modelo | Una pasada SL + una RL | Loop formal `M_t → M_{t+1}` |
| Quién mejora con iter | Solo el generador | Generador **y** juez |
| Riesgo de drift | Bajo (constitución ancla fija) | Alto (juez se mueve con el modelo) |

> 💡 **Una forma de leerlo**: CAI es el **ancestro estructural** de Self-Rewarding. La constitución actúa como **ancla externa** que Self-Rewarding sacrifica para ganar la mejora del juez.

---

## 9. RLAIF como conexión multi-agente cooperativo

Aunque RLAIF opera con un solo flujo de entrenamiento, **puede leerse como sistema de dos roles cooperativos**:

- **Generador**: produce respuestas. Objetivo: que el juez las prefiera.
- **Juez**: evalúa respuestas. Objetivo: producir señales útiles para entrenamiento.

Estos roles cooperan: la mejora del generador produce respuestas más informativas que ayudan al juez, y un juez bien calibrado da señales que llevan al generador a mejor calidad.

Esta lectura cooperativa es uno de los tres pilares de la presentación (cap 06 lo desarrolla en detalle). En RLAIF puro, el juez es congelado, así que **solo el generador aprende**. En Self-Rewarding, ambos aprenden.

---

## 10. Resultados empíricos clave

Lee et al. (2024, ICML) estableció:

- RLAIF logra rendimiento **comparable a RLHF** en tareas de:
  - Resumen.
  - Diálogo útil.
  - Diálogo inofensivo.
- En algunos benchmarks, **iguala o supera** a RLHF.
- Auto-mejora con juez = checkpoint funciona (la frontera con Self-Rewarding).

Estos resultados son los que dieron crédito al campo y permitieron papers como Self-Rewarding.

---

## 11. Variantes y trabajos relacionados

- **Constitutional AI** (Bai 2022) — el progenitor (CAI).
- **RLAIF sistemático** (Lee 2023, arXiv:2309.00267) — formalización empírica a escala.
- **Direct RLAIF** — variante sin RM intermedio.
- **Self-Taught Evaluators** (Wang T. 2024, cap 11) — el juez se auto-entrena sin etiquetas humanas.
- **Anthropic's RLAIF iterativo** (Constitutional AI v2) — versiones más recientes.

---

## 12. Frase para fijar

> *"RLAIF reemplaza al anotador humano por un LLM juez. Constitutional AI es la implementación canónica: usa una lista de principios explícitos para que el LLM se critique y revise. La frontera con Self-Rewarding es porosa: ya en RLAIF el juez puede ser el mismo modelo. Self-Rewarding formaliza la iteración y muestra que el juez también puede mejorar."*

---

## 13. Qué viene

[04-llm-as-judge.md](04-llm-as-judge.md) — el paradigma del juez LLM en detalle, sus sesgos y cómo se mitigan. Material directamente aplicable a Self-Rewarding.

---

## Lecturas relacionadas

- [[concepts/rlaif]] — versión densa.
- [[concepts/constitutional-ai]] — fases SL + RL en detalle.
- [[summaries/zheng-2023-llm-as-judge]] — sesgos del juez.
