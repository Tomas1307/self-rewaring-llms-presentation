# 08 — Variantes y self-play del ecosistema

> **Prerequisitos**: capítulos 02, 04, 06.

> **Lo que sabrás al final**: cómo se relacionan ocho métodos relevantes (SPIN, SPPO, DNO, DICE, CREAM, SCIR, ScPO, SER), qué problema específico ataca cada uno, y cuándo elegir uno sobre otro.

---

## 1. El espacio del ecosistema

Self-Rewarding (Yuan 2024) abrió la línea. La comunidad respondió con muchos refinamientos. Para no perdernos, los agrupo por **qué problema concreto resuelven**:

| Grupo | Problema | Métodos |
|---|---|---|
| Mitigación de sesgo/saturación | Saturación o sesgo acumulado del juez | CREAM, SCIR |
| Reward implícito como bootstrap | Eliminar el LLM-as-a-Judge externo | DICE |
| Generalización del modelo Bradley-Terry | Preferencias intransitivas | DNO, SPPO |
| Reward model que se auto-mejora | Mejorar el RM, no solo la política | SER |
| Self-play vs. SFT como ancla | Self-improvement sin LLM-as-Judge | SPIN |
| Consistency como señal | Tareas con respuesta correcta clara | ScPO |

Vamos uno a uno.

---

## 2. SPIN — Self-Play Fine-Tuning (Chen et al. 2024, ICML)

> 📐 **SPIN en una frase**: *self-play donde el modelo actual aprende a distinguir respuestas humanas reales de las generadas por su versión anterior.*

### 2.1. Idea

Tienes un dataset SFT `D = {(x, y_human)}`. Quieres extraer **más valor** del mismo dataset sin nuevas anotaciones.

```
Iteración t:
  1. Generar respuestas con M_{t-1} (versión anterior, "oponente").
  2. Entrenar M_t a distinguir respuestas humanas (y_w = y_human) de
     respuestas del oponente (y_l = M_{t-1}(x)).
  3. Aplicar DPO sobre estos pares.
```

### 2.2. ¿Por qué funciona?

Conceptualmente, SPIN dice: "Cualquier respuesta que el modelo previo genera es **subóptima** comparada con la humana. Aprender a separarlas refina el modelo".

Esto es **un pseudo-self-play**: el modelo "compite" contra su versión anterior con la respuesta humana como referencia.

### 2.3. Garantía teórica

Los autores demuestran que **el óptimo global se alcanza solo cuando `M_t` matches la distribución humana**. No es solo una heurística — tiene fundamento.

### 2.4. Diferencias con Self-Rewarding

| Aspecto | Self-Rewarding | SPIN |
|---|---|---|
| `y_w` viene de | LLM-as-a-Judge selecciona mejor de N | **Humano** (dataset SFT) |
| `y_l` viene de | LLM-as-a-Judge selecciona peor de N | **Modelo anterior** |
| Necesita juez | Sí | **No** |
| Necesita SFT | Sí | Sí (más esencial — es la única fuente de señal) |
| Mejora juez | Sí (efecto colateral) | No hay juez |
| Ancla | `π_ref` se mueve | `y_w` siempre es humano (ancla fija) |

### 2.5. ¿Cuándo usar SPIN?

- Tienes un buen dataset SFT y quieres exprimirlo más.
- No tienes un juez confiable.
- Tu modelo no puede evaluarse bien a sí mismo (modelo muy pequeño).

### 2.6. Limitación

- **No supera la distribución humana**. La "respuesta humana" es el techo, no puedes mejorar más allá.
- Self-Rewarding en principio puede ir más allá (con caveats).

---

## 3. SPPO — Self-Play Preference Optimization (Wu et al. 2024, NeurIPS)

> 📐 **SPPO en una frase**: *self-play formalizado como juego de suma constante de dos jugadores; busca el equilibrio de Nash.*

### 3.1. Idea

SPPO conceptualiza el alineamiento como un **juego de suma constante**:
- Dos políticas `π_A` y `π_B` "compiten" generando respuestas.
- Hay una función de preferencia `P(y_A ≻ y_B | x)` (otorgada por un juez).
- El óptimo es el **equilibrio de Nash**: la política preferida sobre cualquier otra.

### 3.2. Algoritmo

```
Iteración t:
  1. Generar pares con M_{t-1}.
  2. Evaluar pares con un juez (típicamente PairRM, modelo pequeño 0.4B).
  3. Update con regla SPPO (similar a DPO pero con términos extra).
```

### 3.3. Garantías

SPPO **prueba convergencia a Nash** bajo supuestos razonables. DPO no tiene esta garantía.

### 3.4. Resultados

- Mistral-7B-Instruct-v0.2 + SPPO → 28.5% AlpacaEval 2 LC (vs ~22% para DPO).
- Llama-3-8B + SPPO → 38.77% LC.

### 3.5. Caveats

- Usa **PairRM como juez externo**, no es estrictamente self-rewarding.
- En la comparación head-to-head con Self-Rewarding, SPPO gana ligeramente, pero **usa más información** (un juez externo).

### 3.6. ¿Cuándo usar SPPO?

- Quieres garantías de convergencia formal.
- Tienes acceso a un juez pequeño/barato (PairRM).
- Tu setup es "self-play con juez fijo externo" (no juez = modelo).

---

## 4. DNO — Direct Nash Optimization (Rosset et al. 2024, Microsoft)

> 📐 **DNO en una frase**: *Nash sobre preferencias generales, sin asumir Bradley-Terry.*

### 4.1. El problema que ataca

DPO **asume Bradley-Terry**: `P(y_w ≻ y_l) = σ(r(y_w) − r(y_l))`. Esto implica:
- Preferencias **transitivas**: si A ≻ B y B ≻ C, entonces A ≻ C.
- Preferencias **monótonas**.

En la realidad, las preferencias humanas pueden ser **intransitivas** o **cíclicas**.

### 4.2. Idea

DNO no asume forma específica de `P`. Trata `P(y_A ≻ y_B | x)` como **función arbitraria** y busca el equilibrio de Nash.

### 4.3. Algoritmo

Es **on-policy** (genera durante entrenamiento) con un objetivo de **regression** que aproxima Nash iterativamente:

```
1. Generar pares (y, y') desde M_t.
2. Obtener P(y ≻ y' | x) de un oracle (humano, LLM, etc.).
3. Entrenar M_{t+1} para igualar P:
   M_{t+1}(y|x) ∝ M_t(y|x) · exp(λ · winrate(y, M_t))
4. Iterar.
```

### 4.4. Garantía clave

**Mejora monótona entre iteraciones** — `M_{t+1}` es siempre al menos tan bueno como `M_t` según la preferencia.

### 4.5. Resultados

Orca-2.5-7B + DNO + GPT-4 como oracle → 33% LC vs GPT-4-Turbo en AlpacaEval 2.

### 4.6. Diferencia con Self-Rewarding

DNO **requiere un oracle** (humano o GPT-4). No es self-rewarding puro. Pero sus garantías teóricas son la fortaleza del método.

---

## 5. DICE — DPO ImpliCit rEwards (Chen C. et al. 2024, ICLR 2025)

> 📐 **DICE en una frase**: *usa la reward implícita de DPO como juez interno para generar nuevos pares.*

### 5.1. La observación elegante

DPO entrena una política. Esa política tiene una **reward implícita**:
```
r̂_θ(x, y) = β · log( π_θ(y|x) / π_ref(y|x) )
```

DICE pregunta: ¿podemos **reutilizar** esta reward para generar nuevos pares de preferencia?

### 5.2. Algoritmo

```
Iteración 0: entrenar DPO sobre dataset humano D_0.
Iteración t > 0:
  1. Generar respuestas con M_t.
  2. Calcular r̂_M_t(x, y) para cada respuesta.
  3. Construir pares: y_w = max r̂, y_l = min r̂.
  4. Entrenar DPO sobre nuevos pares.
```

### 5.3. ¿Por qué es interesante?

**Elimina el LLM-as-a-Judge**. La reward implícita actúa como juez. Cero overhead de prompting de un juez.

### 5.4. Adaptaciones

Para evitar sesgos conocidos (e.g., length bias):
- **Regularización de longitud**.
- **Experience replay**: mantener pares antiguos en el dataset.

### 5.5. Resultados

`>8%` mejora en AlpacaEval 2 LC para varios modelos base.

### 5.6. Limitación

- La reward implícita **hereda los sesgos** del DPO inicial. Si el dataset humano original tenía length bias, DICE lo amplifica.
- No mejora explícitamente "la capacidad de juzgar" como Self-Rewarding sí hace.

---

## 6. CREAM — Consistency Regularized Self-Rewarding (Wang Z. et al. 2024, ICLR 2025)

> 📐 **CREAM en una frase**: *Self-Rewarding pero atenuando señales de preferencia poco confiables.*

### 6.1. El problema que ataca

En modelos pequeños (~7B), Self-Rewarding **regresa** tras varias iteraciones por **sesgo acumulado** del juez (etiquetas sobre-confiadas).

### 6.2. Idea

Si la decisión `y_w ≻ y_l` **no es consistente** entre iteraciones consecutivas, esa señal es ruidosa. **Atenúa** la señal de preferencia en función de la consistencia.

### 6.3. Algoritmo

```
Para cada par (x, y_w, y_l):
  consistencia = ¿M_{t-1} y M_t coinciden en preferir y_w sobre y_l?
  peso_par = f(consistencia)  # mayor consistencia, mayor peso
  
L_DPO_modificada += peso_par · L_DPO(par)
```

### 6.4. Resultados

Self-Rewarding sin CREAM regresa después de 3 iter en modelos 7B. **Con CREAM, continúa mejorando.**

### 6.5. Relevancia para la presentación

CREAM es **evidencia directa** de que la saturación tiene una causa diagnosticable (sesgo acumulado por etiquetas sobre-confiadas) y de que existen mitigaciones. Importante para el cap 12.

---

## 7. SCIR — Self-Consistency of Internal Rewards (Zhou et al. 2025)

> 📐 **SCIR en una frase**: *exige que múltiples señales internas de reward coincidan; filtra pares inconsistentes.*

### 7.1. Idea

Un LLM tiene **varias señales internas** que pueden actuar como reward:

1. **Juez generativo**: score producido por LLM-as-a-Judge con rúbrica.
2. **Reward DPO implícito**: `β · log(π/π_ref)`.
3. **Likelihood directo**: `log P(y | x)`.

**Insight**: si estas señales **discrepan** sobre un par `(y_w, y_l)`, ese par es **poco confiable**. Filtrarlo o atenuarlo.

### 7.2. Algoritmo

```
Para cada par candidato (y_w, y_l):
  s_judge = LLM-as-Judge score
  s_dpo = reward DPO implícita
  
  Si s_judge prefiere y_w AND s_dpo prefiere y_w:
    incluir en dataset DPO con peso completo.
  Si discrepan:
    atenuar o descartar.
```

### 7.3. Por qué es notable

Documenta que **el modelo es internamente inconsistente**. El juez generativo (modo prompting) y la reward DPO (modo gradient implícito) **no siempre coinciden**. Esto es un fallo conceptual de Self-Rewarding.

### 7.4. Relevancia

Es **una de las críticas más profundas** a Self-Rewarding clásico. Aparece en el cap 12.

---

## 8. ScPO — Self-Consistency Preference Optimization

> 📐 **ScPO en una frase**: *en tareas con respuesta única correcta, usa la mayoría entre muestras como señal de preferencia.*

### 8.1. El problema

En tareas con respuesta correcta (math, lógica), **el LLM-as-a-Judge no puede evaluar bien** si el modelo no sabe la respuesta. El juez es tan ignorante como el generador.

### 8.2. Idea

Usar **consistencia** entre múltiples cadenas de razonamiento como señal:

```
Para un problema math:
  1. Generar K cadenas de razonamiento (con sampling).
  2. Extraer la respuesta final de cada cadena.
  3. La respuesta final más común (mayoría) → es probablemente la correcta.
  4. Las cadenas que llegan a esa respuesta → y_w (positivas).
  5. Las cadenas que llegan a respuestas distintas → y_l (negativas).
```

### 8.3. Por qué funciona

Mayoría sobre samples diversos tiende a converger a la respuesta correcta (self-consistency, Wang 2022). Si el modelo "sabe" la respuesta pero con probabilidad media, la mayoría la encuentra.

### 8.4. Limitación

- **Falla cuando el modelo no sabe la respuesta**. La mayoría puede converger a una respuesta **incorrecta** consistente.
- **No funciona en tareas open-ended** sin "respuesta correcta única".

---

## 9. SER — Self-Evolved Reward Learning (Huang et al. 2024)

> 📐 **SER en una frase**: *el reward model se auto-mejora etiquetando muestras donde tiene alta confianza.*

### 9.1. El problema

RLHF clásico tiene un **reward model congelado** que no aprende. SER ataca esto desde el ángulo del RM.

### 9.2. Idea

```
Iteración t:
  1. RM_t etiqueta nuevas muestras donde tiene alta confianza
     (margen grande entre y_w y y_l).
  2. RM_{t+1} se entrena sobre dataset humano + auto-etiquetas confiables.
  3. RM_{t+1} es estrictamente mejor que RM_t (en expectativa).
```

### 9.3. Garantías teóricas

Convergencia a un RM cuasi-óptimo incluso con datos humanos limitados, bajo supuestos sobre el ruido.

### 9.4. Posición en el ecosistema

SER es **complementario** a Self-Rewarding. Mientras Self-Rewarding mejora la política (y como efecto el juez), SER mejora **explícitamente el reward model**. Combinarlos es una dirección abierta.

---

## 10. Tabla resumen de las 8 variantes

| Método | Año | Lo que añade | Problema que resuelve |
|---|---|---|---|
| **SPIN** | 2024 | Self-play con SFT como ancla | Extraer más valor de SFT existente sin juez |
| **SPPO** | 2024 | Nash con juez externo pequeño | Garantías de convergencia |
| **DNO** | 2024 | Nash sobre preferencias arbitrarias | Preferencias intransitivas |
| **DICE** | 2024 | Reward DPO implícita como juez | Eliminar LLM-as-Judge |
| **CREAM** | 2024 | Regularización por consistencia inter-iter | Sesgo acumulado en modelos pequeños |
| **SCIR** | 2025 | Filtro por consistencia de señales internas | Inconsistencia entre rewards internos |
| **ScPO** | 2024 | Mayoría entre muestras como preferencia | Tareas con respuesta correcta |
| **SER** | 2024 | Reward model que se auto-mejora | RM congelado |

---

## 11. Cómo encajan en la narrativa de la presentación

Estos métodos no son **competidores** de Self-Rewarding — son **descendientes** que abordan limitaciones específicas. Si la presentación necesita decir "Self-Rewarding tiene problemas pero la comunidad sabe abordarlos", estos son los ejemplos:

- ¿Saturación? → **CREAM**, **Meta-Rewarding** (cap 07).
- ¿Inconsistencia interna? → **SCIR**.
- ¿Falla en math? → **ScPO**, **Process-SRLM** (cap 09).
- ¿Preferencias intransitivas? → **DNO**.
- ¿Reward model congelado? → **SER**.
- ¿Sin juez? → **DICE**.
- ¿Garantías formales? → **SPPO**, **DNO**.

---

## 12. Frase para defensa

> *"Tras Self-Rewarding emerge un ecosistema de métodos que extienden o critican sus límites. Métodos de mitigación de sesgo como CREAM y SCIR atacan la saturación documentando sus causas. Generalizaciones teóricas como DNO y SPPO eliminan la suposición de Bradley-Terry y dan garantías de Nash. Variantes algorítmicas como DICE eliminan incluso el LLM-as-Judge. La línea no es una sola técnica sino una familia de aproximaciones a un mismo objetivo: auto-mejora robusta sin feedback humano."*

---

## 13. Qué viene

[09-self-rewarding-razonamiento.md](09-self-rewarding-razonamiento.md) — el subcampo de Self-Rewarding aplicado a razonamiento (donde el método clásico falla).

---

## Lecturas relacionadas

- [[summaries/wang-2025-temporal-sr]] — relacionado con saturación.
- Entradas en [[sources]] §5 (Métodos adyacentes) para cada uno.
