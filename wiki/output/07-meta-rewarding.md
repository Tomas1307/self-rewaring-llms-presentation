# 07 — Meta-Rewarding (Wu et al. 2024)

> **Prerequisitos**: capítulos 02, 04, 06.

> **Lo que sabrás al final**: qué problema específico de Self-Rewarding resuelve Meta-Rewarding, cómo funciona el rol del meta-juez, los resultados (mejora 22.9% → 39.4% en AlpacaEval 2), y por qué conecta con Hierarchical RL.

---

## 1. El problema que Meta-Rewarding ataca

Self-Rewarding (cap 06) tiene una limitación específica que Wu et al. 2024 identifica con precisión:

> **El entrenamiento DPO mejora la capacidad de generación, pero la mejora del juez es solo un efecto colateral.**

Recordemos del cap 06: cuando aplicas DPO sobre los pares de preferencia, los pesos del modelo se actualizan. Esos mismos pesos producen los juicios → el juez "mejora colateralmente".

**Pero**:
- El gradiente DPO **no apunta** explícitamente a mejorar el juicio.
- El gradiente DPO apunta a **subir la probabilidad de `y_w`** y **bajar la de `y_l`**, lo que es generación.
- La mejora del juez emerge **por compartir pesos**, no por entrenamiento dirigido.

Resultado: tras pocas iteraciones, **el juez se satura más rápido que el generador**. La calidad de la señal de entrenamiento deja de crecer, y el sistema completo plateau.

> 💡 **Meta-Rewarding en una frase**: *"Si el problema es que el juez no se entrena explícitamente, entrenémoslo explícitamente — usando otro nivel de juicio."*

---

## 2. La idea central — añadir un meta-juez

Self-Rewarding tiene dos roles:
- **Actor** (generador): produce respuestas.
- **Juez**: evalúa respuestas.

Meta-Rewarding añade un **tercero**:
- **Actor**: produce respuestas. (igual)
- **Juez**: evalúa respuestas. (igual)
- **Meta-Juez**: evalúa los juicios del juez.

```
Respuestas → Juez → juicios → Meta-Juez → preferencias sobre juicios
                                                      ↓
                                          DPO sobre preferencias de juicios
                                                      ↓
                                          Mejora la capacidad de juzgar
```

> 💡 **Lectura clave**: ahora hay **dos señales de entrenamiento DPO** corriendo en paralelo:
> 1. **Preferencias de actor**: pares de respuestas `(y_w, y_l)` → mejora el generador.
> 2. **Preferencias de juez**: pares de juicios `(j_w, j_l)` → mejora el juez.

---

## 3. Cómo se obtienen pares de juicios

Para evaluar un juicio, el meta-juez necesita **dos juicios contrastantes** sobre la misma respuesta.

### Procedimiento

```
Para cada (x, y):
  1. El modelo genera K juicios distintos sobre y (sampling con temperatura
     en el prompt del juez).
  2. Los K juicios son textos: cadenas como "El razonamiento es...
     Score: 4".
  3. El meta-juez recibe los K juicios y elige cuál es mejor.
```

> 💡 **Sutileza importante**: el meta-juez evalúa **el juicio**, no la respuesta. Le importa:
> - ¿El razonamiento del juicio es coherente?
> - ¿El score refleja realmente lo que dice el razonamiento?
> - ¿La rúbrica se aplica consistentemente?
>
> No le importa si la respuesta `y` es buena. Le importa si el **juicio sobre `y`** es bueno.

### Prompt del meta-juez

Aproximadamente:

```
[SYSTEM]
You are a meta-evaluator. Given a question, a response, and two
judgments of that response, decide which judgment is better.

A good judgment:
- Provides coherent reasoning before scoring.
- Applies the rubric consistently.
- The reasoning supports the assigned score.
- Avoids common biases (length, position, style).

[USER]
Question: {x}
Response: {y}

Judgment A: {j_A}
Judgment B: {j_B}

Which is better? Respond with "A" or "B".
```

---

## 4. El doble entrenamiento DPO

> 📐 **Pérdida combinada de Meta-Rewarding**:
> ```
> L_total = L_DPO_actor + λ · L_DPO_judge
> ```
> donde:
> - `L_DPO_actor` se computa sobre pares `(x, y_w, y_l)` (igual que Self-Rewarding).
> - `L_DPO_judge` se computa sobre pares `(x, y, j_w, j_l)` — donde `j_w, j_l` son juicios y el modelo aprende a producir `j_w` con mayor probabilidad.

Notar que el "input" para el judge-DPO es **el prompt del juez completo** (que incluye `x` y `y`), y la "respuesta" es **el juicio textual** (con razonamiento + score).

> ⚠️ **Práctico**: en el paper se hace **alternando** actor-DPO y judge-DPO en pasos consecutivos, no estrictamente como una suma. El efecto es similar pero la implementación es más estable.

---

## 5. Resultados clave

### 5.1. El headline number

| Modelo | AlpacaEval 2 LC Win Rate vs GPT-4 0314 |
|---|---|
| Llama-3-8B-Instruct (baseline) | 22.9% |
| Self-Rewarding sobre Llama-3-8B-Instruct (4 iter) | ~28% |
| **Meta-Rewarding sobre Llama-3-8B-Instruct (4 iter)** | **39.4%** |
| GPT-4-0314 (referencia) | 50% |

> 💡 **Lo notable**: Meta-Rewarding sobre un modelo de **8B** alcanza ~40% vs GPT-4. Self-Rewarding sobre el mismo modelo no llega a 30%. La diferencia es atribuible al meta-juez.

### 5.2. La capacidad de juicio crece más

En benchmarks de "calidad del juez" (acuerdo con humanos), Meta-Rewarding muestra crecimiento sostenido a través de 4 iteraciones, mientras Self-Rewarding satura a 2-3.

> 💡 **Esto confirma la hipótesis**: si entrenas el juez explícitamente, el juez mejora más y el loop completo mejora más.

### 5.3. Arena-Hard

Otro benchmark: 20.6% → 29.1%. Mejora similar.

---

## 6. La conexión con Hierarchical RL

Hierarchical Reinforcement Learning estudia agentes que operan a **múltiples niveles de abstracción**:

- **Nivel bajo**: ejecutar acciones primitivas.
- **Nivel alto**: decidir qué subgoal perseguir.

En Meta-Rewarding hay análogos:

| Nivel | Quién | Qué hace |
|---|---|---|
| Bajo | Actor | Genera tokens (acciones primitivas). |
| Medio | Juez | Evalúa respuestas (sub-objetivos). |
| Alto | Meta-juez | Evalúa juicios (meta-criterios). |

> 💡 **Lectura**: Meta-Rewarding es **un sistema cooperativo multi-nivel** donde cada nivel evalúa al inmediato inferior. Esto es estructuralmente Hierarchical RL en su modo cooperativo.

Esta conexión es **el refuerzo del Pilar 2** (Cooperative Multi-Agent) en la presentación. Self-Rewarding solo tenía dos niveles (actor, juez); Meta-Rewarding añade el tercero y se acerca más a un MARL formal.

---

## 7. ¿Y por qué no continuar con un meta-meta-juez?

Pregunta natural: ¿podemos añadir un cuarto nivel? ¿Un meta-meta-juez que evalúa los juicios del meta-juez?

**Respuesta práctica**: sí, conceptualmente. Pero:

1. **Sin ancla externa, el nivel más alto tiene los mismos problemas que el original**. Quien evalúa al meta-juez es... el mismo modelo. Los sesgos se acumulan.
2. **Diminishing returns**: cada nivel añade complejidad pero la mejora marginal decrece.
3. **No reportado empíricamente**: Meta-Rewarding se queda en 3 niveles porque es lo que funciona.

> 💡 **Esto conecta con el límite filosófico** de Self-Rewarding: **sin información externa, el sistema tiene un techo**. Más niveles internos no resuelven el problema fundamental — solo lo retrasan.

---

## 8. Trade-offs respecto a Self-Rewarding

### Pros

- Mejora más allá de 3 iteraciones.
- Calidad del juez crece sostenidamente.
- Conecta más naturalmente con frameworks multi-agente.

### Contras

- **Más cómputo por iteración**: hay que generar K juicios por respuesta + correr el meta-juez.
- **Más datos sintéticos**: cada iteración produce más texto.
- **Más complejidad de implementación**: dos DPOs paralelos.
- **No resuelve self-preference bias**: el meta-juez sigue siendo el mismo modelo.

---

## 9. ¿Resuelve todos los problemas de Self-Rewarding?

No. Resuelve **uno específico**: la saturación del juez. Pero:

- **Self-preference bias**: persiste. El meta-juez puede preferir juicios estilísticamente familiares.
- **Falla en razonamiento técnico**: persiste. Si el modelo no sabe resolver math, no puede ni juzgar respuestas ni meta-juzgar juicios sobre math.
- **Drift acumulativo**: persiste. `KL(M_t || M_0)` sigue creciendo.
- **Reward hacking**: posible. El modelo puede aprender a producir juicios "que el meta-juez prefiere" sin que sean mejores.

Otras direcciones (cap 08, 09, 12) atacan estos problemas con técnicas distintas.

---

## 10. ¿Cuándo usar Meta-Rewarding sobre Self-Rewarding?

| Situación | Recomendación |
|---|---|
| Tienes 3 iteraciones de cómputo y suficientes datos | Self-Rewarding está bien. |
| Quieres iterar 5-10 veces y mantener la mejora | Meta-Rewarding. |
| Tu modelo base es pequeño (~7B) | Meta-Rewarding ayuda más (juez débil necesita más entrenamiento). |
| Tu modelo base es grande (~70B) | Self-Rewarding ya funciona razonablemente; Meta-Rewarding añade marginalmente. |
| Tarea: razonamiento matemático | **Ni uno ni otro**; usa Process-SRLM o GRPO (caps 09, 13). |

---

## 11. Frase para defensa

> *"Self-Rewarding mejora el generador y el juez, pero solo el generador se entrena explícitamente; el juez mejora por compartir pesos. Meta-Rewarding identifica esto como la causa de la saturación a 3 iteraciones, y añade un tercer rol — el meta-juez — que entrena la capacidad de juzgar directamente. Sobre Llama-3-8B-Instruct, mejora el win rate en AlpacaEval 2 de 22.9% a 39.4%. Conecta naturalmente con Hierarchical RL al introducir niveles de abstracción cooperativos."*

---

## 12. Qué viene

[08-variantes-self-play.md](08-variantes-self-play.md) — el ecosistema de variantes que extienden Self-Rewarding desde otros ángulos: SPIN, SPPO, DNO, DICE, CREAM, SCIR, ScPO, SER.

---

## Lecturas relacionadas

- [[summaries/wu-2024-meta-rewarding]] — resumen denso.
- [[concepts/self-rewarding-loop]] — algoritmo base.
- [[synthesis/pilar-2-multi-agente-cooperativo]] — framing cooperativo extendido.
- [[synthesis/diagnostico-fallos]] §2.1 — el problema de saturación.
