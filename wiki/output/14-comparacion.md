# 14 — Comparación final: el espacio completo de métodos

> **Prerequisitos**: todos los capítulos anteriores.

> **Lo que sabrás al final**: una vista panorámica del campo, qué método usar en cada escenario, las tres lecturas estructurales que sostienen la narrativa de la presentación, y la frase final de cierre.

---

## 1. Las dos dimensiones de la evolución

Todo el panorama cabe en **un plano de dos ejes**:

### Eje 1 — Origen del feedback

```
Humanos              → LLM externo     → Mismo modelo     → Mismo modelo (3+ roles)
[RLHF]                 [RLAIF / CAI]      [Self-Rewarding]   [Meta-Rewarding, Multi-Agent Evolve]
```

### Eje 2 — Algoritmo de optimización

```
PPO (4 modelos)      → GRPO (3 modelos) → DPO (2 modelos)
[RLHF clásico]         [DeepSeek-R1]      [DPO, Self-Rewarding, todas las variantes]
```

Cada método cae en una **celda** del producto cruzado:

```
                |  Humanos   |  LLM ext.  |  Mismo m.  |  3+ roles
----------------+------------+------------+------------+----------
PPO (4 modelos) |   RLHF     | RLAIF/CAI  |     —      |     —
GRPO (3)        |     —      |     —      | R1-Zero    |     —
DPO (2)         | DPO offline| DPO+RLAIF  | Self-Rew   | Meta-Rew
```

> 💡 **Las casillas vacías son interesantes**: ¿por qué nadie hace PPO con mismo modelo, o GRPO con humanos, o DPO con 3+ roles? Es ejercicio mental válido para la sesión de preguntas.

---

## 2. La tabla canónica

| Aspecto | RLHF | DPO offline | RLAIF | SPIN | Self-Rewarding | Meta-Rewarding | GRPO (R1) |
|---|---|---|---|---|---|---|---|
| **Fuente de feedback** | Humanos | Humanos (offline) | LLM ext. | Self vs SFT | Mismo modelo | Mismo modelo (3 roles) | Reward verificable |
| **Reward model** | Separado | Implícito | Separado / directo | Implícito | Implícito (judge) | Implícito + meta | Función externa |
| **Algoritmo** | PPO | Pérdida cerrada | PPO/DPO | DPO iter | DPO iter | DPO iter | GRPO on-policy |
| **Modelos en memoria** | 4 | 2 | 3-4 | 2 | 2 | 2 | 3 |
| **Iterativo** | No | No | Posible | Sí | **Sí (core)** | **Sí (con meta)** | Sí (rollouts) |
| **Juez mejora con t** | No | No | No | No | **Sí (implícita)** | **Sí (explícita)** | N/A |
| **Anotación humana** | Miles | Solo inicial | Solo SFT | Solo SFT | Solo SFT | Solo SFT | Solo SFT |
| **Dominio óptimo** | Cualquiera | Cualquiera | Cualquiera | Cualquiera | Open-ended | Open-ended | Verificable |
| **Costo relativo** | Muy alto | Bajo | Medio | Medio-bajo | Medio-bajo | Medio-bajo | Medio |
| **Riesgo principal** | Costo, cuello humano | Sobreajuste | Sesgo de juez | Plateau | Self-preference, sat. | Same + complejidad | Reward hacking |

---

## 3. Guía práctica: ¿qué usar cuando?

| Escenario | Recomendación | Por qué |
|---|---|---|
| Tienes presupuesto + anotadores + necesitas control fino | **RLHF** | Más control, mejor calidad final si bien hecho |
| Tienes presupuesto limitado, dataset preferencias humanas | **DPO** | Simple, estable, casi como SFT en costo |
| Sin presupuesto anotación, tienes LLM grande externo | **RLAIF** | Escalable, mantiene mucho de RLHF |
| Sin presupuesto, sin modelo externo, instrucciones abiertas | **Self-Rewarding** | El paradigma del paper ancla |
| Self-Rewarding satura, quieres más iteraciones | **Meta-Rewarding** | Entrena el juez explícitamente |
| Tarea con reward verificable (math, code) | **GRPO / R1** | El paradigma alternativo correcto |
| Quieres reward verificable + razonamiento limpio | **SAR** | Híbrido verificable + self-aligned |
| Modelo pequeño (~7B), Self-Rewarding regresa | **CREAM** | Regularización por consistencia |
| Self-Rewarding falla en math | **Process-SRLM** | Juicio paso a paso |
| Preferencias intransitivas | **DNO** | Nash sin Bradley-Terry |
| Sin juez, solo SFT | **SPIN** | Aprovecha la distribución humana como ancla |
| Tareas multimodales con alucinación | **CSR** / **Vision-SR1** | Restricciones visuales explícitas |
| Modelo debe generar su propio currículo | **RLSR** | Auto-mejora curricular |

---

## 4. Los tres pilares (síntesis final)

La presentación argumenta que Self-Rewarding es la intersección de **tres temas del curso**. Aquí están en resumen final:

### Pilar 1 — Inverse RL implícito

**Argumento (cap 05)**:
- RLHF KL-restringido tiene solución cerrada de forma Boltzmann.
- Esa forma es MaxEnt-IRL con prior `π_ref`.
- DPO invierte la ecuación: la política codifica una reward implícita.
- Self-Rewarding aplica DPO sobre pares generados por el juez interno → IRL del juez.

**Validación empírica**: Joselowitz 2025 (IRL recupera reward de LLMs RLHF con 85% acc).

**Frase**: *"Self-Rewarding ejecuta una recuperación implícita de recompensa estructuralmente equivalente al óptimo MaxEnt-IRL con prior, donde el experto es el propio modelo en su modo juez."*

### Pilar 2 — Cooperative Multi-Agent RL

**Argumento (cap 06 §6.2, cap 07, cap 11)**:
- Self-Rewarding tiene dos roles cooperativos: generador y juez.
- Cooperación emerge: mejor juez → mejor generador → mejor juez.
- Meta-Rewarding añade tercer rol (meta-juez), acercándose a Hierarchical RL.
- Multi-Agent Evolve usa múltiples instancias, acercándose a MARL formal.

**Frase**: *"Self-Rewarding es un sistema cooperativo de dos roles donde la mejora del juez retroalimenta la mejora del generador. Meta-Rewarding y Multi-Agent Evolve generalizan a jerarquías superiores y múltiples instancias."*

### Pilar 3 — DPO en el loop

**Argumento (cap 02, cap 06)**:
- DPO es **offline** (pares fijos) → no requiere generar durante entrenamiento.
- DPO es **estable** → el loop no se rompe por inestabilidad de PPO.
- DPO es **iterable** → cada iteración es bien definida.
- Sin DPO, el loop sería impráctico (PPO iterativo es ingobernable).

**Frase**: *"DPO no es solo un truco de optimización — es lo que hace posible el loop iterativo limpio de Self-Rewarding. Su forma cerrada, derivada de la equivalencia con MaxEnt-IRL, captura precisamente lo que el problema necesita."*

---

## 5. La narrativa completa de la presentación (esqueleto)

```
1. El problema: alinear LLMs requiere RL post-entrenamiento.

2. La evolución del feedback:
   RLHF (humanos) → RLAIF (LLM ext.) → Self-Rewarding (mismo) → Meta-Rewarding (jerarquía)

3. La evolución del algoritmo:
   PPO (4 modelos) → GRPO (3) → DPO (2)

4. Self-Rewarding (el paper ancla):
   - Loop iterativo: generar → juzgar → DPO → repetir.
   - El juez mejora junto con el generador.
   - 3 iteraciones sobre Llama 2 70B superan Claude 2, Gemini Pro, GPT-4 0613.

5. Tres lecturas teóricas (los pilares):
   - IRL implícito (DPO ↔ MaxEnt-IRL).
   - Cooperative MARL (generador + juez).
   - DPO en el loop (offline, estable, iterable).

6. Crítica honesta:
   - Saturación a 3 iteraciones (Wang 2025).
   - Self-preference bias amplificado.
   - Falla en math (Zhang 2025).
   - Inconsistencia interna (SCIR).
   → Mitigaciones: Meta-Rewarding, Temporal SR, Process-SRLM, etc.

7. El paradigma paralelo:
   - DeepSeek-R1 + GRPO: reward verificable.
   - Comportamientos emergentes.
   - Complementario a Self-Rewarding.

8. Frontera abierta:
   - Self-Taught Evaluators, RLSR, Multi-Agent Evolve.
   - Safety: ¿cómo controlar auto-mejora sin supervisión?
```

---

## 6. La frase de cierre

Para terminar la presentación:

> *"Self-Rewarding Language Models es la intersección de tres pilares del RL: Inverse RL implícito (DPO codifica una reward via la equivalencia MaxEnt-Boltzmann), Cooperative Multi-Agent RL (dos roles cooperando en un mismo modelo), y DPO como técnica de optimización que viabiliza el loop iterativo. Tres iteraciones sobre Llama 2 70B superan a GPT-4 0613, Claude 2 y Gemini Pro en AlpacaEval 2.0, sin un solo nuevo anotador humano. Sus límites — saturación a 3 iteraciones, self-preference bias, falla en razonamiento verificable — son conocidos, documentados, y abordados por una familia emergente de métodos: Meta-Rewarding, Temporal Self-Rewarding, Process-SRLM, SCIR, CREAM. El paradigma paralelo de DeepSeek-R1 con GRPO confirma que la dirección — auto-mejora autónoma sin humanos post-SFT — es la frontera del campo. La pregunta abierta no es '¿funciona?' (la evidencia dice que sí, en regímenes específicos), sino '¿cómo controlamos esta autonomía?'"*

---

## 7. Mapa visual sugerido para la slide central

```
                        AUTO-MEJORA SIN HUMANOS
                                |
              +-----------------+-----------------+
              |                                   |
       FEEDBACK SUBJETIVO                  FEEDBACK VERIFICABLE
       (LLM-as-a-Judge)                    (función objetiva)
              |                                   |
       Self-Rewarding                          R1-Zero
       Meta-Rewarding                         (GRPO + RL puro)
       SPIN, SPPO, DNO, DICE                  
       CREAM, SCIR, ScPO
              |                                   |
       Ecosistema textual                  Ecosistema reasoning
                                                    
   Frontera común: cómo controlar la auto-mejora sin supervisión
```

---

## 8. Lo que un compañero debería llevarse

Al final de la presentación, un compañero del curso debería poder decir:

1. **Self-Rewarding es** un loop donde el modelo se evalúa y se mejora iterativamente con DPO.
2. **Funciona en** instruction-following abierto, con resultados validados en Llama 2 70B.
3. **No funciona en** razonamiento técnico estricto (math, code) sin modificaciones.
4. **Es IRL implícito** vía la equivalencia DPO ↔ MaxEnt-IRL.
5. **Es Cooperative MARL** vía los dos roles del modelo.
6. **DPO es central** porque hace el loop iterable y estable.
7. **Tiene límites documentados** (saturación, self-preference bias) y mitigaciones (Meta-Rewarding, Temporal SR).
8. **El paradigma paralelo** R1 + GRPO confirma la dirección general.
9. **Hay preguntas abiertas** importantes sobre safety/alignment.

---

## 9. Tres preguntas que pueden surgir y cómo responderlas

### Q: "¿Cuál de los tres pilares es el más importante?"

A: "DPO en el loop, porque sin él los otros dos no operarían. IRL implícito es la lectura teórica más elegante y conecta con el syllabus del curso. Cooperative MARL es la lectura que mejor caracteriza la dinámica del sistema. Los tres juntos dan el marco completo."

### Q: "Si me limito a 10 minutos, ¿qué incluyo?"

A: "Tesis del trabajo + pipeline del loop (con el diagrama) + un pilar (recomendado IRL si la audiencia es RL) + una crítica (saturación con Wang 2025) + frase de cierre. Aproximadamente 6 slides."

### Q: "¿Es Self-Rewarding 'el futuro' del alineamiento?"

A: "Es **una de las dos direcciones** convergentes (la otra es reward verificable + GRPO). Probablemente el futuro híbrido combine ambas: reward verificable cuando exista, self-rewarding cuando no. La frontera abierta es cómo controlar la autonomía sin supervisión continua."

---

## 10. Cierre

Este es el final del material `output/`. Tienes ahora:

- 14 capítulos pedagógicos cubriendo todo el campo.
- Conexión explícita con tu background NLP.
- Argumentos defensivos para cada parte de la presentación.
- Las críticas honestas que un profesor podría plantear.
- Frases de defensa preparadas.

**Buena suerte en la presentación.**

---

## Lecturas relacionadas

- [[synthesis/comparacion-metodos]] — la tabla canónica en versión wiki.
- [[synthesis/pilar-1-irl-implicito]], [[synthesis/pilar-2-multi-agente-cooperativo]], [[synthesis/pilar-3-dpo-en-loop]] — los tres pilares en versión densa.
- [[synthesis/diagnostico-fallos]] — críticas en versión densa.
