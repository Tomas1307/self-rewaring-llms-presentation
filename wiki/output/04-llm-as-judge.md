# 04 — LLM-as-a-Judge: el paradigma del juez LLM

> **Prerequisitos**: [03-rlaif.md](03-rlaif.md).

> **Lo que sabrás al final**: cómo se construye un juez LLM, qué rúbricas se usan, qué sesgos tiene (todos documentados), cómo se mitigan, y por qué este componente es **el cuello de botella crítico** de Self-Rewarding.

---

## 1. ¿Qué es "LLM-as-a-Judge"?

> 📐 **LLM-as-a-Judge**: paradigma donde un LLM evalúa el output de otro LLM (o del mismo LLM) mediante un prompt estructurado, produciendo un score numérico o una preferencia entre opciones.

Es el componente que **convierte calidad subjetiva en señal numérica**. Sin él, no hay forma de generar pares de preferencia automáticamente.

> 💡 **Conceptualmente**: es un **reward model construido por prompting** (en lugar de fine-tuning). En vez de entrenar una red para predecir scores, le pides al LLM que produzca un score directamente.

---

## 2. Las dos formas básicas del juez

### 2.1. Scoring absoluto

Le das al juez **una sola respuesta** y le pides un score (típicamente 1-5 o 1-10):

```
Prompt: "Evalúa la siguiente respuesta del 1 al 5 según utilidad,
        coherencia y precisión."

Respuesta a evaluar: "..."

Output esperado:
- Razonamiento: ...
- Score: 4
```

**Ventaja**: simple, paralelizable, fácil de comparar entre prompts.

**Desventaja**: calibración entre llamadas inconsistente. El juez puede dar 4 a una respuesta hoy y 3 mañana a la misma respuesta.

### 2.2. Pairwise comparison

Le das al juez **dos respuestas** y le pides elegir la mejor:

```
Prompt: "Lee la pregunta del usuario y las dos respuestas candidatas.
        Decide cuál es mejor."

Respuesta A: "..."
Respuesta B: "..."

Output:
- Razonamiento: ...
- Veredicto: A | B
```

**Ventaja**: más consistente (es más fácil comparar que calificar absolutamente).

**Desventaja**: cuadrático en número de comparaciones si quieres rankear todo.

### 2.3. La rúbrica de Yuan 2024 (Self-Rewarding)

Self-Rewarding usa una rúbrica **aditiva de 5 puntos**:

```
- 1 punto: la respuesta es relevante y aporta algo de información
           aunque sea incompleta.
- 1 punto: la respuesta atiende la mayoría del pedido del usuario.
- 1 punto: la respuesta atiende el pedido principal de manera útil.
- 1 punto: la respuesta es clara, completa y bien escrita.
- 1 punto: la respuesta es de alta calidad, perspectiva experta,
           sin información extraneous.

Score total: suma (0-5).
```

> 💡 **Por qué aditiva**: cada criterio es **independiente** y se suma. Esto reduce ambigüedad: el juez puede señalar "sí cumple 1, 2 y 3, pero no 4 y 5" en lugar de tener que decidir "esto es un 3.5".

---

## 3. Anatomía del prompt del juez

Un prompt de juez bien diseñado tiene:

1. **Rol del sistema**: "Eres un evaluador imparcial..."
2. **Criterios explícitos**: lista de qué evaluar.
3. **Formato de respuesta**: cómo debe responder el juez.
4. **El input a evaluar**: prompt + respuesta(s).
5. **Trigger de razonamiento**: "Razona antes de dar veredicto".

Ejemplo real (estilo Yuan 2024):

```
[SYSTEM]
Review the user's question and the corresponding response using
the additive 5-point scoring system described below.

Points are accumulated based on the satisfaction of each criterion:
- Add 1 point if the response is relevant and provides some info
  related to the user's inquiry, even if incomplete or contains
  irrelevant content.
- Add another point if the response addresses a substantial portion...
- (etc.)

[USER]
Question: {prompt}
Response: {response}

After examining, briefly justify your total score. Conclude with:
"Score: <total>"

[ASSISTANT]
```

> ⚠️ **Detalle de ingeniería**: el LLM **escribe el razonamiento antes del score**. Esto se llama **chain-of-thought reasoning** y reduce errores. Sin razonamiento previo, el juez "dispara" un score y suele ser ruidoso.

---

## 4. Sesgos documentados

Aquí está el **catálogo de problemas** del juez LLM, todos con evidencia empírica.

### 4.1. Position bias

Si presentas dos respuestas A/B, el juez tiende a preferir **una posición específica** sistemáticamente. Algunos modelos prefieren A, otros B.

**Evidencia**: Zheng et al. (2023, *Judging LLM-as-a-Judge with MT-Bench*) cuantifica este sesgo. GPT-4 muestra preferencia significativa por la primera posición en ciertos formatos.

**Mitigación**: **swap evaluation** — evaluar dos veces con orden invertido y promediar.

### 4.2. Length bias

El juez prefiere respuestas **más largas**, incluso si la información extra es redundante o irrelevante.

**Evidencia**: documentado en múltiples papers; es **el sesgo más difícil de mitigar**.

**Mitigación**: AlpacaEval 2.0 introduce **length-controlled win rate** que ajusta por longitud post-hoc.

### 4.3. Self-enhancement bias (auto-preferencia)

El juez prefiere outputs **estilísticamente similares** a los que él mismo produciría. En el caso extremo de Self-Rewarding, esto significa que **el juez prefiere outputs del propio modelo** que entrena.

**Evidencia cuantitativa**: arXiv:2410.21819 (*Self-Preference Bias in LLM-as-a-Judge*) mide que GPT-4 prefiere consistentemente sus propios outputs, y vincula el sesgo a **baja perplexity** sobre textos familiares.

**Por qué importa para Self-Rewarding**: el juez = generador en Self-Rewarding. Este sesgo se **amplifica máximamente** — el modelo refuerza su propia distribución preferida.

**Mitigación**: cuando sea posible, evaluar con un **juez externo** distinto al modelo entrenado. Yuan 2024 lo hace evaluando con GPT-4 Turbo aunque el entrenamiento sea con Llama 2 70B como juez.

### 4.4. Style over substance

El juez puede preferir respuestas **bien formateadas** (listas, negritas, estructura) por encima de respuestas correctas pero "feas".

**Mitigación**: dejar la rúbrica explícita en "no premies formato; premia contenido".

### 4.5. Confidence bias

Respuestas que **suenan seguras** son preferidas sobre respuestas que admiten incertidumbre, incluso cuando la incertidumbre es la actitud correcta.

### 4.6. Sesgos culturales/lingüísticos

El juez hereda los sesgos del modelo base. Si el modelo base es predominantemente inglés/occidental, los criterios reflejan esa cultura.

---

## 5. ¿Qué tan confiable es un juez LLM comparado con humanos?

Zheng et al. (2023) mide esto con MT-Bench y Chatbot Arena:

- **Acuerdo entre GPT-4 y humanos**: ~80-85% (similar al acuerdo entre humanos diferentes, que es ~80-83%).
- **Acuerdo entre Claude/GPT-3.5 y humanos**: 70-75%.
- **Acuerdo entre humanos diferentes** (baseline): ~80%.

> 💡 **Lectura**: GPT-4 como juez es **estadísticamente indistinguible de un anotador humano**. Esta es la base empírica que legitima RLAIF/Self-Rewarding.

> ⚠️ **Caveat**: "indistinguible" significa el mismo nivel de acuerdo, no que el juez sea óptimo. Los humanos también tienen sesgos, y el juez los hereda + añade los suyos.

---

## 6. Self-Rewarding y el juez

Self-Rewarding hace algo notable: **el juez es el mismo modelo que se entrena**. Tres consecuencias:

### 6.1. El juez mejora con el tiempo

Cada iteración de DPO actualiza los pesos del modelo. Como el modelo **es** el juez, el juez también mejora. Esto es **el hallazgo central** de Yuan 2024 — la capacidad de juicio crece junto con la capacidad de generación.

> 💡 **¿Por qué mejora el juez si la pérdida DPO solo entrena la generación?** Sutil pero importante: el prompt del juez es **una continuación de texto** que el LLM produce. Cuando entrenas con DPO, mueves los pesos del LLM. Esos mismos pesos producen los juicios. **No hay separación física**.

### 6.2. Self-preference bias se amplifica

Ya mencionado en §4.3. Si el juez prefiere sus propios outputs, y el modelo es a la vez juez y generador, el sesgo es máximo. **Este es el riesgo central de Self-Rewarding.**

### 6.3. El juez no tiene ancla externa

En CAI, la constitución actúa como ancla. En RLAIF con juez externo, el juez externo es la ancla. En Self-Rewarding, **no hay ancla externa** — el juez se mueve con el modelo.

Esto puede llevar a "drift": el modelo y el juez convergen juntos a una distribución preferida, sin garantía de que sea la distribución óptima en el mundo real.

---

## 7. Mitigaciones específicas para Self-Rewarding

Cuatro estrategias documentadas:

### 7.1. Filtrado por consistencia

**Idea**: solo usar pares donde múltiples señales internas coincidan.

- **SCIR** (Zhou 2025): exige consistencia entre juez generativo + DPO implícito + likelihood.
- **CREAM** (Wang Z. 2024): regulariza por consistencia entre iteraciones consecutivas.

### 7.2. Meta-juez

**Idea**: el modelo no solo juzga respuestas, también **juzga sus propios juicios**.

- **Meta-Rewarding** (Wu 2024, cap 07): añade un tercer rol que evalúa la calidad de los juicios.

### 7.3. Decoupling temporal

**Idea**: usar versiones temporales distintas del modelo para generar `y_w` y `y_l`.

- **Temporal SR** (Wang 2025): `y_l` desde modelo pasado (SFT inicial), `y_w` desde modelo futuro (EMA).

### 7.4. Evaluación externa

**Idea**: para la métrica final reportada, usar un juez **distinto** al usado en entrenamiento.

- Yuan 2024 entrena con Llama 2 70B como juez, evalúa con GPT-4 Turbo (un modelo diferente).

---

## 8. El "judge score reasoning" — un detalle implementacional

Cuando le pides al juez que razone antes del veredicto, hay un detalle sutil:

```
Prompt: "Razona, luego da score."
Output: "El razonamiento es que la respuesta... [varias frases]... Score: 4"
```

¿Qué probabilidad asigna el modelo a cada token? **No es uniforme**. El modelo tiene **certidumbres distintas** según el token.

> 💡 **Implementación realista**: en lugar de tomar solo el token de "4", se puede tomar la **distribución sobre {1, 2, 3, 4, 5}** y computar el score esperado:
> ```
> score = Σ_i  i · P(token_i | contexto)
> ```
> Esto da **scores continuos** en [1, 5] en vez de discretos, lo que mejora la granularidad para construir pares de preferencia.

Yuan 2024 reporta hacer esto.

---

## 9. Cómo se eligen `y_w` y `y_l` en Self-Rewarding

Dado un prompt `x`:
1. Generar `N = 4` respuestas con sampling de temperatura ~0.7.
2. Cada respuesta evaluada con el juez (rúbrica 5 puntos aditiva, posiblemente con score continuo).
3. Tomar:
   - `y_w` = respuesta con score **máximo**.
   - `y_l` = respuesta con score **mínimo**.
4. Si todas tienen el mismo score, descartar el prompt (no hay señal).
5. El par `(x, y_w, y_l)` va al dataset de DPO.

> ⚠️ **Punto crítico**: si el modelo es muy bueno, **todas las respuestas tienen score alto** y el margen `y_w − y_l` se estrecha. Esto **destruye el gradiente DPO** (ver cap 02 §6). Es **la causa de la saturación**.

---

## 10. ¿Funciona el juez en dominios técnicos?

**No siempre.**

| Dominio | ¿El juez LLM funciona bien? |
|---|---|
| Instruction-following abierto | Sí (cuasi-humano). |
| Resumen, paráfrasis | Sí. |
| Escritura creativa | Sí. |
| Razonamiento matemático | **No** sin modificaciones. |
| Código | Parcialmente — necesita ejecución. |
| Verificación factual | Inestable — alucinaciones. |

> 💡 **Por qué falla en matemáticas**: si el modelo no sabe resolver un problema, **el modelo como juez tampoco puede verificarlo correctamente**. El juez hereda los huecos del modelo.

Soluciones para razonamiento (cap 09):
- **Process-SRLM**: juicio paso a paso, no respuesta final.
- **ScPO**: consistencia entre múltiples cadenas de razonamiento como señal.
- **GRPO + R1** (cap 13): reward verificable (correcto/incorrecto), no juez LLM.

---

## 11. Frase para fijar

> *"LLM-as-a-Judge convierte calidad subjetiva en señal numérica vía prompting. Es el corazón de RLAIF y Self-Rewarding. Sus sesgos son reales y cuantificados (position, length, self-enhancement). En Self-Rewarding el sesgo de auto-preferencia se amplifica porque juez = generador. Las mitigaciones (consistencia, meta-juez, decoupling temporal, juez externo) son las que distinguen métodos buenos de métodos colapsantes."*

---

## 12. Qué viene

[05-irl.md](05-irl.md) — Inverse Reinforcement Learning. Es el marco teórico que conecta DPO con "recuperar una reward implícita". Pilar 1 de la presentación.

---

## Lecturas relacionadas

- [[concepts/llm-as-a-judge]] — versión densa.
- [[summaries/zheng-2023-llm-as-judge]] — paper canónico con métricas de acuerdo.
- [[summaries/yuan-2024-self-rewarding]] — la rúbrica de 5 puntos en su contexto.
