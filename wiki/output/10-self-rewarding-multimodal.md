# 10 — Self-Rewarding multimodal (vision-language)

> **Prerequisitos**: capítulos 04, 06.

> **Lo que sabrás al final**: por qué Self-Rewarding necesita adaptaciones específicas para VLMs, cómo funcionan CSR y Vision-SR1, y qué problemas resuelven (alucinaciones visuales, atajos lingüísticos).

---

## 1. ¿Por qué un capítulo aparte para multimodal?

Aplicar Self-Rewarding a Vision-Language Models (VLMs) no es trivial. Hay **dos problemas específicos** que no aparecen en LLMs puros:

### 1.1. Alucinaciones visuales

El modelo dice cosas sobre la imagen que **no están en la imagen**. Ejemplo:

```
Imagen: gato en una mesa
Modelo: "Veo un gato negro sentado en una mesa de madera junto
         a un florero con tulipanes."
```

Aunque el gato esté ahí, los tulipanes y el florero **fueron inventados**. El modelo tiene un prior textual fuerte que llena gaps con cosas plausibles pero falsas.

### 1.2. Atajos lingüísticos (language shortcuts)

El modelo **ignora la imagen** y responde solo desde el prior textual. Ejemplo:

```
Imagen: una manzana roja
Pregunta: "¿De qué color es?"
Modelo: "Las manzanas suelen ser rojas o verdes." 
         ← no mira la imagen, usa prior textual
```

El modelo puede tener razón por suerte (manzanas son rojas a menudo), pero **no está usando la información visual**.

### 1.3. Self-Rewarding ingenuo amplifica ambos problemas

Si el juez es el mismo VLM y tiene los mismos sesgos, el juez:
- Premia respuestas plausibles aunque inventen detalles (alucinación).
- No penaliza respuestas que ignoran la imagen.

Resultado: Self-Rewarding ingenuo **refuerza los problemas en lugar de mitigarlos**.

---

## 2. CSR — Calibrated Self-Rewarding (Zhou Y. et al. 2024, NeurIPS)

### 2.1. La idea central

**Forzar al juez a mirar la imagen** incorporando restricciones visuales explícitas en el score.

### 2.2. El score combinado

> 📐 **Score CSR**:
> ```
> s(image, text_response) = s_text(response) + λ · s_visual(response, image)
> ```
>
> donde:
> - `s_text`: score lingüístico estándar (similar al LLM-as-Judge de Yuan 2024).
> - `s_visual`: medida de alineación entre la respuesta y la imagen (e.g., CLIP score, image-text matching, segmentación).

> 💡 **Lectura**: la respuesta no solo debe ser **buena textualmente** sino también **fiel visualmente**. Si dice "tulipanes" y la imagen no los contiene, `s_visual` baja.

### 2.3. Estrategia step-wise

CSR aplica esto **paso a paso** (similar a Process-SRLM cap 09):
- La respuesta se divide en oraciones.
- Cada oración se evalúa contra la imagen.
- Score agregado guía la construcción de pares.

### 2.4. Resultados

Mejora hasta **+7.62% promedio en 10 benchmarks multimodales**. Reduce significativamente alucinaciones visuales en LLaVA y derivados.

### 2.5. Por qué funciona

Las restricciones visuales son **ancla externa** al loop. Aunque el juez es el mismo VLM, ahora hay una señal independiente (el CLIP score, o cualquier medida visual) que **no se mueve con el modelo**. Es **el equivalente a la constitución de CAI** pero para visión.

---

## 3. Vision-SR1 — Reasoning Decomposition (2025)

### 3.1. Idea

Vision-SR1 toma una aproximación **completamente diferente**: descomponer el razonamiento multimodal en **dos fases** y derivar reward de la consistencia entre ellas.

### 3.2. Las dos fases

**Fase 1 — Visual Perception**:
```
Input: imagen + pregunta
Output: descripción visual "auto-contenida" — texto que captura todo
        lo necesario de la imagen para responder, sin volver a mirarla.
```

**Fase 2 — Language Reasoning**:
```
Input: SOLO la descripción visual de la Fase 1 (sin imagen).
Output: respuesta a la pregunta.
```

### 3.3. La señal de auto-reward

> 📐 **Self-reward de Vision-SR1**:
>
> Si la respuesta de Fase 2 (sin imagen) **coincide** con la respuesta del modelo dado imagen + pregunta directamente:
>
> ```
> r = + (señal positiva) — la percepción visual es FIEL.
> ```
>
> Si no coincide:
>
> ```
> r = − (señal negativa) — la percepción visual está incompleta o
>     se está usando atajos lingüísticos.
> ```

> 💡 **Por qué funciona**: si el modelo puede responder bien **solo con la percepción textualizada**, esa percepción captura lo importante. Si necesita "consultar" la imagen otra vez, la percepción es deficiente.

### 3.4. Combinación con GRPO

Vision-SR1 combina esta señal de auto-reward con supervisión de la respuesta final (reward verificable cuando la pregunta tiene respuesta correcta) en un setup tipo GRPO. Híbrido similar a SAR (cap 09).

### 3.5. Resultados

Reduce alucinaciones y atajos lingüísticos **sin supervisión externa** ni etiquetas costosas de pixel-level alignment.

---

## 4. Comparación CSR vs Vision-SR1

| Aspecto | CSR | Vision-SR1 |
|---|---|---|
| Estrategia | Score visual explícito (CLIP) | Descomposición percepción/razonamiento |
| Requiere modelo externo | Sí (CLIP o similar) | No |
| Granularidad | Step-wise | Two-stage |
| Problema atacado | Alucinaciones visuales | Atajos lingüísticos + alucinaciones |
| Algoritmo de actualización | DPO | GRPO |
| Reward | Continua | Combinación binaria/continua |

---

## 5. La lección general

Ambos métodos comparten una intuición:

> 💡 **En multimodal, Self-Rewarding necesita un ancla**. CSR usa CLIP como ancla. Vision-SR1 usa la consistencia entre dos fases del propio modelo como ancla interna estructural.

> ⚠️ **Sin ancla, Self-Rewarding en VLMs simplemente refuerza los priors textuales del modelo**. El modelo "aprende" a ser más confiado en sus alucinaciones.

---

## 6. ¿Por qué este capítulo importa para la presentación?

Probablemente **no** vas a incluir multimodal en las slides principales (la presentación es sobre LLMs textuales). Pero **vale tenerlo en mente** porque:

1. Muestra que **el paradigma generaliza**: Self-Rewarding no es solo para texto.
2. Muestra que **la idea de "ancla externa" reaparece** en cada extensión. CAI tenía la constitución, multimodal tiene CLIP/consistencia. Esto es **una lección estructural**.
3. Permite responder preguntas del estilo "¿esto funciona en otros dominios?" con datos concretos.

---

## 7. Conexión con los pilares

- **Pilar 1 (IRL implícito)**: en multimodal, la reward implícita debe codificar **ambos** dominios (texto + visión). Los métodos extienden la dualidad MaxEnt-IRL ↔ DPO al espacio multimodal.
- **Pilar 2 (Cooperative)**: roles de "perceptor" y "razonador" en Vision-SR1 son una **descomposición multi-agente** del rol generador.
- **Pilar 3 (DPO en loop)**: CSR usa DPO estándar; Vision-SR1 mezcla con GRPO. La maquinaria de optimización se adapta al setup.

---

## 8. Frase para defensa (si la pregunta surge)

> *"Self-Rewarding ingenuo en VLMs amplifica alucinaciones y atajos lingüísticos porque el juez hereda los priors textuales del modelo. Dos soluciones distintas: CSR (Zhou 2024) introduce restricciones visuales explícitas (CLIP) como ancla externa; Vision-SR1 (2025) descompone el razonamiento en percepción + razonamiento lingüístico, derivando reward de la consistencia entre ambas fases. Ambos métodos comparten la lección: Self-Rewarding necesita un ancla que no se mueva con el modelo."*

---

## 9. Qué viene

[11-self-rewarding-autonomia.md](11-self-rewarding-autonomia.md) — los métodos que llevan la autonomía al extremo: el modelo genera tareas, multi-agente co-evolución, jueces que se auto-entrenan.

---

## Lecturas relacionadas

- Entradas CSR y Vision-SR1 en [[sources]].
