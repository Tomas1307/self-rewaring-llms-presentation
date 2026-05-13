# 09 — Self-Rewarding para razonamiento

> **Prerequisitos**: capítulos 04, 06, 08.

> **Lo que sabrás al final**: por qué Self-Rewarding clásico falla en math, qué adaptaciones funcionan (Process-SRLM, LaTRO, SAR, CoVo, Self-Rew-Correction), y cómo este subcampo conecta con GRPO/R1.

---

## 1. El problema central: el juez ignorante

Self-Rewarding asume que el juez sabe distinguir bien entre `y_w` y `y_l`. En instruction-following abierto (escribe un poema, resume este texto), esto es razonable: hay "espacio subjetivo" donde el juez puede tener razón.

En **razonamiento matemático estricto** (resuelve esta ecuación, prueba este teorema), esta suposición se rompe:

> ⚠️ **El juez tiene acceso al mismo modelo que produjo el error**. Si `M_t` no sabe resolver un problema, **`M_t` como juez tampoco sabe verificarlo correctamente**.

Resultado: en math, Self-Rewarding tradicional **degrada** la performance del modelo. Zhang et al. 2025 lo documenta cuantitativamente.

---

## 2. Las cuatro estrategias para math

| Estrategia | Métodos | Idea |
|---|---|---|
| **Step-wise judging** | Process-SRLM | Evaluar el razonamiento paso a paso, no la respuesta final |
| **Consistency entre samples** | ScPO, CoVo | Múltiples cadenas → mayoría/consistencia → señal de calidad |
| **Self-correction durante inferencia** | Self-Rew-Correction (Xiong) | Modelo razona + verifica + corrige iterativamente |
| **Híbrido verificable + self** | SAR | Combinar reward verificable (binario) con self-aligned reward (granular) |

Vamos cada una.

---

## 3. Process-SRLM — Process-based Self-Rewarding (Zhang et al. 2025, ACL)

### 3.1. La intuición

En math, **el proceso importa tanto como el resultado**. Una respuesta puede ser correcta por casualidad (suerte) o el razonamiento intermedio puede tener errores que se compensan. Evaluar solo el resultado final es **señal pobre**.

Process-SRLM evalúa **paso a paso**.

### 3.2. Algoritmo

```
Para cada problema math:
  1. Generar N cadenas de razonamiento {chain_1, ..., chain_N}.
     Cada chain es una secuencia de pasos.
  2. Para cada chain:
     a. Dividir en pasos individuales.
     b. Para cada paso: el modelo evalúa "¿este paso es correcto?".
        (LLM-as-a-step-judge)
     c. Score de la chain = función agregada de scores de pasos.
  3. Construir pares (y_w, y_l) usando scores agregados.
  4. DPO sobre pares.
```

### 3.3. Por qué funciona mejor que respuesta-final

- **Detecta errores tempranos**: una chain con buen resultado pero malos pasos intermedios recibe score bajo.
- **Señal más granular**: en lugar de un bit por respuesta (correcto/incorrecto), tienes K bits por respuesta (uno por paso).
- **Más alineado con cómo "deben razonar" los humanos**: paso a paso, justificando cada paso.

### 3.4. Limitación

El **juez-paso** sigue siendo el mismo modelo. Si el modelo no sabe que cierto paso es incorrecto, no lo detecta. Mejor que evaluar respuesta final, pero no perfecto.

---

## 4. ScPO — Self-Consistency como preferencia (ya en cap 08)

Recordatorio rápido (detalle en cap 08 §8):

```
Para un problema:
  1. Generar K cadenas con sampling.
  2. La respuesta final más común (mayoría) → probablemente correcta.
  3. Chains con esa respuesta → positivas.
  4. Chains con respuestas minoritarias → negativas.
  5. DPO.
```

> 💡 **Por qué funciona**: si el modelo "sabe" la respuesta pero con probabilidad media, la mayoría sobre samples la encuentra. Es **wisdom-of-crowds** aplicado a un solo modelo.

> ⚠️ **Cuándo falla**: si el modelo está **confiadamente equivocado**, la mayoría converge a la respuesta incorrecta y ScPO la refuerza. Es una técnica para tareas donde el modelo es "ligeramente competente", no para tareas que no sabe hacer.

---

## 5. LaTRO — Language Models are Hidden Reasoners (Salesforce 2024)

### 5.1. Idea

LaTRO conceptualiza el **razonamiento como variable latente**. Formalización tipo variacional:

```
y = respuesta final
z = cadena de razonamiento (latente)
x = problema

P(y | x) = E_z[ P(y | x, z) · P(z | x) ]
```

El modelo tiene una **distribución sobre cadenas de razonamiento** y la respuesta es resultado de muestrear esa distribución.

### 5.2. El "self-reward"

LaTRO usa el propio modelo para evaluar qué tan **plausible** es una cadena. Define:
```
r(z, x, y) = P_M(y correcto | x, z)
```

— "¿qué tan probable es que esta cadena de razonamiento `z` lleve a una respuesta correcta para el problema `x`?"

Esto es **un reward inducido por el LLM mismo**. No requiere verificador externo.

### 5.3. Optimización

Optimización conjunta:
```
max_θ  E_{z ~ q(z|x)}[ r_θ(z, x, y) ]
```

con `q` aprendido también del propio LLM.

### 5.4. Resultado

Mejora en GSM8K y otros benchmarks reasoning. Conceptualmente notable porque:
- Da **fundamento variacional** al self-rewarding en razonamiento.
- Comprime el coste de inferencia (cadenas más cortas pero más informativas).
- "Razonadores ocultos": los LLMs pre-entrenados ya **codifican** información sobre la calidad del razonamiento, solo hay que extraerla.

---

## 6. Self-Rewarding Correction for Math (Xiong et al. 2025)

### 6.1. Idea

Entrenar al modelo a **razonar y auto-corregir en tiempo de inferencia**, sin verificador externo.

### 6.2. Framework en dos etapas

**Etapa 1 — Síntesis de trayectorias de auto-corrección**:
```
Para un problema:
  1. Modelo genera intento de solución.
  2. Modelo se evalúa: "¿es correcto?"
  3. Si no: modelo revisa.
  4. Repetir hasta convergencia o límite.
```

Esta trayectoria multi-paso (genera + auto-evalúa + corrige) se usa como **dato de entrenamiento** mediante rejection sampling: se quedan las que llegan a respuesta correcta.

**Etapa 2 — RL con reglas**:
- Reward = cuán bien sigue el patrón de auto-corrección.
- Aplicar RL (no necesariamente DPO; pueden ser otros algos) para reforzar el comportamiento.

### 6.3. Resultado

Llama-3 y Qwen-2.5 entrenados así igualan sistemas con verificadores externos en math benchmarks.

### 6.4. Por qué es interesante

**Internaliza la verificación** en el comportamiento del modelo. En lugar de necesitar un verificador externo durante despliegue, el modelo mismo se auto-verifica.

---

## 7. SAR — Self-Aligned Reward (2025)

### 7.1. El problema que ataca

GRPO y otros métodos con reward verificable producen:
- **Respuestas demasiado largas**: el modelo aprende que más pasos = más probabilidad de acertar.
- **Razonamiento "shallow"**: muchos tokens pero pocos saltos cognitivos.

### 7.2. Idea

**Híbrido**: combinar reward verificable (binario, robusto) con un self-aligned reward (granular, sutil).

```
r_total = α · r_verificable + (1−α) · r_self_aligned
```

donde:
- `r_verificable`: 1 si correcto, 0 si no.
- `r_self_aligned`: el modelo mismo asigna un score continuo de "calidad del proceso".

### 7.3. Por qué funciona

- `r_verificable` previene reward hacking (no puedes ganar sin respuesta correcta).
- `r_self_aligned` da granularidad: respuestas correctas con razonamiento más limpio reciben más reward que respuestas correctas con razonamiento desordenado.

### 7.4. Resultado

Reasoners **más cortos y más precisos** en MATH-500 y AMC/AIME.

---

## 8. CoVo — Consistency + Volatility (Zhang K. et al. 2025)

### 8.1. Idea geométrica

CoVo extrae reward de **patrones en las trayectorias de razonamiento**:

- **Consistencia**: estados intermedios convergen hacia la respuesta final del propio chain.
- **Volatilidad**: oscilación entre respuestas candidatas durante la trayectoria.

Las **respuestas correctas** muestran alta consistencia + baja volatilidad. Las **incorrectas** oscilan.

### 8.2. Algoritmo

Para una chain:
```
1. En cada paso intermedio, predecir la respuesta final del chain.
2. Calcular:
   - consistencia = correlación entre predicciones intermedias y respuesta final
   - volatilidad = varianza de predicciones intermedias
3. r_CoVo = consistencia − λ · volatilidad
4. Combinar con bono de curiosidad (KL para fomentar exploración).
```

### 8.3. Resultados

Qwen2.5 y Llama3.2 (3B-7B) con CoVo igualan **GRPO con verificador externo** sin usar reward model ni etiquetas.

### 8.4. Por qué es elegante

No requiere LLM-as-Judge (que falla en math). No requiere reward model. La señal viene de **propiedades geométricas** de las trayectorias del modelo mismo.

---

## 9. Comparación de las 5 técnicas

| Método | Cómo evalúa | Cuándo brilla | Cuándo falla |
|---|---|---|---|
| **Process-SRLM** | Paso a paso con LLM-judge | El modelo conoce reglas pero comete errores intermedios | El modelo no sabe la respuesta |
| **ScPO** | Mayoría entre samples | El modelo es "ligeramente competente" | Confiado y equivocado |
| **LaTRO** | Reward inducido por likelihood | Cadenas latentes informativas | Modelo base muy pobre |
| **Self-Rew-Correction** | Auto-corrección durante inferencia | Se quiere internalizar verificación | Tasks sin estructura de verificación |
| **SAR** | Híbrido verificable + self | Hay reward verificable disponible | No hay verificador binario |
| **CoVo** | Geometría de trayectorias | Cadenas largas, estructura emergente | Cadenas cortas |

---

## 10. La conexión con GRPO/R1 (preview cap 13)

DeepSeek-R1 (cap 13) es **el caso opuesto**: en lugar de buscar señales internas para math (como los métodos arriba), usa **reward verificable explícita** (correcto/incorrecto) + GRPO. Cero LLM-as-Judge.

¿Cuál es mejor?

| Tarea | Mejor opción |
|---|---|
| Math con respuesta verificable computacionalmente | GRPO + reward verificable (R1) |
| Math sin verificador automático fiable | Process-SRLM o SAR |
| Razonamiento abierto (e.g., legal, ético) | LaTRO o Self-Rew-Correction |
| Trade-off respuesta correcta + razonamiento limpio | SAR |

Son complementarios. La presentación puede usar esta dicotomía como ejemplo de cómo el campo está bifurcándose:
- **Reward verificable** (GRPO, R1) — donde la verdad es accesible.
- **Self-reward** (los del cap 09) — donde la verdad no se puede verificar automáticamente.

---

## 11. Frase para defensa

> *"Self-Rewarding clásico falla en razonamiento matemático estricto porque el juez es ignorante en la misma medida que el generador. Cuatro estrategias mitigan esto: Process-SRLM evalúa paso a paso, ScPO usa mayoría entre muestras, LaTRO trata el razonamiento como variable latente, Self-Rew-Correction internaliza la verificación. SAR combina reward verificable y self-aligned. CoVo deriva reward de patrones geométricos en trayectorias. Donde hay reward verificable computable (math, código), GRPO + R1 son superiores. Donde no, los métodos self-reward refinados son la frontera."*

---

## 12. Qué viene

[10-self-rewarding-multimodal.md](10-self-rewarding-multimodal.md) — extensión a vision-language (CSR, Vision-SR1).

---

## Lecturas relacionadas

- Entradas en [[sources]] §8 y §5 para cada paper.
- [[synthesis/comparacion-metodos]] §4.bis donde aparecen agrupados.
