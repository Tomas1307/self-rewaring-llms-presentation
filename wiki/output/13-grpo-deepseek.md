# 13 — GRPO y DeepSeek-R1: el paradigma alternativo

> **Prerequisitos**: capítulo 00 (PPO), 06.

> **Lo que sabrás al final**: qué es GRPO y por qué simplifica PPO, qué hace DeepSeek-R1-Zero notable, qué son los "comportamientos emergentes", y por qué es complementario (no competidor) a Self-Rewarding.

---

## 1. La otra rama de la evolución

La presentación tiene dos ejes (recordatorio del cap 14):

```
Eje "origen del feedback":
  Humanos → LLM externo → Mismo modelo → Mismo modelo (3 roles)
  [RLHF] → [RLAIF/CAI] → [Self-Rewarding] → [Meta-Rewarding]

Eje "algoritmo de optimización":
  PPO (4 modelos) → GRPO (3 modelos) → DPO (2 modelos)
```

Self-Rewarding está en `(mismo modelo, DPO)`. DeepSeek-R1 está en `(reward verificable, GRPO)`. **Son rincones distintos** del espacio de soluciones.

Pero comparten **la filosofía profunda**: auto-mejora sin humanos post-inicialización.

---

## 2. ¿Qué es GRPO?

**Group Relative Policy Optimization**, propuesto por Shao et al. (2024, *DeepSeekMath*, arXiv:2402.03300).

### 2.1. El problema que ataca

Recordatorio del cap 00 (PPO): se necesitan **4 modelos en memoria**:

| Modelo | Función |
|---|---|
| Política `π_θ` | Se entrena |
| Política de referencia `π_ref` | Ancla KL |
| Reward model `r̂` | Asigna reward |
| **Critic `V_φ`** | Estima `V(s)` para advantage |

El **critic** es otra red del tamaño del LLM, costosa de mantener. GRPO se pregunta: **¿podemos prescindir del critic?**

### 2.2. La idea

En lugar de estimar `V(s)` con una red separada, **muestrear `G` respuestas** al mismo prompt y usar **su distribución de rewards como baseline**.

```
Para cada prompt x:
  1. Muestrear G respuestas {y_1, ..., y_G} de π_θ_old.
  2. Calcular reward r_i = r(x, y_i) para cada una.
  3. Normalizar dentro del grupo:
       A_i = (r_i − mean(r_1..G)) / std(r_1..G)
  4. Usar A_i como advantage en el objetivo PPO clippado.
```

**El grupo es el baseline.** Sin red de value adicional.

### 2.3. El objetivo de GRPO

> 📐 **Pérdida GRPO** (esencialmente PPO con advantage grupal):
> ```
> L_GRPO(θ) = E_{x, {y_i}~π_θ_old} [
>     (1/G) Σ_i  min(
>         ρ_i(θ) · A_i,
>         clip(ρ_i(θ), 1−ε, 1+ε) · A_i
>     )
> ]  −  β · KL( π_θ || π_ref )
> ```
>
> donde:
> - `ρ_i(θ) = π_θ(y_i|x) / π_θ_old(y_i|x)`
> - `A_i` = advantage normalizada (paso 3)
> - `ε` = clip threshold (igual que PPO)

### 2.4. Función de reward — el detalle crucial

GRPO requiere una **función de reward computable directamente**, no un reward model entrenado. Esto lo limita a:

- **Matemáticas**: `r = 1` si la respuesta final es correcta, `r = 0` si no.
- **Código**: `r = 1` si el código pasa los tests, `r = 0` si no.
- **Formato**: penalizaciones por no seguir patrones específicos.

> 💡 **Crucial**: la reward es **verificable y objetiva**. No es un LLM-as-Judge (subjetivo). Esto hace que el setup sea **diferente** al de Self-Rewarding.

---

## 3. DeepSeek-R1 — el hito empírico

DeepSeek-AI lanzó R1 en enero 2025 (arXiv:2501.12948). Hito por varias razones.

### 3.1. R1-Zero — RL puro sin SFT

La variante más notable: **entrenada solo con RL (GRPO) sin SFT inicial**.

```
Modelo base: DeepSeek-V3-Base (preentrenamiento estándar)
              ↓
        Sin SFT, sin cold-start
              ↓
   GRPO con reward verificable
              ↓
        R1-Zero (capacidades de razonamiento emergentes)
```

> 💡 **Hito**: tradicionalmente se asumía que el SFT cold-start era necesario para "darle al modelo la habilidad de seguir instrucciones" antes de RL. R1-Zero muestra que **no es estrictamente necesario** — el RL puede generar instruction-following emergente si la señal de reward es lo suficientemente densa.

### 3.2. Comportamientos emergentes

Durante el entrenamiento de R1-Zero emergen comportamientos **no programados explícitamente**:

1. **Auto-verificación**: el modelo escribe pasos de verificación de su propia respuesta antes de comprometerse.

   ```
   "La respuesta podría ser 42. Déjame verificar...
    Si x=10, entonces 4x+2=42. Correcto."
   ```

2. **Reflexión**: identifica errores en su razonamiento mid-stream.

   ```
   "Wait, let me reconsider. The previous calculation
   assumed X=5 but the problem states X=4..."
   ```

3. **Razonamiento de cadena larga**: el modelo aprende a usar **miles de tokens internos** (entre `<think>...</think>`) antes de emitir la respuesta final. La longitud crece con el entrenamiento.

4. **"Aha moment"**: saltos discretos de performance donde el modelo "descubre" estrategias nuevas de razonamiento.

> 💡 **Analogía con self-play en juegos**: estos comportamientos emergentes son análogos al "aha moment" de AlphaGo Move 37 — el modelo descubre estrategias que ningún humano le enseñó porque la única señal externa es el reward verificable.

### 3.3. Por qué importa filosóficamente

R1-Zero es **evidencia complementaria** a Self-Rewarding de que **auto-mejora sin humanos funciona**.

- Self-Rewarding: auto-mejora en **instruction-following** vía LLM-as-Judge.
- R1-Zero: auto-mejora en **razonamiento** vía reward verificable.

Ambos sin anotaciones humanas post-inicialización. Ambos producen capacidades emergentes notables.

---

## 4. R1 (modelo de producción)

R1-Zero tiene problemas:
- **Lenguaje mixto**: las cadenas de pensamiento mezclan chino e inglés.
- **Legibilidad pobre**: respuestas técnicamente correctas pero estilísticamente erráticas.
- **Repetición**: frases redundantes en cadenas largas.

R1 (la versión de producción) resuelve esto con un pipeline de 4 etapas:

```
1. Cold-start SFT sobre miles de demostraciones de razonamiento
   curadas (a partir de R1-Zero filtrado).
2. RL con GRPO (igual que R1-Zero pero partiendo del modelo SFT).
3. Rejection sampling + SFT: usar el modelo de la etapa 2 para
   generar demostraciones masivas, filtrar con reward, fine-tune SFT.
4. RL final: ronda extra de RL para alinear con preferencias humanas
   + razonamiento.
```

El SFT cold-start resuelve los problemas de legibilidad **sin sacrificar** la capacidad de razonamiento.

---

## 5. Resultados destacados

| Benchmark | R1 | Comparable |
|---|---|---|
| MATH-500 | ~97.3% | o1: ~96% |
| AIME 2024 | ~79.8% pass@1 | Top performance abierto |
| Codeforces | Rating Elo ~2029 | Top 4% de competidores humanos |
| MMLU | ~90.8% | GPT-4o: ~88% |

> 💡 **Importante**: R1 **rivaliza con o1** en razonamiento, siendo open-weight y un órden de magnitud más barato. Es un hito de eficiencia.

---

## 6. Comparación lado-a-lado: Self-Rewarding vs R1-Zero

| Aspecto | Self-Rewarding (Yuan 2024) | DeepSeek-R1-Zero |
|---|---|---|
| Filosofía | Auto-mejora sin feedback externo | Auto-mejora sin feedback externo |
| Señal de aprendizaje | LLM-as-a-Judge (subjetiva) | Reward verificable (objetiva) |
| Dominio óptimo | Instruction-following abierto | Razonamiento (math, code) |
| Algoritmo | DPO (offline) | GRPO (on-policy) |
| Tipo de loop | Iteraciones del modelo `M_t` | Rollouts del mismo modelo |
| Modelos en memoria | 2 | 3 (sin critic) |
| Comportamientos emergentes | Mejora generación + juicio | Auto-verif, reflexión, "aha moment" |
| Riesgo principal | Self-preference bias, saturación | Reward hacking en RL prolongado |

**Son complementarios.** La presentación puede usar R1 como **evidencia adicional** de que "auto-mejora sin humanos" es una dirección viable, ampliando la legitimidad de Self-Rewarding más allá del único punto de datos de Yuan 2024.

---

## 7. ¿Cuándo elegir GRPO sobre DPO (y viceversa)?

| Escenario | Recomendación |
|---|---|
| Tarea con reward verificable (math, código) | **GRPO** |
| Tarea subjetiva (instrucciones, escritura) | **DPO** (Self-Rewarding) |
| Quieres exploración on-policy | **GRPO** |
| Quieres simplicidad y bajo costo | **DPO** |
| Quieres comportamientos emergentes | **GRPO** (más cercanos a RL puro) |
| Tienes presupuesto limitado de cómputo | **DPO** |

---

## 8. Limitaciones de GRPO/R1

### 8.1. Requiere reward verificable

No aplica a tareas subjetivas. Aquí Self-Rewarding sigue siendo necesario.

### 8.2. Costo de muestreo

GRPO requiere `G` rollouts por prompt (típicamente `G = 16` o `64`). Más caro que DPO offline.

### 8.3. Reward hacking en RL prolongado

Mismo problema que PPO en LLMs — el modelo puede aprender a explotar la función reward sin cumplir el objetivo real. Shafayat 2025 documenta esto.

### 8.4. Lenguaje mixto / legibilidad

R1-Zero los tiene; R1 los resuelve con SFT cold-start. Esto sugiere que **RL puro sin SFT** tiene un costo de legibilidad.

---

## 9. La filosofía conjunta

Self-Rewarding y R1 representan **dos brazos** de una misma evolución:

```
                Auto-mejora sin humanos
                         |
             +-----------+-----------+
             |                       |
        SUBJETIVO              VERIFICABLE
        (juez LLM)              (función reward)
             |                       |
       Self-Rewarding              R1-Zero
       Meta-Rewarding              GRPO
       SPIN, DICE, etc.           (math, code)
             |                       |
       Riesgo principal:    Riesgo principal:
       self-preference      reward hacking
       saturación           
```

Estos dos enfoques **se complementan** y se anticipa que el futuro híbrido (como SAR, cap 09) los combine.

---

## 10. Para la presentación: ¿incluir GRPO/R1?

**Recomendación**: incluir, pero **brevemente**.

- **1-2 slides** dedicadas a R1 como "el paradigma paralelo".
- Énfasis: "auto-mejora no es solo Self-Rewarding. R1 muestra que también funciona con reward verificable + GRPO. Confirma la dirección general."
- **No profundizar** en algoritmo GRPO completo — distrae del paper ancla.

Si el profesor pregunta "¿y DeepSeek-R1?", puedes responder con:

> *"DeepSeek-R1 es el paradigma complementario: reward verificable + GRPO + RL puro sin SFT cold-start (en R1-Zero). Comparte filosofía con Self-Rewarding (auto-mejora sin humanos post-inicialización) pero opera en dominios distintos (razonamiento verificable vs instruction-following abierto). Comportamientos emergentes notables: auto-verificación, reflexión, 'aha moment'. Es evidencia adicional de que la dirección general 'auto-mejora autónoma' es viable, no un experimento aislado."*

---

## 11. Frase para defensa

> *"GRPO simplifica PPO eliminando el critic, usando un grupo de rollouts como baseline. Lo aplica DeepSeek con R1, un hito empírico: capacidades de razonamiento avanzado emergen de RL puro sobre rewards verificables, sin SFT cold-start (R1-Zero) o con cold-start mínimo (R1). Comportamientos emergentes — auto-verificación, reflexión, 'aha moment' — son análogos al self-play en juegos. R1 y Self-Rewarding son complementarios: cubren regímenes distintos (verificable vs subjetivo), confirman conjuntamente que la dirección 'auto-mejora sin humanos' es la frontera del campo."*

---

## 12. Qué viene

[14-comparacion.md](14-comparacion.md) — la tabla canónica de todos los métodos. Síntesis final.

---

## Lecturas relacionadas

- [[concepts/grpo]] — versión densa de GRPO.
- [[entities/deepseek-r1]] — R1-Zero y R1 en detalle.
- [[summaries/acosta-2026-deep-research]] §7 — la integración en el deep research propio.
