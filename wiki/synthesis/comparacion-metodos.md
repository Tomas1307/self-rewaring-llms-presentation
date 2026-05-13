---
name: comparacion-metodos
description: Comparación canónica de RLHF, DPO, RLAIF, Self-Rewarding, Meta-Rewarding (+ SPIN, SPPO, GRPO) en una matriz unificada.
metadata:
  type: synthesis
  tags: [comparacion, pipeline, rlhf, dpo, rlaif, self-rewarding, meta-rewarding, grpo]
  last_updated: 2026-05-12
---

# Comparación de métodos del pipeline

> **Tesis.** La evolución RLHF → RLAIF → Self-Rewarding → Meta-Rewarding puede leerse como **reducción progresiva de dependencia externa**. En paralelo, la evolución PPO → GRPO → DPO simplifica el algoritmo. Las dos dimensiones (origen del feedback × algoritmo de optimización) definen el espacio donde encaja cada método.

## 1. Las dos dimensiones de la evolución

### 1.1. Origen del feedback (qué genera los pares de preferencia)

```
Humanos              →  Modelo externo     →  Mismo modelo       →  Mismo modelo (3 roles)
[RLHF]                  [RLAIF / CAI]         [Self-Rewarding]      [Meta-Rewarding]
```

### 1.2. Algoritmo de optimización (cómo se actualiza la política)

```
PPO (4 modelos)      →  GRPO (3 modelos)   →  DPO (2 modelos)
[RLHF clásico]          [DeepSeek-R1]         [DPO, Self-Rewarding, Meta-Rewarding]
```

Cada método cae en una celda específica del producto cruzado. Self-Rewarding ocupa la celda `(mismo modelo, DPO)`.

## 2. Tabla canónica

| Aspecto | RLHF | DPO | RLAIF | SPIN | Self-Rewarding | Meta-Rewarding | GRPO (R1) |
|---|---|---|---|---|---|---|---|
| **Fuente de feedback** | Humanos | Datos humanos (offline) | LLM externo | Self-play vs SFT | El mismo modelo | El mismo modelo (3 roles) | Reward verificable |
| **Reward model** | Separado, entrenado | Implícito en `π_θ` | Separado o directo | Implícito | Implícito (LLM-as-Judge) | Implícito + meta-juicio | Función externa (no LLM) |
| **Algoritmo** | PPO online | Pérdida cerrada | PPO o DPO | DPO iterativo | Iterative DPO | Iterative DPO | GRPO on-policy |
| **Modelos en memoria** | 4 | 2 | 3-4 | 2 | 2 | 2 | 3 |
| **Iterativo** | No (RM congelado) | No (una pasada) | Posible | Sí | Sí (núcleo del método) | Sí (con meta-juez) | Sí (rollouts) |
| **Mejora del juicio del modelo** | No | No | No | No (juez fijo SFT) | Sí (implícita) | Sí (explícita) | N/A |
| **Anotación humana requerida** | Sí (miles) | Solo inicial | Solo SFT | Solo SFT | Solo SFT | Solo SFT | Solo SFT |
| **Dominio óptimo** | Cualquiera | Cualquiera | Cualquiera | Cualquiera | Instruction-following | Instruction-following | Razonamiento verificable |
| **Costo relativo** | Muy alto | Bajo | Medio | Medio-bajo | Medio-bajo | Medio-bajo | Medio |

## 3. Lecturas verticales de la tabla

### 3.1. ¿Quién provee el feedback?

```
RLHF        : humanos                    (costoso, sesgado, no escala)
RLAIF       : LLM externo                (escalable, pero requiere otro modelo)
SPIN        : SFT congelado vs política  (auto-mejora vs ancla fija)
Self-Reward : mismo modelo               (cierra el loop, riesgo de drift)
Meta-Reward : mismo modelo (3 roles)     (jerarquía interna, mitiga drift parcial)
GRPO-R1     : función verificable        (objetiva pero limitada a tareas con respuesta)
```

### 3.2. ¿Qué tan complejo es el algoritmo?

```
PPO (RLHF)  : 4 modelos, hiperparams sensibles, infraestructura RL completa
GRPO (R1)   : 3 modelos, baseline grupal, sin critic
DPO         : 2 modelos, pérdida supervisada, estable
```

### 3.3. ¿Mejora el juez del modelo?

| Método | Juez | Mejora con t |
|---|---|---|
| RLHF | RM separado congelado | No |
| RLAIF | LLM externo congelado | No |
| SPIN | SFT congelado | No |
| Self-Rewarding | `M_t` mismo | **Sí (efecto colateral del DPO)** |
| Meta-Rewarding | `M_t` mismo + meta-juez | **Sí (objetivo explícito)** |

Esta diferencia es central para la tesis del Pilar 2 ([[synthesis/pilar-2-multi-agente-cooperativo]]): solo en Self-Rewarding y Meta-Rewarding el juez es un agente que aprende.

## 4. Trade-offs principales por método

### 4.1. RLHF
- **Pro**: el más establecido, evidencia abundante, control humano.
- **Contra**: costo, escalabilidad, cuello de botella superhumano, 4 modelos.

### 4.2. DPO
- **Pro**: simple, estable, costo SFT.
- **Contra**: offline (sin exploración), depende del dataset estático, colapso documentado bajo preferencias deterministas (Azar 2023, ver [[summaries/azar-2023-ipo]]).

### 4.3. RLAIF / Constitutional AI
- **Pro**: escalable, principios auditables (CAI), consistente.
- **Contra**: hereda sesgos del juez, requiere otro modelo capaz.

### 4.4. SPIN
- **Pro**: solo requiere SFT, pseudo-self-play, ancla fija (`π_ref = SFT`).
- **Contra**: el ancla no mejora, plateau después de pocas iteraciones.

### 4.5. Self-Rewarding
- **Pro**: cero feedback externo post-SFT, juez mejora con t, iteración formalizada.
- **Contra**: self-preference bias, saturación a 3-5 iteraciones (Wang 2025, ver [[synthesis/diagnostico-fallos]]).

### 4.6. Meta-Rewarding
- **Pro**: mejora explícitamente el juicio, mitiga saturación.
- **Contra**: añade complejidad (3 roles), no resuelve self-preference fundamentalmente.

### 4.7. GRPO (DeepSeek-R1)
- **Pro**: sin critic, sin RM entrenado, comportamientos emergentes documentados.
- **Contra**: requiere reward verificable (no aplica a tareas subjetivas), riesgo de reward hacking en RL prolongado (Shafayat 2025).

## 4.bis. Variantes 2024-2025 que extienden Self-Rewarding

Trece papers posteriores a Yuan 2024 refinan o mitigan limitaciones específicas. Agrupadas por dimensión del problema que atacan:

### Mitigación del sesgo acumulado / saturación

- **CREAM** (Wang Z. 2024, ICLR 2025): regulariza la señal de preferencia por **consistencia de rankings entre iteraciones consecutivas**. Si la consistencia es baja, atenúa la señal.
- **SCIR** (Zhou 2025): exige **consistencia entre múltiples reward signals internos** (juez generativo + DPO implícito + likelihood); filtra pares inconsistentes.
- **Temporal SR** (Wang 2025): **decouple temporal** — `y_l` desde modelo pasado (SFT), `y_w` desde modelo futuro (EMA).

### Reward implícito como bootstrap

- **DICE** (Chen C. 2024, ICLR 2025): explota `r̂ = β·log(π_θ/π_ref)` (la reward DPO implícita) **como juez** para generar nuevos pares. Variante "sin LLM-as-Judge externo".
- **SER** (Huang 2024): el reward model **se auto-mejora**, no solo la política.

### Preferencias generales / sin Bradley-Terry

- **DNO** (Rosset 2024, Microsoft): equilibrio de Nash sobre preferencias **arbitrarias** (incluye intransitivas). Garantía de mejora monotónica.
- **SPPO** (Wu 2024, NeurIPS): self-play como juego de suma constante; aproxima Nash con PairRM como juez externo pequeño.

### Auto-mejora autónoma extendida

- **Self-Taught Evaluators** (Tianlu Wang 2024, Meta): el **juez** se auto-entrena sin etiquetas humanas. Llama3-70B-Instruct pasa de 75.4 a 88.3 en RewardBench.
- **RLSR** (2025): el modelo **genera las tareas** que practica (curriculum auto-dirigido).
- **Multi-Agent Evolve** (2025): múltiples instancias del LLM co-evolucionan con tres tipos de reward (juez + dificultad + formato).

### Self-Rewarding para razonamiento (donde el clásico falla)

- **Process-SRLM** (Zhang 2025, ACL): juez **paso-a-paso** + razonamiento de cadena larga.
- **LaTRO** (Salesforce 2024): razonamiento como variable latente, reward también del LLM.
- **ScPO** (2024): **mayoría entre samples** como señal de preferencia (cuando el juez no puede evaluar bien).
- **Self-Rewarding Correction** (Xiong 2025): genera y verifica corrección durante inferencia.
- **CoVo** (Zhang K. 2025): reward geométrica de **consistencia + baja volatilidad** en trayectorias.
- **SAR** (2025): **híbrido** verificable + self-aligned, mitiga respuestas demasiado largas.

### Self-Rewarding multimodal

- **CSR** (Zhou Y. 2024, NeurIPS): vision-language con restricciones explícitas de alineación visual.
- **Vision-SR1** (2025): descomposición percepción + razonamiento; auto-reward por consistencia.

**Ver detalle de cada variante en sus entradas correspondientes en [[sources]].**

## 5. Comparación experimental head-to-head

La comparación más limpia disponible es [[summaries/wu-2024-sppo]] (Wu 2024 SPPO), que evalúa **SPPO vs Self-Rewarding vs SPIN vs DPO vs IPO** sobre el mismo setup:

- **Modelo base**: Mistral-7B-Instruct-v0.2.
- **Dataset**: UltraFeedback 60k prompts.
- **Benchmark**: AlpacaEval 2.0 length-controlled.

Ranking aproximado (de los resultados reportados):

```
SPPO  >  Self-Rewarding (iter 3)  ≈  SPIN (iter 3)  >  DPO (single)  >  IPO
```

**Caveats** (importantes para la presentación):
- SPPO usa un juez externo (PairRM), no es estrictamente self-rewarding.
- Self-Rewarding compite competitivamente sin juez externo, lo que es notable.
- No hay comparación head-to-head GRPO vs Self-Rewarding porque cubren dominios distintos.

## 6. Cuándo elegir cada método (guía práctica)

Si la presentación debe responder "¿qué usar?":

| Escenario | Método recomendado |
|---|---|
| Dataset de preferencias humanas, recursos para PPO | RLHF |
| Dataset de preferencias, recursos limitados | DPO |
| Sin presupuesto de anotación, modelo externo disponible | RLAIF |
| Sin modelo externo, instrucciones abiertas | Self-Rewarding |
| Self-Rewarding satura, se quiere mejor juez | Meta-Rewarding |
| Tareas con respuesta verificable (math, code) | GRPO |
| Self-play sin externo, ancla estable | SPIN |

## 7. La narrativa para la presentación

Tres slides posibles a partir de esta página:

### Slide A — "La evolución del pipeline en una imagen"

Diagrama de dos ejes:
- Eje X: dependencia externa (humanos → LLM externo → mismo modelo → meta-jerarquía).
- Eje Y: complejidad algorítmica (PPO → GRPO → DPO).

Cada método como punto en el plano. Self-Rewarding queda en `(mismo modelo, DPO)`.

### Slide B — "Tabla maestra"

La tabla del §2 reducida a 5 columnas (omitir GRPO si no entra) y 6 filas (las más críticas).

### Slide C — "¿Por qué Self-Rewarding y no otro?"

Justificación de la elección del paper ancla mostrando:
- Self-Rewarding cubre el caso límite "mismo modelo, DPO".
- Es el primer método que **formaliza la iteración** del loop.
- Empíricamente compite con RLHF sin anotación humana posterior al SFT.

## 8. Lecturas relacionadas

- [[concepts/rlhf]], [[concepts/dpo]], [[concepts/rlaif]], [[concepts/grpo]], [[concepts/self-rewarding-loop]] — cada concepto en detalle.
- [[entities/deepseek-r1]] — caso de estudio paralelo (GRPO).
- [[synthesis/pilar-2-multi-agente-cooperativo]] — quién aprende como agente en cada método.
- [[synthesis/diagnostico-fallos]] — fallos específicos de Self-Rewarding.
- [[summaries/wu-2024-sppo]] — comparación experimental head-to-head.
- [[summaries/acosta-2026-deep-research]] §6 — tabla original que sirve de esqueleto.
