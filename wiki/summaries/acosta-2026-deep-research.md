---
name: acosta-2026-deep-research
description: Deep research propio del autor consolidando el pipeline RLHF → RLAIF → Self-Rewarding → DPO, con extensión a GRPO/DeepSeek-R1 y conexión IRL.
metadata:
  type: summary
  tags: [meta-fuente, sintesis-propia, pipeline, panorama]
  last_updated: 2026-05-12
---

# Acosta (2026) — Investigación Técnica: RL para Alineamiento de LLMs

> **Naturaleza**: documento de síntesis propia, no paper externo. Sirve como **mapa narrativo** de la presentación y como fuente para varios sub-temas que no aparecían en la primera ronda bibliográfica.

## Metadatos

- **Autor**: Tomás Acosta. Universidad de los Andes — Maestría en Ingeniería de Sistemas y Computación.
- **Fecha**: mayo 2026.
- **Archivo**: `wiki/raw/Investigacion_RL_para_LLMs_RLHF_RLAIF_SelfRewarding_DPO.pdf` (11 páginas).
- **Tipo**: deep research interno, no peer-reviewed.

## Por qué se ingiere

Este documento es **el borrador maestro** de la narrativa que sostiene la presentación. Su estructura define qué slides deben existir y en qué orden. Adicionalmente trae contenido **nuevo respecto al wiki**:

1. **GRPO + DeepSeek-R1** (sección 7) — caso de estudio que cierra la evolución del pipeline. Ver [[concepts/grpo]] y [[entities/deepseek-r1]].
2. **Constitutional AI desglosado en fases SL + RL** (4.4) — material para [[concepts/constitutional-ai]].
3. **PPO en detalle (4 modelos en memoria)** (2.4) — incorporado en [[concepts/rlhf]].
4. **Tabla comparativa de 5 columnas** (sección 6) — esqueleto de [[synthesis/comparacion-metodos]].
5. **Conexión formal IRL ↔ Self-Rewarding** (5.5) — refuerzo de [[synthesis/pilar-1-irl-implicito]] vía Joselowitz 2025 y Sun-vanderSchaar 2024.

## Estructura del documento

| § | Contenido | Páginas wiki afectadas |
|---|---|---|
| 1 | Introducción y motivación | — |
| 2 | RLHF: pipeline 3 etapas, PPO en detalle, limitaciones | [[concepts/rlhf]] |
| 3 | DPO: motivación, derivación, ventajas, variantes (IPO/KTO/GRPO) | [[concepts/dpo]], [[concepts/grpo]] |
| 4 | RLAIF: pipeline, direct RLAIF, Constitutional AI, conexión multi-agente | [[concepts/rlaif]], [[concepts/constitutional-ai]] |
| 5 | Self-Rewarding: paradigma, loop iterativo, Meta-Rewarding, conexión IRL y multi-agente | [[concepts/self-rewarding-loop]], [[synthesis/pilar-1-irl-implicito]], [[synthesis/pilar-2-multi-agente-cooperativo]] |
| 6 | Tabla comparativa RLHF / DPO / RLAIF / Self-Rewarding / Meta-Rewarding | [[synthesis/comparacion-metodos]] |
| 7 | Caso DeepSeek-R1 y GRPO | [[concepts/grpo]], [[entities/deepseek-r1]] |
| 8 | Problemas abiertos: saturación, calibración, process-based, autosupervisión | [[synthesis/diagnostico-fallos]] |
| 9 | Referencias | [[sources]] |

## Tesis central del documento

> La evolución del pipeline puede leerse como **reducción progresiva de dependencia externa**: humanos (RLHF) → modelo externo (RLAIF) → modelo mismo (Self-Rewarding) → meta-jerarquía interna (Meta-Rewarding). En paralelo se simplifica el algoritmo: PPO con critic → DPO sin RL online → GRPO con baseline grupal sin critic.

Esto es la narrativa que la presentación debe transmitir en su primer tercio.

## Aportes nuevos respecto al wiki preexistente

### 1. GRPO como puente algorítmico

El PDF presenta GRPO (Shao et al. 2024, *DeepSeekMath*) como solución al problema del critic en PPO:

- En PPO se necesita un critic (otra red neuronal) para estimar el value function. Esto duplica el costo de memoria.
- GRPO genera `G` respuestas al mismo prompt, evalúa cada una con la función de reward (verificable), y **normaliza el reward dentro del grupo** para estimar el advantage. Sin critic.
- Ventaja: combina la exploración on-policy de PPO con costo cercano a DPO.

Ver [[concepts/grpo]] para derivación y comparación.

### 2. DeepSeek-R1: RL puro sin SFT

El PDF documenta R1-Zero como hito experimental:

- Entrenado únicamente con RL (GRPO) sin supervised fine-tuning inicial.
- Emergieron comportamientos no programados: auto-verificación, reflexión, "aha moment", razonamiento de cadena larga.
- Analogía con self-play: el modelo descubre estrategias que ningún humano le enseñó.

Ver [[entities/deepseek-r1]].

### 3. Relación complementaria GRPO ↔ Self-Rewarding

Ambos paradigmas comparten la filosofía de auto-mejora pero difieren en mecanismo:

| Aspecto | GRPO / R1-Zero | Self-Rewarding |
|---|---|---|
| Señal de reward | Verificable (correcto/incorrecto en matemáticas) | LLM-as-a-Judge (rúbrica subjetiva) |
| Dominio | Razonamiento con respuesta verificable | Instruction-following abierto |
| Algoritmo RL | GRPO (on-policy con baseline grupal) | DPO (offline) |
| Loop iterativo | Sí (sobre rollouts) | Sí (sobre iteraciones de modelo) |

Son **complementarios, no competidores**: cubren regímenes distintos del espacio de tareas.

### 4. IRL aplicado a LLMs alineados

El PDF cita dos trabajos que no estaban en el wiki:

- **Joselowitz et al. (2025)** — IRL recupera reward functions de LLMs RLHF con hasta 85% precisión. Refuerzo cuantitativo del Pilar 1. Ver [[summaries/joselowitz-2025-irl-llm]].
- **rePIRL (Meta/UC Santa Barbara, 2025)** — IRL token-wise para razonamiento. [VERIFICAR cita exacta — el PDF no provee arXiv ID].
- **Sun & van der Schaar (2024)** — tutorial/survey *Inverse RL Meets LLM Alignment*. Posicionamiento académico de la conexión. Ver [[summaries/sun-vanderschaar-2024-irl-llm]].

### 5. PPO requiere 4 modelos en memoria

Detalle pedagógico útil para la slide "¿por qué DPO simplifica?":

| Modelo | Rol | Estado en PPO | Estado en DPO |
|---|---|---|---|
| Política actual (`π_θ`) | Genera tokens, se entrena | activo | activo |
| Política de referencia (`π_ref`) | Ancla KL | congelado | congelado |
| Reward model | Asigna `r(x,y)` | congelado, separado | **no existe** |
| Critic / Value head | Predice `V(s)` para advantage | activo | **no existe** |
| **Total** | | **4** | **2** |

## Contradicciones / matices a marcar

### Matiz 1: "DPO converge al mismo óptimo global que RLHF con PPO"

El PDF (§3.3) afirma esto bajo el supuesto de "condiciones ideales (modelo Bradley-Terry perfecto, reward model óptimo)". Es técnicamente correcto, pero hay que matizar:

- **Equivalencia teórica**: sí, bajo Bradley-Terry y datos infinitos. Demostrada por Rafailov 2023.
- **Equivalencia empírica**: no garantizada. [[summaries/azar-2023-ipo]] muestra que DPO puede **colapsar** bajo preferencias deterministas o casi-deterministas, alejándose drásticamente del óptimo RLHF.

La presentación debe presentarlo como "equivalencia bajo supuestos fuertes, divergencia bajo supuestos realistas".

### Matiz 2: RLAIF puede lograr auto-mejora incluso con juez = checkpoint

El PDF (§4.3) cita Lee et al. 2024 afirmando que RLAIF logra auto-mejora cuando el LLM juez es del **mismo tamaño** que la política, o **incluso el mismo checkpoint**. Esto **desdibuja la frontera** RLAIF vs Self-Rewarding:

- En el límite `juez = generador`, RLAIF se convierte en una variante off-policy de Self-Rewarding.
- Yuan 2024 hace explícito el cierre del loop (iteración formal), pero la idea de auto-mejora con juez interno ya estaba implícita en Lee 2024.

Esto matiza la narrativa "Self-Rewarding inventa la auto-mejora". Es más justo decir "Self-Rewarding **formaliza** y **iteriza** una idea presente en RLAIF".

### Matiz 3: rePIRL no verificado

El PDF menciona "rePIRL (Meta/UC Santa Barbara, 2025)" sin arXiv ID. Búsqueda inicial no localiza un paper con ese nombre exacto. Marcar como [VERIFICAR] hasta confirmar referencia.

## Hallazgos secundarios

- **Self-Rewarding aplicado a Llama 2 70B**: 3 iteraciones produjeron modelo que superó a Claude 2, Gemini Pro y GPT-4 0613 en AlpacaEval 2.0. Ya documentado en [[summaries/yuan-2024-self-rewarding]].
- **Meta-Rewarding en Llama-3-8B-Instruct**: mejora win rate de 22.9% → 39.4%, superando GPT-4-0314. Ya en [[summaries/wu-2024-meta-rewarding]].
- **R1-Zero**: emergen comportamientos de auto-verificación, reflexión y "aha moment" sin programación explícita. Nuevo en [[entities/deepseek-r1]].

## Cómo usar este resumen

Esta página actúa como **índice narrativo**. Cuando se redacte una slide o se busque un tema concreto, partir de aquí para localizar la sección del PDF + las páginas del wiki que lo desarrollan.

## Lecturas relacionadas

- [[concepts/grpo]], [[entities/deepseek-r1]] — material nuevo principal.
- [[concepts/rlhf]], [[concepts/rlaif]], [[concepts/constitutional-ai]] — páginas que este PDF ayuda a completar.
- [[synthesis/comparacion-metodos]] — tabla canónica de los métodos.
- [[summaries/joselowitz-2025-irl-llm]], [[summaries/sun-vanderschaar-2024-irl-llm]] — nuevas fuentes IRL ↔ LLM.
- [[synthesis/pilar-1-irl-implicito]] — síntesis reforzada con las nuevas fuentes.
