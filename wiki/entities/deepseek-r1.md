---
name: deepseek-r1
description: DeepSeek-R1 — modelo de razonamiento entrenado con RL puro (GRPO) sin SFT inicial; hito en auto-mejora con rewards verificables.
metadata:
  type: entity
  tags: [modelo, deepseek, grpo, razonamiento, rl-puro, hito]
  last_updated: 2026-05-12
---

# DeepSeek-R1

> Familia de modelos de razonamiento lanzada por DeepSeek-AI en enero de 2025. **R1-Zero** fue entrenado con RL puro (GRPO) sin supervised fine-tuning inicial, demostrando que capacidades de razonamiento avanzado pueden emerger únicamente de la señal de reward verificable.

## 1. Variantes principales

| Modelo | Base | Entrenamiento | Notable por |
|---|---|---|---|
| **R1-Zero** | DeepSeek-V3-Base | RL puro con GRPO, **sin SFT** | Hito: capacidades emergentes |
| **R1** | DeepSeek-V3-Base | Cold-start SFT + RL con GRPO + rejection sampling | Modelo de producción |
| **R1-Distill-{7B, 32B, 70B}** | Llama/Qwen | SFT sobre trayectorias de R1 | Versiones pequeñas open-weight |

## 2. Por qué importa para esta presentación

DeepSeek-R1 es **el caso de estudio paralelo a Self-Rewarding** en la narrativa del curso:

- Ambos son paradigmas de **auto-mejora** sin feedback humano post-inicialización.
- Difieren en el régimen donde funcionan: R1 en razonamiento verificable, Self-Rewarding en instruction-following abierto.
- Su existencia simultánea valida la dirección general "reducir dependencia externa" como tendencia consolidada del campo en 2024-2025.

Ver [[summaries/acosta-2026-deep-research]] §7 para la integración completa.

## 3. R1-Zero: RL puro sin SFT

### 3.1. Setup

- **Modelo base**: DeepSeek-V3-Base (preentrenamiento estándar, sin instrucción).
- **Algoritmo**: [[concepts/grpo]] con `G = 64` rollouts por prompt.
- **Función de reward**:
  - Reward de **precisión**: `+1` si la respuesta final es correcta, `0` si no (matemáticas, código).
  - Reward de **formato**: `+1` si la respuesta sigue el patrón `<think>...</think><answer>...</answer>`, penalización si no.
- **Sin SFT cold-start**. El modelo aprende de cero a razonar a través de RL.

### 3.2. Comportamientos emergentes documentados

Durante el entrenamiento emergen comportamientos **no programados explícitamente**:

1. **Auto-verificación.** El modelo escribe pasos de verificación de su propia respuesta antes de comprometerse.
2. **Reflexión.** Identifica errores en su razonamiento mid-stream: "Wait, let me reconsider...".
3. **Razonamiento de cadena larga.** El modelo aprende a usar **miles de tokens internos** (`<think>`) antes de emitir la respuesta final. La longitud crece con el entrenamiento.
4. **"Aha moment".** Saltos discretos de performance donde el modelo "descubre" estrategias nuevas de razonamiento.

Estos son análogos al **self-play** en juegos (AlphaGo, AlphaZero). El modelo descubre estrategias que ningún humano le enseñó porque la única señal externa es el reward verificable.

### 3.3. Limitaciones de R1-Zero

- **Lenguaje mixto.** Las cadenas de pensamiento mezclan chino e inglés porque el modelo no fue restringido a un idioma.
- **Legibilidad pobre.** Las respuestas son técnicamente correctas pero estilísticamente erráticas.
- **Repetición.** El modelo repite frases en las cadenas largas.

Estas son las razones por las que R1 (con SFT cold-start) existe como modelo de producción.

## 4. R1 (modelo de producción)

Pipeline en 4 etapas:

1. **Cold-start SFT**: fine-tuning supervisado sobre miles de demostraciones de razonamiento de alta calidad (curadas por humanos a partir de R1-Zero).
2. **RL con GRPO**: igual que R1-Zero pero partiendo del modelo SFT.
3. **Rejection sampling + SFT**: usar el modelo de la etapa 2 para generar masivas demostraciones, filtrar con reward, fine-tune SFT.
4. **RL final**: otra ronda de RL para alinear con preferencias humanas + razonamiento.

El SFT cold-start resuelve los problemas de lenguaje mixto y legibilidad, manteniendo la capacidad de razonamiento.

## 5. Resultados destacados

- **MATH-500**: R1 alcanza ~97.3% (comparable a o1).
- **AIME 2024**: R1 alcanza ~79.8% pass@1 (top performance abierto en la fecha de release).
- **Codeforces**: rating Elo equivalente a 2029 (top 4% de competidores humanos).
- **MMLU**: ~90.8% (5-shot).

Verificable en el [paper técnico](https://arxiv.org/abs/2501.12948).

## 6. Relación con Self-Rewarding

| Aspecto | Self-Rewarding (Yuan 2024) | DeepSeek-R1-Zero |
|---|---|---|
| Filosofía | Auto-mejora sin feedback externo | Auto-mejora sin feedback externo |
| Señal de aprendizaje | LLM-as-a-Judge (subjetiva) | Reward verificable (objetiva) |
| Dominio óptimo | Instruction-following abierto | Razonamiento (math, code) |
| Algoritmo | DPO (offline) | GRPO (on-policy) |
| Tipo de loop | Iteraciones del modelo `M_t` | Rollouts del mismo modelo |
| Riesgo principal | Self-preference bias, saturación | Reward hacking en RL prolongado |

**Son complementarios.** La narrativa de la presentación puede usar R1 como evidencia adicional de que "auto-mejora sin humanos" es una dirección viable, ampliando la legitimidad de Self-Rewarding más allá del único punto de datos de Yuan 2024.

## 7. Lecturas relacionadas

- [[concepts/grpo]] — algoritmo subyacente.
- [[summaries/yuan-2024-self-rewarding]] — paradigma análogo en dominio distinto.
- [[summaries/acosta-2026-deep-research]] §7 — caso de estudio dentro del documento maestro.
- [[synthesis/comparacion-metodos]] — donde aparece en la tabla canónica.

## 8. Fuentes

- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948. URL: https://arxiv.org/abs/2501.12948
- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv. URL: https://arxiv.org/abs/2402.03300 (paper donde se introduce GRPO).
