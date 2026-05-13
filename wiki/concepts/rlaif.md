---
name: rlaif
description: Reinforcement Learning from AI Feedback — reemplaza anotadores humanos por un LLM externo que genera las preferencias para entrenar el reward model.
metadata:
  type: concept
  tags: [rlaif, ai-feedback, llm-as-judge, escalabilidad, constitutional-ai]
  last_updated: 2026-05-12
---

# RLAIF — Reinforcement Learning from AI Feedback

> Variante de RLHF donde un LLM (típicamente más grande o más capaz) reemplaza a los anotadores humanos. Formalizado por Lee et al. (2023, Google DeepMind); origen conceptual en [[concepts/constitutional-ai]] (Bai et al. 2022, Anthropic). Paso intermedio en la evolución hacia [[concepts/self-rewarding-loop]].

## 1. El problema que resuelve

[[concepts/rlhf]] tiene una limitación estructural: **el costo y la escalabilidad de la anotación humana**.

- Cada round de mejora requiere miles de comparaciones humanas.
- Variabilidad inter-anotador introduce ruido.
- Si el modelo supera al humano, los humanos no pueden anotar bien.

**Insight de RLAIF**: reemplazar la anotación humana por **un LLM**. Es más rápido, más consistente y mucho más barato.

## 2. Pipeline canónico

### Versión "RM intermedio" (clásica)

```
1. Modelo SFT genera N respuestas candidatas por prompt.
2. LLM juez (off-the-shelf) compara las respuestas y produce preferencias.
3. Con esas preferencias AI-generadas, entrenar un reward model r̂ (Bradley-Terry).
4. Aplicar PPO usando r̂ como en RLHF clásico.
```

### Versión "direct RLAIF"

```
1. Modelo SFT genera respuestas.
2. LLM juez asigna directamente un score numérico s(x, y).
3. Usar s(x, y) como señal de reward durante PPO (sin entrenar RM separado).
```

La diferencia: en la versión clásica, las preferencias AI alimentan un RM dedicado; en direct RLAIF, el juez actúa como RM directo.

## 3. Diseño del juez

El LLM juez recibe un **prompt estructurado** con la rúbrica de evaluación. Típicamente:

```
System: Eres un evaluador imparcial. Tu tarea es comparar dos respuestas
y decidir cuál es mejor según los criterios listados.

Criterios:
- Utilidad: ¿la respuesta atiende la pregunta?
- Honestidad: ¿es factualmente correcta?
- Seguridad: ¿evita contenido dañino?

User prompt: [prompt original]
Respuesta A: [y_A]
Respuesta B: [y_B]

Output:
- Razonamiento: [explicación breve]
- Veredicto: A o B
```

Detalles del paradigma en [[concepts/llm-as-a-judge]].

## 4. Resultados empíricos clave

**Lee et al. (2023, 2024 ICML)** estableció empíricamente:

- RLAIF logra **rendimiento comparable a RLHF** en tareas de resumen, diálogo útil, y diálogo inofensivo.
- En algunos benchmarks, RLAIF **iguala o supera** a RLHF.
- RLAIF puede lograr auto-mejora **incluso cuando el LLM juez es del mismo tamaño que la política** (o el mismo checkpoint).

Este último hallazgo es crítico para la narrativa: **desdibuja la frontera** entre RLAIF y Self-Rewarding. Si el juez puede ser el mismo modelo, RLAIF se convierte en una variante de Self-Rewarding off-policy. Ver matiz en [[summaries/acosta-2026-deep-research]].

## 5. Constitutional AI como caso emblemático

[[concepts/constitutional-ai]] (Bai et al. 2022, Anthropic) es la implementación más influyente de RLAIF:

- **Fase SL**: el modelo se auto-critica según principios constitucionales explícitos y revisa sus respuestas.
- **Fase RL**: el modelo evalúa pares de respuestas según principios aleatorios, generando preferencias para PPO.

**Diferencia clave**: los principios son **explícitos y auditables**, no preferencias implícitas como en RLHF.

## 6. Relación con Self-Rewarding

| Aspecto | RLAIF (Lee 2023) | Self-Rewarding (Yuan 2024) |
|---|---|---|
| Juez | LLM externo (puede ser el mismo) | El mismo modelo en rol de juez |
| Iteración | Una sola pasada típicamente | Loop `M_t → M_{t+1}` formalizado |
| Mejora del juez | No (juez fijo) | Sí (el juez mejora con t) |
| Algoritmo | PPO o DPO | DPO iterativo |
| Pares de preferencia | Sobre dataset estático | Sobre dataset que cambia con t |

**Self-Rewarding = RLAIF + iteración formal + DPO + juez = generador.**

La novedad de Self-Rewarding no es la idea de "modelo evalúa modelo" (eso ya está en RLAIF), sino:
1. **Hacer explícito** que el juez puede ser el mismo modelo.
2. **Iterar** el loop produciendo una secuencia `M_0, M_1, M_2, ...`.
3. **Mostrar empíricamente** que ambas capacidades (generar y juzgar) mejoran simultáneamente.

## 7. Ventajas y limitaciones

### 7.1. Ventajas

- **Escalabilidad masiva**: sin costo de anotación humana.
- **Consistencia**: el juez aplica los mismos criterios uniformemente.
- **Velocidad**: ciclos de iteración órdenes de magnitud más rápidos.
- **Trazabilidad** (en CAI): principios explícitos auditables.

### 7.2. Limitaciones

- **Sesgos del juez** ([[concepts/llm-as-a-judge]]):
  - **Self-enhancement bias**: el juez prefiere outputs similares a los que él mismo produciría.
  - **Length bias**: prefiere respuestas largas.
  - **Position bias**: en pairwise, prefiere la primera o segunda respuesta sistemáticamente.
- **Amplificación de errores**: si el juez tiene una limitación sistemática (e.g., entiende mal una categoría), se propaga al modelo entrenado.
- **Dependencia de un modelo externo** capaz. Si el juez es débil, el resultado es débil.

## 8. Multi-agent cooperativo

RLAIF puede interpretarse como sistema cooperativo de dos roles:

- **Generador**: produce respuestas. Maximiza la calidad según el juez.
- **Juez**: evalúa respuestas. Objetivo: producir señales útiles para entrenamiento.

Aunque no es multi-agente en el sentido formal de MARL (los agentes no comparten un entorno ni aprenden simultáneamente), la **dinámica cooperativa** es análoga. Ver [[synthesis/pilar-2-multi-agente-cooperativo]] para el desarrollo del framing.

## 9. Variantes y trabajos relacionados

- **Constitutional AI** (Bai 2022) — el progenitor. [[concepts/constitutional-ai]].
- **RLAIF sistemático** (Lee 2023, arXiv:2309.00267) — formalización a escala.
- **Direct RLAIF** — variante sin RM intermedio.
- **RLAIF iterativo** — re-evaluar con el juez después de cada round.

## 10. Lecturas relacionadas

- [[concepts/rlhf]] — paradigma del que RLAIF es modificación.
- [[concepts/constitutional-ai]] — caso emblemático con principios explícitos.
- [[concepts/self-rewarding-loop]] — extensión al caso juez = generador.
- [[concepts/llm-as-a-judge]] — paradigma del juez en detalle.
- [[summaries/zheng-2023-llm-as-judge]] — sesgos cuantificados.
- [[synthesis/comparacion-metodos]] — tabla canónica donde aparece.
- [[summaries/acosta-2026-deep-research]] §4 — síntesis original.
