---
name: joselowitz-2025-irl-llm
description: Joselowitz et al. 2025 — IRL aplicado a LLMs RLHF para reconstruir sus objetivos de entrenamiento; alcanza ~85% precisión en predecir preferencias.
metadata:
  type: summary
  tags: [irl, rlhf, reward-recovery, pilar-1, refuerzo-empirico]
  last_updated: 2026-05-12
---

# Joselowitz et al. (2025) — Insights from the Inverse: Reconstructing LLM Training Goals Through Inverse RL

> Aplicación directa de IRL a LLMs alineados con RLHF: **recuperar la reward function implícita** del modelo y validar que esa reward predice preferencias humanas con alta precisión. Refuerzo empírico clave para [[synthesis/pilar-1-irl-implicito]].

## Metadatos

- **Autores**: Joselowitz et al.
- **arXiv**: 2410.12491.
- **URL**: https://arxiv.org/abs/2410.12491
- **Año**: 2025 (preprint 2024).
- **Verificación**: cita aparece en [[summaries/acosta-2026-deep-research]] §5.5. Referencia confirmada por arXiv ID.

## Por qué importa para la presentación

Esta fuente provee **evidencia cuantitativa directa** de la tesis del Pilar 1:

> *"DPO/RLHF puede leerse como IRL implícito porque la política resultante codifica una reward function recuperable."*

Antes de este paper, esta era una afirmación teórica con respaldo formal (Ziebart 2008 ↔ Rafailov 2023). Joselowitz et al. la convierten en **afirmación empírica verificable**.

## Resumen del aporte

### Pregunta de investigación

Dado un LLM ya alineado con RLHF (donde la reward function `r` fue usada implícitamente durante PPO pero no se observa directamente), **¿puede IRL recuperar `r`** observando solo el comportamiento del modelo?

### Método

1. **Tomar un LLM RLHF** (e.g., Llama-2-Chat, GPT-3.5 calibrado).
2. **Generar trayectorias** del modelo en una distribución de prompts.
3. **Aplicar IRL** (variante MaxEnt adaptada a tokens) para recuperar `r̂` consistente con las trayectorias.
4. **Evaluar `r̂`** comparando sus predicciones de preferencia contra un conjunto held-out de preferencias humanas reales.

### Resultado central

`r̂` recuperada predice las preferencias humanas con **hasta 85% de precisión** en datasets de comparación (Anthropic HH, UltraFeedback subset). Esto significa:

- El modelo RLHF **sí codifica** internamente algo que se parece a la reward humana.
- IRL puede **extraer** esa reward sin acceso a los datos originales de entrenamiento.
- La reward recuperada **generaliza** a prompts/preferencias no vistas.

### Implicación teórica

Si IRL puede recuperar `r` de un modelo entrenado con RLHF, entonces **la política RLHF es funcionalmente equivalente** a una política producida por IRL con esa misma reward — exactamente lo que predice la dualidad MaxEnt-IRL ↔ DPO ([[synthesis/pilar-1-irl-implicito]] §2).

## Cómo encaja con el Pilar 1

El Pilar 1 argumenta que Self-Rewarding ejecuta IRL implícito. La cadena lógica es:

1. **RLHF KL-restringido tiene solución cerrada en forma MaxEnt** (Rafailov 2023, ec. 4).
2. **DPO invierte la ecuación** y recupera `r̂` desde `π` (Rafailov 2023, ec. 6).
3. **Self-Rewarding genera sus propios pares** vía LLM-as-a-Judge y aplica DPO (Yuan 2024).
4. **∴ Self-Rewarding = IRL implícito de su propio juez** ([[synthesis/pilar-1-irl-implicito]] §3).

Joselowitz et al. **completan el círculo empírico**:

5. **IRL aplicado a un LLM RLHF recupera una `r̂` que predice preferencias humanas con 85% acc** → la reward implícita codificada por la política **existe**, **es recuperable**, y **es semánticamente significativa**.

Esto valida que el paso 1-2 no es solo una elegancia matemática sino que captura algo real sobre cómo el modelo internamente representa las preferencias.

## Frase para defensa

> *"La afirmación 'el modelo codifica una reward function implícita' no es solo teórica. Joselowitz et al. (2025, arXiv:2410.12491) muestran empíricamente que IRL aplicado a LLMs RLHF recupera funciones de reward que predicen preferencias humanas con hasta 85% de precisión. Self-Rewarding, al usar DPO sobre pares generados internamente, está haciendo IRL sobre el comportamiento de su propio juez — y el objeto matemático que recupera es estructuralmente el mismo."*

## Limitaciones del paper

- **No aplica IRL específicamente a Self-Rewarding**, solo a RLHF clásico. La extensión es teórica.
- **Precisión < 100%**: 85% es alto pero no perfecto — la reward recuperada captura la mayoría, no todo.
- **Datasets de evaluación específicos**: la precisión puede variar con la distribución de prompts.
- **No accede a la reward original**: la "verdad de fondo" se aproxima por preferencias humanas, no por la `r` real de entrenamiento.

## Conexiones en el wiki

- [[synthesis/pilar-1-irl-implicito]] — refuerzo principal de la tesis.
- [[summaries/ziebart-2008-maxent-irl]] — fundamento formal MaxEnt-IRL.
- [[summaries/rafailov-2023-dpo]] — inversión de la solución cerrada.
- [[summaries/sun-vanderschaar-2024-irl-llm]] — survey/tutorial complementario.
- [[summaries/acosta-2026-deep-research]] §5.5 — fuente original de la referencia.
