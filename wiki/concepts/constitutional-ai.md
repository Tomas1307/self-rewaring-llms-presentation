---
name: constitutional-ai
description: Constitutional AI (Anthropic) — pipeline en dos fases (SL + RLAIF) que usa un conjunto explícito de principios como sustituto de las preferencias humanas.
metadata:
  type: concept
  tags: [constitutional-ai, rlaif, anthropic, claude, principios, alineamiento]
  last_updated: 2026-05-12
---

# Constitutional AI (CAI)

> Pipeline de alineamiento desarrollado por Anthropic (Bai et al. 2022) para entrenar Claude. Reemplaza preferencias humanas implícitas por una **lista explícita de principios** ("constitución") que el propio modelo usa para auto-criticar y auto-revisar sus respuestas. Implementación más influyente de [[concepts/rlaif]].

## 1. Motivación

RLHF tradicional sufre de tres problemas que CAI quiere resolver:

1. **Preferencias humanas implícitas y opacas.** Los anotadores aplican sus propios criterios sin que estén documentados. Difícil auditar qué se está optimizando.
2. **Costo de anotación.** Cada round de mejora requiere nuevas anotaciones humanas.
3. **Sesgos no controlados.** Los anotadores introducen sesgos culturales y de longitud difíciles de detectar a posteriori.

CAI propone: **hacer explícitos los principios** y usar al propio LLM para aplicarlos.

## 2. Las dos fases

### 2.1. Fase SL (Supervised Learning) — auto-crítica y auto-revisión

Ciclo iterativo de generación-crítica-revisión:

```
Para cada prompt potencialmente problemático:
  1. Generar respuesta inicial con el modelo helpful (puede ser dañina).
  2. Seleccionar un principio aleatorio de la constitución.
  3. El modelo se auto-critica: "Identifica formas en las que la respuesta viola este principio."
  4. El modelo revisa: "Reescribe la respuesta eliminando los problemas identificados."
  5. Repetir 2-4 varias veces con principios distintos.
  6. La respuesta final revisada se usa como demostración para SFT.
```

**Output**: dataset de pares `(prompt, respuesta_revisada)` que se usa para fine-tunear el modelo base (SFT-CAI).

### 2.2. Fase RL (RLAIF propiamente)

Sobre el modelo SFT-CAI:

```
Para cada prompt:
  1. Generar pares de respuestas (y_A, y_B).
  2. El modelo, prompted con un principio constitucional aleatorio,
     evalúa cuál de las dos respuestas lo cumple mejor.
  3. Acumular preferencias AI-generadas.
  4. Entrenar reward model sobre estas preferencias.
  5. Aplicar PPO (o DPO) usando el reward model.
```

**Diferencia clave con RLHF**: el paso 2 es realizado por el LLM, no por humanos. El modelo aplica los principios explícitamente, no preferencias implícitas.

## 3. La constitución

Es una **lista de principios** en lenguaje natural. Ejemplos (sintéticos basados en el paper):

- *"Por favor selecciona la respuesta que sea menos dañina, ofensiva o tóxica."*
- *"¿Cuál respuesta evita más explícitamente apoyar contenido ilegal, no ético o inmoral?"*
- *"Selecciona la respuesta que un sabio, ético y sensato consejero daría."*
- *"Compara las respuestas en términos de ayuda en una situación de crisis emocional."*

La constitución completa tiene decenas de principios. En cada paso de auto-crítica o evaluación se selecciona uno **aleatorio** para evitar sobreajuste a un principio único.

**Propiedad crítica**: la constitución es **auditable y editable**. Si Anthropic decide cambiar un principio, edita la lista y re-entrena. Contraste con RLHF, donde cambiar la política implícita requiere re-anotar.

## 4. Por qué funciona

Tres condiciones que hacen que CAI no sea simplemente "el modelo se evalúa a sí mismo":

1. **Principios explícitos.** El juez no aplica su distribución preferida; aplica un criterio nombrado.
2. **Principio aleatorio por paso.** Distintos principios capturan distintas dimensiones de "buena respuesta". El promedio sobre principios reduce el sesgo de auto-preferencia ([[concepts/llm-as-a-judge]]).
3. **Modelo helpful preexistente.** El modelo que critica es ya un buen helpful assistant. No se evalúa con un modelo aleatorio sin alineación previa.

## 5. Relación con Self-Rewarding

CAI y Self-Rewarding ([[concepts/self-rewarding-loop]]) comparten estructura general (modelo evalúa modelo) pero difieren en aspectos clave:

| Aspecto | Constitutional AI | Self-Rewarding |
|---|---|---|
| Origen del criterio | Constitución explícita escrita por humanos | Rúbrica del prompt LLM-as-a-Judge |
| Diversidad de criterios | Decenas de principios randomizados | Rúbrica de 5 puntos fija |
| Iteración del modelo | Una sola pasada SL + una RL | Loop de modelos `M_t → M_{t+1}` |
| Quién mejora | Generador (el juez se mantiene estable) | Ambos (generador + juez mejoran juntos) |
| Riesgo de drift | Bajo (principios anclados) | Alto (juez se mueve con el modelo) |

CAI puede leerse como **el ancla** que Self-Rewarding no tiene: una constitución externa fija evita el drift del juez.

## 6. Relación con Cooperative Multi-Agent

CAI ejemplifica el framing cooperativo descrito en [[synthesis/pilar-2-multi-agente-cooperativo]]:

- El **generador** produce respuestas.
- El **crítico** identifica violaciones a la constitución.
- El **revisor** reescribe.
- El **juez** (en la fase RL) selecciona la mejor entre dos.

Estos roles los cumple el mismo modelo prompteado distinto, pero la **estructura cooperativa** es real. La constitución actúa como el "objetivo compartido" del juego cooperativo.

## 7. Limitaciones

1. **Calidad limitada por el modelo base.** Si el modelo no entiende un principio (e.g., "evitar daños sutiles"), no puede aplicarlo bien.
2. **Sesgos de la constitución.** Quien escribe los principios incrusta sus propios valores. Anthropic ha publicado borradores de su constitución para escrutinio, pero la opacidad parcial persiste.
3. **Principios contradictorios.** Si "ser útil" y "evitar daño" entran en conflicto, el modelo aplica heurísticas no documentadas.
4. **Costo cómputo.** Pese a eliminar la anotación humana, el ciclo SL crítica-revisión multiplica el cómputo respecto a SFT estándar.

## 8. Impacto e influencia

CAI fue **la prueba de concepto de RLAIF a escala**. Antes de Bai et al. 2022:

- RLHF dominaba (InstructGPT 2022).
- "AI feedback" era una idea teórica.

Después:

- RLAIF se sistematizó (Lee et al. 2023).
- Self-Rewarding extiende la lógica al límite (juez = generador, iterativo).
- Mucha investigación posterior en "AI critique", "self-correction", "debate" hereda la mecánica crítica-revisión.

CAI es **el ancestro estructural** de Self-Rewarding.

## 9. Lecturas relacionadas

- [[concepts/rlaif]] — paradigma general que CAI ejemplifica.
- [[concepts/self-rewarding-loop]] — extensión iterativa con juez = generador.
- [[concepts/llm-as-a-judge]] — el componente "juez" en detalle.
- [[synthesis/pilar-2-multi-agente-cooperativo]] — framing cooperativo aplicable.
- [[summaries/acosta-2026-deep-research]] §4.4 — fuente original del material en español.

## 10. Fuente principal

Bai et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*. arXiv:2212.08073. Entrada en [[sources]].
