---
name: llm-as-a-judge
description: Paradigma LLM-as-a-Judge — rúbrica aditiva, prompts, sesgos conocidos y cómo se internaliza en Self-Rewarding.
metadata:
  type: concept
  tags: [llm-as-a-judge, evaluacion, rubrica, sesgos]
  last_updated: 2026-05-12
---

# LLM-as-a-Judge

> Paradigma que usa un LLM fuerte como evaluador automático de outputs de otros (o el mismo) LLMs. Fundamento empírico: [[summaries/zheng-2023-llm-as-judge]]. Internalizado por [[summaries/yuan-2024-self-rewarding]].

## 1. La idea

Antes de Zheng 2023, evaluar calidad de outputs LLM requería anotación humana — costosa, lenta, sesgada por anotador. **LLM-as-a-Judge** propone: usar un LLM fuerte (GPT-4) como sustituto del anotador humano.

Zheng et al. 2023 muestran que GPT-4 acuerda con humanos > 80 % del tiempo en preferencia pairwise — comparable al acuerdo inter-humano. Eso hace LLM-as-a-Judge **viable como métrica automática**.

## 2. Dos modos de uso

### 2.1. Pairwise comparison

Dado `(x, y_A, y_B)`, el juez decide cuál es mejor:
```
Prompt: "Eres un evaluador imparcial. Lee la pregunta y las dos respuestas.
         Decide cuál respuesta es mejor o si están empatadas.
         Pregunta: {x}
         Respuesta A: {y_A}
         Respuesta B: {y_B}
         Respuesta: A / B / Empate"
```

Mitigación obligatoria del sesgo posicional: **swap A ↔ B** y promediar.

### 2.2. Pointwise scoring

Dado `(x, y)`, el juez asigna un score:
```
Prompt: "Evalúa la respuesta en una escala según los criterios.
         Pregunta: {x}
         Respuesta: {y}
         Criterios: [lista]
         Score: 0-5"
```

Esto es lo que [[summaries/yuan-2024-self-rewarding]] usa internamente.

## 3. La rúbrica de Yuan 2024 — 5 criterios binarios

Rúbrica aditiva: 5 criterios, cada uno 0 o 1, score final 0-5.

| # | Criterio | Pregunta operacional |
|---|---|---|
| 1 | **Relevancia** | ¿La respuesta aborda lo que el prompt pide? |
| 2 | **Cobertura** | ¿Cubre los puntos principales sin omitir nada importante? |
| 3 | **Utilidad** | ¿Le sería útil al usuario? ¿Resuelve su problema? |
| 4 | **Claridad** | ¿Está bien escrita, organizada y es fácil de seguir? |
| 5 | **Experticia** | ¿Demuestra conocimiento profundo / preciso del tema? |

**Por qué binario** (no escala continua):
- Más fácil de aplicar consistentemente.
- Menos sensible a calibración del juez.
- Score final `∈ {0,1,2,3,4,5}` es interpretable.

## 4. Sesgos conocidos (Zheng 2023)

| Sesgo | Descripción | Impacto en Self-Rewarding |
|---|---|---|
| **Posicional** | Preferencia por la primera o última opción. | Mitigable con swap aleatorio. |
| **Verbosidad** | Preferencia por respuestas más largas. | Genera *length inflation* iterativa en el loop. |
| **Self-enhancement** | Preferencia por outputs del propio modelo / familia. | **Amplificado al máximo** — juez = generador. |
| **Limitaciones de razonamiento** | El juez no puede verificar lo que él mismo no sabe. | Causa falla en dominios verificables (Zhang 2025). |

## 5. Self-enhancement bias en detalle

Documentado cualitativamente por Zheng 2023 y cuantitativamente por *Self-Preference Bias in LLM-as-a-Judge* (arXiv:2410.21819):

- GPT-4 prefiere outputs de modelos OpenAI a outputs equivalentes de otros.
- Mecanismo: **menor perplexity** sobre textos estilísticamente familiares → score más alto.
- Independiente de la calidad real.

**En Self-Rewarding** el juez es el **mismo modelo** que genera. El sesgo se amplifica máximamente: `M_t` genera con su distribución preferida → `M_t` como juez asigna alto score *precisamente porque* la respuesta es familiar → DPO refuerza esa distribución. El loop puede mejorar la auto-preferencia más que la calidad real.

**Mitigación parcial** en Yuan 2024: evaluar finalmente con GPT-4 Turbo (modelo distinto a Llama-2-70B) en AlpacaEval. Pero el sesgo en la construcción interna de pares persiste.

## 6. Variantes y mejoras

- **Panel de jueces** (multi-judge ensemble): promediar scores de varios jueces independientes. Costoso pero reduce sesgo individual.
- **Meta-Rewarding** ([[summaries/wu-2024-meta-rewarding]]): agrega un meta-juez que evalúa los juicios del juez. Reduce sesgo posicional.
- **Multi-Agent Debate** (Liang 2024, MAD): dos agentes con posiciones contrarias moderados por un juez separado. Mitiga la *degeneración del pensamiento*.
- **Process-SRLM** (Zhang 2025): juicio step-wise para razonamiento matemático.
- **Reference-based scoring**: pedir al juez que compare contra una respuesta de referencia conocida. Más estable.

## 7. Prompts ejemplares

Prompt típico para scoring aditivo (paráfrasis del estilo de Yuan 2024):

```
Eres un evaluador experto. Evalúa la siguiente respuesta según 5 criterios.
Para cada criterio, decide 0 (no cumple) o 1 (cumple). Suma los puntos.

Criterios:
1. Relevancia (0/1)
2. Cobertura (0/1)
3. Utilidad (0/1)
4. Claridad (0/1)
5. Experticia (0/1)

Pregunta: {x}
Respuesta: {y}

Explica tu razonamiento brevemente, luego termina con:
"Score: X/5"
```

(El prompt exacto de Yuan 2024 está en el apéndice del paper. Esta es la estructura.)

## 8. Benchmarks que usan LLM-as-a-Judge

- **MT-Bench** ([[entities/mt-bench]]): 80 preguntas multi-turn en 8 categorías; juez = GPT-4.
- **AlpacaEval 2.0** ([[entities/alpaca-eval]]): comparación pairwise vs GPT-4 Turbo; length-controlled.
- **Arena-Hard** ([[entities/arena-hard]]): prompts difíciles de Chatbot Arena.

## 9. Limitaciones fundamentales

1. **El juez no es objetivo**: hereda los sesgos del modelo subyacente.
2. **El juez tiene techo cognitivo**: no puede evaluar correctamente problemas que él mismo no podría resolver.
3. **Reproducibilidad**: GPT-4 cambia entre versiones; lo que medías el mes pasado no es lo mismo hoy.
4. **Self-enhancement** en Self-Rewarding: el bug más serio. No tiene solución limpia sin un juez externo.

## 10. Lecturas relacionadas

- [[summaries/zheng-2023-llm-as-judge]] — paper fundador.
- [[summaries/yuan-2024-self-rewarding]] — internalización del paradigma.
- [[summaries/wu-2024-meta-rewarding]] — extensión con meta-judge.
- [[concepts/preference-pair]] — cómo los scores se traducen en pares.
- [[synthesis/diagnostico-fallos]] §2.4 — el self-enhancement bias amplificado.
- [[entities/mt-bench]], [[entities/alpaca-eval]] — benchmarks que dependen de este paradigma.
