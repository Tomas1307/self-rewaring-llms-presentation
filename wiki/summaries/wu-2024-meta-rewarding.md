---
name: wu-2024-meta-rewarding
description: Resumen de Meta-Rewarding (Wu et al. 2024) — extensión natural de Self-Rewarding con un tercer rol meta-judge, arXiv:2407.19594.
metadata:
  type: summary
  tags: [meta-rewarding, self-rewarding, multi-agent, dpo, llama-3]
  last_updated: 2026-05-12
---

# Wu et al. (2024) — Meta-Rewarding Language Models

## Cita

Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, Sainbayar Sukhbaatar (Meta FAIR / UC Berkeley / NYU). *Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge*. arXiv:2407.19594, julio 2024.

- **arXiv**: https://arxiv.org/abs/2407.19594

## Resumen técnico

Meta-Rewarding extiende [[summaries/yuan-2024-self-rewarding]] con un **tercer rol**: el modelo no solo actúa (genera) y juzga, sino que además **meta-juzga sus propios juicios**.

En cada iteración `t`, `M_t` opera como:

1. **Actor** — genera N respuestas `y_1, ..., y_N` para un prompt `x`.
2. **Judge** — asigna scores `s_i(y_i | x)`; los scores producen pares de preferencia sobre **respuestas** para entrenar al actor (igual que en Yuan 2024).
3. **Meta-judge** — compara *cadenas de razonamiento del judge* sobre el mismo par; selecciona el mejor juicio; produce pares de preferencia sobre **juicios** para entrenar al judge mediante un canal DPO independiente.

Adicionalmente introduce un mecanismo de **length-control** para mitigar el sesgo de longitud que arrastra Self-Rewarding original.

## Aporte clave para la presentación

Tres aportes complementarios:

1. **Validación empírica del framing cooperative multi-agent**: Meta-Rewarding desacopla formalmente los roles "generator" y "judge" como sub-agentes con sus propias señales de mejora (dos canales DPO). Esto fortalece [[synthesis/pilar-2-multi-agente-cooperativo]].

2. **Diagnóstico del reward hacking**: el paper documenta explícitamente el reward hacking del juez interno como problema central de Self-Rewarding, justificando críticamente el diseño. Insumo para [[synthesis/diagnostico-fallos]].

3. **Mejora cuantitativa sobre Yuan 2024**: muestra que separar la señal del juez mejora resultados sin requerir anotaciones humanas adicionales.

## Fórmulas centrales

Reutiliza la pérdida DPO de [[summaries/rafailov-2023-dpo]] en dos canales:

**Canal Actor (igual a Yuan 2024)**:
```
L_actor(θ) = L_DPO(θ; π_ref=M_t)  sobre pares (x, y_w, y_l)
                                  donde y_w, y_l vienen del juez M_t.
```

**Canal Judge (nuevo)**:
```
L_judge(θ) = L_DPO(θ; π_ref=M_t)  sobre pares (x, y, c_w, c_l)
                                  donde c_w, c_l son juicios (cadenas de razonamiento del judge),
                                  comparados por el meta-judge M_t.
```

El meta-judge compara dos juicios `c_1, c_2` sobre el mismo par `(x, y)` y selecciona el más coherente / correctamente argumentado.

## Hiperparámetros verificados (apéndice)

- **DPO**: 10 épocas, LR = 5e-6, β = 0.1.
- **SFT**: 10 épocas, LR = 5e-8, batch global 32, cosine schedule, checkpoint de época 5 seleccionado.
- **Iteraciones**: 4 (más que las 3 de Yuan 2024).
- **Pool de prompts**: 20.000 prompts generados por Llama-2-70B-Chat con 8-shot.
- **Modelo base**: **Llama-3-8B-Instruct** (no 70B — más eficiente que Yuan 2024).

## Resultados empíricos clave

**AlpacaEval 2.0 length-controlled win rate** (vs GPT-4 Turbo):

| Modelo | Win rate |
|---|---|
| Llama-3-8B-Instruct (baseline) | 22.9 % |
| Después de 4 iteraciones Meta-Rewarding | **39.4 %** |

A 39.4 % supera a GPT-4-0314 y se aproxima a Claude 3 Opus en ese benchmark.

**Arena-Hard**:

| Modelo | Win rate |
|---|---|
| Baseline | 20.6 % |
| Meta-Rewarding | **29.1 %** |

**Habilidad del juez**: correlación con humanos mejora junto con la del actor, **a tasa mayor** que en Self-Rewarding original (donde el juez mejora solo como subproducto).

## Limitaciones conocidas

1. **Sesgo posicional del meta-judge persiste** aunque atenuado vs el juez de un solo nivel.
2. **Inflación de puntajes**: el judge tiende a inflar scores, lo que acelera la saturación del par chosen-rejected.
3. **Sin garantía formal de convergencia**: sigue siendo un loop iterativo finito; no hay teorema de convergencia.
4. **Más cómputo**: cuatro iteraciones con dos canales DPO ≈ 8× el costo de Self-Rewarding por iteración. Pero el modelo base es 8B en vez de 70B, así que en términos absolutos es más barato.
5. **Length inflation parcialmente mitigada**: el mecanismo de length-control reduce pero no elimina el sesgo.

## Conexiones en el wiki

- [[summaries/yuan-2024-self-rewarding]] — paper base que este extiende.
- [[summaries/rafailov-2023-dpo]] — DPO usado en ambos canales.
- [[concepts/llm-as-a-judge]] — paradigma del juez, ahora con meta-juez.
- [[concepts/self-rewarding-loop]] — el loop extendido a tres roles.
- [[synthesis/pilar-2-multi-agente-cooperativo]] — Meta-Rewarding como respaldo formal del framing cooperativo.
- [[synthesis/diagnostico-fallos]] — el paper documenta el reward hacking que motiva su diseño.
