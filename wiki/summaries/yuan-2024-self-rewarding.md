---
name: yuan-2024-self-rewarding
description: Resumen del paper ancla — Yuan et al. 2024, Self-Rewarding Language Models (Meta FAIR), arXiv:2401.10020.
metadata:
  type: summary
  tags: [paper-ancla, self-rewarding, dpo, llm-as-a-judge, llama-2]
  last_updated: 2026-05-12
---

# Yuan et al. (2024) — Self-Rewarding Language Models

## Cita

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston (Meta FAIR / NYU). *Self-Rewarding Language Models*. arXiv:2401.10020, enero 2024. ICML 2024 (PMLR vol. 235).

- **arXiv**: https://arxiv.org/abs/2401.10020
- **ICML**: https://proceedings.mlr.press/v235/yuan24d.html
- **OpenReview**: https://openreview.net/pdf?id=0NphYCmgua

## Resumen técnico

El paper articula un paradigma de auto-mejora donde un mismo LLM cumple dos roles durante el entrenamiento:

1. **Generador** de respuestas a instrucciones.
2. **Juez** de su propia producción mediante *LLM-as-a-Judge prompting*.

**Motivación**: el RLHF clásico tiene un doble cuello de botella — la anotación humana es cara y limita la calidad al techo del anotador, y un reward model congelado no puede co-evolucionar con la política. Self-Rewarding intenta saltar ambos límites.

**Pipeline**:

- Modelo base: **Llama-2-70B**.
- SFT inicial con dos conjuntos semilla:
  - *IFT (Instruction Fine-Tuning)*: pares (prompt, respuesta) de Open Assistant.
  - *EFT (Evaluation Fine-Tuning)*: ejemplos del LLM evaluando con rúbrica aditiva de 5 puntos (relevancia, cobertura, utilidad, claridad, experticia).
- En cada iteración `t`:
  1. `M_t` genera nuevos prompts vía *self-instruction creation*.
  2. Muestrea N=4 respuestas con T=0.7, top-p=0.9.
  3. Puntúa cada respuesta tres veces y promedia.
  4. Construye pares de preferencia tomando `argmax` y `argmin`.
  5. Entrena `M_{t+1}` mediante **DPO** sobre esos pares.

Tras tres iteraciones, el modelo mejora **tanto como generador como juez**.

## Aporte clave para la presentación

Es el paper ancla. **Instancia los tres pilares simultáneamente**:

- **IRL implícito**: el modelo recupera vía DPO la `r̂_θ(x,y) = β·log(π_θ/π_ref)` que mejor explica las decisiones de su propio juez ([[synthesis/pilar-1-irl-implicito]]).
- **Cooperative multi-agent**: actor y juez como sub-agentes del mismo backbone, mejora win-win ([[synthesis/pilar-2-multi-agente-cooperativo]]).
- **DPO**: motor del loop, sin reward model separado, sin PPO ([[synthesis/pilar-3-dpo-en-loop]]).

## Fórmulas centrales

Las del paper son heredadas de [[summaries/rafailov-2023-dpo]] (pérdida DPO).

**Rúbrica del juez** — score aditivo:

```
s(y | x) = Σ_{i=1..5} c_i ,  con c_i ∈ {0, 1}
         ∈ {0, 1, 2, 3, 4, 5}
```

donde los 5 criterios binarios son:
1. relevancia,
2. cobertura,
3. utilidad,
4. claridad,
5. experticia.

**Construcción de pares**:

```
(x, y_w, y_l)  con  y_w = argmax_y s(y|x),  y_l = argmin_y s(y|x)
```

Pares con `s(y_w) = s(y_l)` se descartan.

## Hiperparámetros verificados (apéndice)

- **DPO**: β = 0.1, LR inicial 1e-6 con decay a 1e-7, batch size 16, dropout 0.1.
- **SFT**: LR 5.5e-6 → 1.1e-6 con cosine decay.
- **Generación**: T = 0.7, top-p = 0.9, N = 4 respuestas por prompt.
- **Validación / early stopping**: cada 200 steps con Claude 2 sobre 253 ejemplos.
- **Tamaños de datasets**:
  - AIFT(M_1) = 3.964 pares de preferencia → M_2.
  - AIFT(M_2) = 6.942 pares → M_3.

## Resultados empíricos clave

**AlpacaEval 2.0 vs GPT-4 Turbo** (length-controlled win rate):

| Modelo | Win rate |
|---|---|
| M_1 (solo SFT con IFT+EFT) | 9.94 % |
| M_2 (DPO sobre AIFT(M_1)) | 15.38 % |
| M_3 (DPO sobre AIFT(M_2)) | **20.44 %** |

**Comparación external** (M_3): supera a Claude 2, Gemini Pro y GPT-4-0613 en el leaderboard de AlpacaEval 2.0 en el momento de publicación.

**Head-to-head iteraciones**:

| Comparación | Win rate primer modelo |
|---|---|
| M_2 vs M_1 | 55.5 % gana M_2 (M_1: 11.7 %) |
| M_3 vs M_2 | 47.7 % gana M_3 (M_2: 12.5 %) |
| M_3 vs SFT baseline | 62.5 % gana M_3 |

**Habilidad de juez**: correlación con humanos mejora monotónicamente entre iteraciones (reportado sin entrenamiento explícito del rol juez — confirma la "doble mejora").

## Limitaciones conocidas

1. **Solo 3 iteraciones evaluadas.** Trabajos posteriores muestran saturación o degradación más allá ([[summaries/wang-2025-temporal-sr]]).
2. **Sesgo de longitud del juez.** Respuestas más largas reciben más puntaje → *length inflation* acumulativa entre iteraciones.
3. **Reward hacking sobre el juez interno** — documentado explícitamente por [[summaries/wu-2024-meta-rewarding]] como motivación para introducir el meta-judge.
4. **No mejora explícita del juez**: el juez mejora como subproducto, no como objetivo entrenado.
5. **Dominio**: instruction-following abierto. No evalúa razonamiento matemático verificable, generación de código, dominios estrictos. Zhang 2025 (Process-SRLM, arXiv:2503.03746) muestra falla en matemáticas.
6. **Self-preference bias** ([[summaries/zheng-2023-llm-as-judge]] y arXiv:2410.21819): el juez prefiere outputs similares al propio modelo, lo que infla el reward implícito.

## Conexiones en el wiki

- [[concepts/self-rewarding-loop]] — pseudocódigo y mecánica detallada del loop.
- [[concepts/llm-as-a-judge]] — paradigma y rúbrica detallada.
- [[concepts/dpo]] — pérdida usada en el paso 5 del loop.
- [[summaries/rafailov-2023-dpo]] — papel del cual hereda DPO.
- [[summaries/wu-2024-meta-rewarding]] — extensión directa al rol meta-judge.
- [[summaries/wang-2025-temporal-sr]] — crítica empírica obligatoria.
- [[synthesis/pilar-1-irl-implicito]], [[synthesis/pilar-2-multi-agente-cooperativo]], [[synthesis/pilar-3-dpo-en-loop]] — los tres pilares de la presentación.
- [[synthesis/diagnostico-fallos]] — modos de fallo del loop.
