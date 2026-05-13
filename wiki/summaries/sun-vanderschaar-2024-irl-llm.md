---
name: sun-vanderschaar-2024-irl-llm
description: Sun & van der Schaar 2024 — survey/tutorial que posiciona el alineamiento de LLMs dentro del marco formal de Inverse RL.
metadata:
  type: summary
  tags: [irl, llm-alignment, survey, tutorial, pilar-1, respaldo-teorico]
  last_updated: 2026-05-12
---

# Sun & van der Schaar (2024) — Inverse RL Meets LLM Alignment

> Survey/tutorial que articula formalmente la conexión IRL ↔ alineamiento de LLMs. Respaldo académico para [[synthesis/pilar-1-irl-implicito]]: si la comunidad RL ya lee RLHF/DPO como IRL, la presentación puede apoyarse en ese consenso.

## Metadatos

- **Autores**: Hao Sun, Mihaela van der Schaar.
- **Tipo**: tutorial / survey (no peer-reviewed estándar, pero coautora reconocida en ML).
- **Verificación**: citado en [[summaries/acosta-2026-deep-research]] §9 referencias. **[VERIFICAR arXiv ID exacto]** — la cita del PDF no incluye número; búsqueda apunta a contenido tutorial publicado en 2024 por van der Schaar Lab.
- **URL aproximada**: probable lab page o arXiv 2024.

## Por qué importa para la presentación

No aporta resultados experimentales nuevos. Su valor es **académico y argumentativo**:

1. **Consolidación**: muestra que el campo RL académico **acepta** la lectura RLHF-como-IRL como marco unificador.
2. **Vocabulario formal**: provee terminología consistente (apprenticeship learning, MaxEnt-IRL, preference-based IRL) aplicable al alineamiento.
3. **Bibliografía estructurada**: organiza los puentes IRL ↔ LLM en una sola referencia consultable.

Es el tipo de fuente que se cita en una slide como "ver también: Sun & van der Schaar 2024" sin desarrollarla en detalle.

## Contenido principal (inferido del título y la cita del deep research)

El tutorial cubre:

- **IRL clásico** (Ng-Russell, Abbeel-Ng, Ziebart) como contexto.
- **El problema de alineamiento de LLMs** formulado como IRL: el "experto" son los humanos vía preferencias.
- **RLHF como MaxEnt-IRL con prior** — la misma observación de [[synthesis/pilar-1-irl-implicito]] §2.
- **DPO como IRL en forma cerrada** — equivalente a Rafailov 2023.
- **Direcciones futuras**: preference-based IRL, self-supervised reward inference, conexión con IRL inverso jerárquico.

## Cómo encaja con el Pilar 1

[[synthesis/pilar-1-irl-implicito]] argumenta que Self-Rewarding es IRL funcional. El respaldo bibliográfico actual de esa página es:

- **Formal**: Ziebart 2008 + Rafailov 2023 (equivalencia matemática).
- **Empírico**: Joselowitz 2025 (recuperación experimental — ver [[summaries/joselowitz-2025-irl-llm]]).
- **Pedagógico**: Lambert RLHF Book cap. 5.

Sun & van der Schaar añade **respaldo académico**: un tutorial específicamente dedicado a la conexión, escrito por una investigadora establecida en RL/healthcare-ML. Esto neutraliza objeciones del tipo "esto es solo una analogía suelta" — la analogía está articulada en literatura tutorial.

## Frase para defensa (uso conservador)

> *"La conexión IRL ↔ alineamiento de LLMs no es una invención de esta presentación. Sun y van der Schaar (2024) la articulan como tutorial completo, y trabajos empíricos como Joselowitz et al. (2025) la validan cuantitativamente. La novedad de Self-Rewarding es que esta IRL implícita se vuelve auto-referencial: el experto es el propio modelo."*

## Limitaciones

- **Sin arXiv ID confirmado.** [VERIFICAR antes de citar en la presentación] — la cita del deep research no incluye número. Buscar autoría confirmada antes de usar en slide.
- **Tutorial, no paper de investigación.** No aporta resultados nuevos; solo organización del campo.
- **Posible solapamiento** con materiales del RLHF Book de Lambert. Verificar qué añade que no esté ya en Lambert.

## Acción pendiente

- [ ] Confirmar arXiv ID o URL canónica antes de incluir como referencia formal en slide.
- [ ] Evaluar si conviene mantener como respaldo o reemplazar por Lambert RLHF Book §5 que ya está usado.

## Conexiones en el wiki

- [[synthesis/pilar-1-irl-implicito]] — respaldo académico de la tesis.
- [[summaries/joselowitz-2025-irl-llm]] — respaldo empírico complementario.
- [[summaries/ziebart-2008-maxent-irl]], [[summaries/rafailov-2023-dpo]] — fundamento formal.
- [[summaries/acosta-2026-deep-research]] §5.5 y §9 — fuente original de la referencia.
