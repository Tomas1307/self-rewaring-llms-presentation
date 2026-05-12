# Self-Rewarding LLMs — Presentación

Presentación de maestría para un curso de Reinforcement Learning sobre **Self-Rewarding Language Models**. Se construye en dos fases: primero un wiki de conocimiento persistente mantenido por el LLM (en español), luego un deck en HTML al estilo de las presentaciones keynote de OpenAI (también en español), destilado del wiki.

Este archivo es el **schema** del wiki y la presentación. Define cómo se organiza el wiki, qué convenciones seguir y cómo debe verse la presentación. Léelo al inicio de cada sesión.

## Idioma

- **Wiki: español.** Todo el contenido de las páginas del wiki está en español académico. Términos técnicos estándar pueden permanecer en inglés (reward model, preference pair, LLM-as-a-Judge, fine-tuning, etc.), pero la prosa es en español.
- **Presentación: español.** Todo el texto visible para la audiencia es español. Las fórmulas matemáticas son universales.
- **Este archivo (CLAUDE.md): inglés.** Es un documento de instrucciones interno entre el usuario y el LLM, no contenido para audiencia.
- **Citas y referencias bibliográficas: idioma original** (no traducir títulos de papers).

## Tesis de la presentación

La presentación enmarca Self-Rewarding LMs como la intersección de **tres pilares del curso**:

1. **Inverse RL (IRL)** — el modelo aprende una *reward function implícita* a partir de su propio comportamiento, en lugar de recibirla externamente.
2. **Cooperative multi-agent RL** — aunque es un solo modelo, opera como dos agentes cooperando: un generador y un juez (LLM-as-a-Judge), mejorándose mutuamente en cada iteración.
3. **Direct Preference Optimization (DPO)** — mecanismo de optimización de política directamente sobre pares de respuestas preferidas, sin entrenar un reward model separado.

Mensaje central: este paradigma elimina la dependencia de anotadores humanos (RLHF) y de modelos externos (RLAIF), produciendo un loop de auto-mejora donde el modelo se vuelve mejor generador *y* mejor juez simultáneamente.

El paper ancla es **Yuan et al. 2024 ("Self-Rewarding Language Models", Meta, arXiv:2401.10020)**, complementado por DPO (Rafailov 2023) y posiblemente Meta-Rewarding (Wu 2024). Confirmar con el usuario antes de cerrar la lista.

---

## Fase 1 — El Wiki (fase actual)

### Idea central

El wiki es un **artefacto persistente y acumulativo** — no RAG. Cuando llega una nueva fuente no se indexa simplemente; se lee, se extrae lo importante y se integra al wiki existente — actualizando páginas de entidades, revisando resúmenes temáticos, marcando contradicciones, fortaleciendo la síntesis. El conocimiento se compila una vez y se mantiene al día, nunca se rederiva por consulta.

El usuario dirige las fuentes, la exploración y las preguntas. **El LLM escribe y mantiene cada página del wiki.** El usuario rara vez edita páginas a mano.

### Arquitectura

Tres capas:

1. **Raw sources** (`wiki/raw/`) — papers (PDF), artículos, transcripciones. Inmutables. Solo lectura.
2. **El wiki** (`wiki/`) — archivos markdown generados por el LLM. Resúmenes, páginas de entidades, conceptos, comparaciones, síntesis. Propiedad exclusiva del LLM.
3. **El schema** — este archivo (`CLAUDE.md`). Coevoluciona con el wiki.

### Layout

```
self-rewaring-llms-presentation/
├── CLAUDE.md                  # schema (este archivo)
├── wiki/
│   ├── index.md               # catálogo de contenido — toda página con resumen de 1 línea
│   ├── log.md                 # registro cronológico append-only
│   ├── sources.md             # bibliografía (paper + arXiv ID + fecha de ingesta)
│   ├── raw/                   # fuentes inmutables (PDFs, etc.)
│   ├── concepts/              # páginas conceptuales (DPO, RLHF, LLM-as-Judge, IRL, ...)
│   ├── entities/              # papers, modelos, autores como ciudadanos de primera clase
│   ├── summaries/             # un resumen por fuente ingerida
│   └── synthesis/             # análisis transversal (pipeline, comparaciones, conexiones con los 3 pilares, preguntas abiertas)
└── presentation/              # Fase 2 — NO crear hasta que el wiki esté aprobado
```

### Archivos obligatorios

**`wiki/index.md`** — catálogo orientado a contenido. Cada página listada con link, resumen de una línea y metadatos opcionales (fecha, conteo de fuentes, estado). Organizado por categoría (concepts / entities / summaries / synthesis). **Actualizar en cada ingest.** Al responder una consulta, leer este archivo primero para localizar páginas relevantes.

**`wiki/log.md`** — cronológico, append-only. Cada ingest, query y lint se registra con prefijo consistente para que sea grep-eable:

```
## [YYYY-MM-DD] ingest | <título de la fuente>
## [YYYY-MM-DD] query  | <pregunta de una línea>
## [YYYY-MM-DD] lint   | <resumen del pase>
```

Cuerpo: 2–5 líneas sobre qué pasó, qué páginas se tocaron, qué se aprendió.

**`wiki/sources.md`** — bibliografía. Una entrada por fuente: título, autores, año, arXiv ID / DOI / URL, fecha de ingesta, link a su página de resumen.

### Convenciones de páginas

- Solo markdown. Frontmatter YAML opcional para `tags`, `last_updated`, `source_count`.
- Cross-link generosamente con `[[wikilink]]` — compatible con Obsidian.
- Citar fuentes para toda afirmación técnica no obvia. Notación: `(Autor Año, §X)` o footnotes `[^cite]`. Ser consistente dentro de una página.
- Al introducir una fórmula: (1) definir cada símbolo, (2) intuición en lenguaje llano, (3) link a la sección del paper.
- Mantener páginas enfocadas. Si una pasa de ~300 líneas, considerar dividir.
- Las páginas se revisan cuando nuevas fuentes contradicen o refinan. Registrar revisiones en `log.md`.

### Operaciones

**Ingest** — el usuario deja una fuente en `wiki/raw/` y pide procesarla. Flujo:

1. Leer la fuente de principio a fin.
2. Discutir los hallazgos clave con el usuario antes de escribir.
3. Escribir un resumen en `wiki/summaries/`.
4. Actualizar o crear páginas relevantes en `wiki/concepts/` y `wiki/entities/`.
5. Actualizar `wiki/index.md` con páginas nuevas.
6. Actualizar `wiki/sources.md`.
7. Añadir entrada en `wiki/log.md`.
8. Marcar contradicciones con páginas existentes — no sobrescribir en silencio.

Un solo ingest puede tocar 10–15 páginas. Es lo esperado.

**Query** — el usuario hace una pregunta. Leer `index.md`, identificar páginas relevantes, leerlas, sintetizar con citas. Si la respuesta es sustancial (una comparación, una derivación, una conexión nueva), **ofrecer archivarla** como página nueva en `synthesis/` para que las exploraciones acumulen.

**Lint** — el usuario pide un health check. Buscar: contradicciones entre páginas, afirmaciones obsoletas, páginas huérfanas (sin inbound links), conceptos referenciados sin página propia, cross-references faltantes, lagunas donde un web search ayudaría. Reportar; no auto-corregir sin confirmación.

### Alcance temático

El wiki debe cubrir (construir conforme lleguen fuentes — no stubear todo de entrada):

- **Fundamentos RL / RLHF** — policy gradient, reward modeling, PPO vs DPO, el cuello de botella del feedback humano.
- **Inverse RL** — IRL clásico (Ng & Russell, MaxEnt IRL), cómo el "reward implícito" de Self-Rewarding se conecta (o no) con IRL estricto.
- **DPO** — derivación, intuición, el rol de β, ventajas vs PPO.
- **LLM-as-a-Judge** — el rubric, los prompts, sesgos conocidos.
- **El loop Self-Rewarding** — seed → generar → auto-juzgar → pares de preferencia → DPO → iterar. Cada etapa con inputs, outputs y los prompts/templates involucrados.
- **Fórmulas clave** — DPO loss, judge scoring, construcción de pares de preferencia.
- **Cooperative multi-agent framing** — análogos formales (juegos cooperativos, actor-critic, GANs) y los límites de la analogía.
- **Resultados empíricos** — AlpacaEval, MT-Bench, qué mejora, qué se estanca.
- **Limitaciones** — reward hacking, sesgo del juez, distribution drift entre iteraciones.
- **Métodos adyacentes** — RLHF, RLAIF, Constitutional AI, SPIN, Meta-Rewarding — para posicionamiento.

---

## Fase 2 — La Presentación (NO empezar hasta que el wiki esté aprobado)

### Requisitos de formato (constraints duros)

- **Un archivo HTML por slide.** `presentation/slides/01-titulo.html`, `02-...html`, etc.
- **Una plantilla compartida.** Mismo esqueleto (header/footer/layout/grid), solo cambia el *contenido* entre slides.
- **CSS compartido** para color, tipografía y spacing. Un solo `presentation/styles/main.css` importado por cada slide. Paleta de colores idéntica en todas las slides.
- **Una slide final de recapitulación** que reúne toda la información clave del deck en una sola vista (la slide "todo en una pantalla").
- Un `index.html` raíz que navega entre slides (flechas de teclado + click). Cada slide HTML también debe abrirse standalone.

### Audiencia y profundidad

- **Audiencia**: profesor y compañeros del curso de RL de maestría. Asumir fluidez en policy gradient, value functions y RLHF básico. Refrescos de una línea son suficientes; no re-enseñar PPO desde cero.
- **Profundidad**: *conceptual + fórmulas clave.* Recorrer el pipeline visualmente. Incluir las 2–3 ecuaciones más importantes (DPO loss, judge scoring, construcción de preferencias). No derivación matemática completa. Toda afirmación debe ser defendible si un compañero pregunta.

### Estilo visual — estética keynote de OpenAI

Marcas distintivas a emular:

- **Whitespace generoso.** Una idea por slide — a menudo una frase o un diagrama. Nunca apretado.
- **Paleta restringida.** Casi-negro sobre off-white (o invertido) con un solo color de acento usado con cuentagotas. Sin gradientes, sin sombras, sin floritura decorativa. Fijar la paleta en `main.css` desde el día 1 y no desviarse.
- **La tipografía es el diseño.** Headlines en sans-serif grande y confiado (familia Söhne / Inter / Helvetica Neue). Tracking ajustado, contraste de pesos cuidadoso. Body pequeño y discreto.
- **Diagramas como line art.** Trazos finos, mucho espacio negativo, etiquetas en la misma tipografía del body. Cajas y flechas, nada 3D ni glossy. SVG preferido.
- **Movimiento sutil.** Cross-fades y traslaciones pequeñas. Nada de transiciones llamativas, parallax ni fondos animados.
- **Math renderizado limpio** con KaTeX o MathJax. Las ecuaciones tienen su propio espacio de respiración.
- **Código (si lo hay)**: monoespaciada, grande, tema de syntax discreto. Elemento visual de primer nivel, no decoración.

Ante la duda: quitar algo. Las slides de OpenAI se sienten casi vacías — esa es la meta.

### Stack

- HTML / CSS / JS planos. Sin frameworks (no Reveal.js, no Slidev, no React).
- KaTeX para math.
- SVG para diagramas (a mano o generados, archivos finales en `presentation/assets/`).
- Un solo JS compartido (`presentation/scripts/nav.js`) para navegación con flechas.

---

## Normas de trabajo

- **No empezar Fase 2 hasta que el usuario apruebe el wiki explícitamente.**
- **No cerrar la lista de papers ancla sin discusión.** Candidato principal: Yuan et al. 2024.
- **Citar TODA afirmación técnica.** Mejor sobrecitar que subcitar — es trabajo académico.
- **Actualizar `index.md` y `log.md` en cada ingest.** No negociable.
- **Discutir antes de escribir.** Conversar lo que contiene una nueva fuente antes de mutar archivos.
- **Marcar contradicciones; no sobrescribir en silencio.** Cuando una fuente nueva contradice una página, llevarlo al usuario y resolver juntos.
- **Si una afirmación no se puede verificar en una fuente primaria, marcarla como `[VERIFICAR]`** en lugar de inventar.
