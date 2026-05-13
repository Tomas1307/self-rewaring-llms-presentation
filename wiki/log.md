# Log del Wiki

Registro cronológico append-only. Prefijo consistente para grep:

```
## [YYYY-MM-DD] ingest | <título>
## [YYYY-MM-DD] query  | <pregunta>
## [YYYY-MM-DD] lint   | <resumen del pase>
```

---

## [2026-05-12] bootstrap | Wiki inicializado

- Estructura de carpetas creada: `raw/`, `concepts/`, `entities/`, `summaries/`, `synthesis/`.
- Stubs de `index.md`, `log.md`, `sources.md`.
- Schema definido en `CLAUDE.md`.
- Tesis: tres pilares (IRL, Cooperative multi-agent, DPO) intersectando en Self-Rewarding LMs.
- Idioma: español.

## [2026-05-12] research | Investigación bibliográfica primera ronda

- Cobertura arXiv: papers ancla (Yuan 2024, Rafailov 2023, Wu 2024 Meta-Rewarding), contexto RLHF (Christiano 2017, Ouyang 2022, Schulman 2017 PPO), métodos adyacentes (Bai 2022 Constitutional AI, Lee 2023 RLAIF, Chen 2024 SPIN, Wu 2024 SPPO, Zheng 2023 LLM-as-Judge), fundamentos IRL (Ng-Russell 2000, Abbeel-Ng 2004, Ziebart 2008 MaxEnt, Arora-Doshi 2021 survey), análogos multi-agente (Goodfellow 2014, CORY, MAE).
- Tres síntesis críticas redactadas: IRL ↔ Self-Rewarding, Cooperative multi-agent ↔ Self-Rewarding, DPO en el loop.

## [2026-05-12] research | Investigación bibliográfica segunda ronda (ampliación editorial)

- Cobertura extendida fuera de arXiv: JMLR (Wirth 2017), RSS (Sadigh 2017), EMNLP/ACL Anthology (Hong 2024 ORPO, Liang 2024 MAD, Kim 2024 Prospector, Zhang 2025 Process-SRLM, Li 2025 LLM-judge survey, Huang 2023), NeurIPS proceedings (Hejna-Sadigh 2023 IPL, Madaan 2023, Zelikman 2022), AISTATS (Azar 2023 IPO), ICML 2024 (Ethayarajh 2024 KTO, Du 2024 multi-agent debate).
- Nuevas direcciones cubiertas: variantes post-DPO (IPO, KTO, ORPO), self-improvement clásico (STaR, Self-Improve, Self-Refine, Saunders), multi-agent debate y cooperación (Du, Liang, Kim Prospector), preference-based IRL (Wirth, Sadigh, Hejna-Sadigh), críticas post-Yuan (Wang 2025 Temporal SR, Zhou 2025 SCIR, Zhang 2025 Process-SRLM, Shafayat 2025, Self-Preference Bias 2024).
- Verificaciones: β = 0.1 en Yuan 2024 confirmado; hiperparámetros completos de Yuan 2024 y Wu 2024 documentados en `sources.md`; autoría de Temporal Self-Rewarding confirmada; "Prospector" como Actor-Critic LLM confirmado; "Self-Rewarding Reasoning arxiv:2504+" NO existe — los candidatos cercanos son arXiv:2502.19613, arXiv:2506.08745, arXiv:2505.21444.
- Hallazgo central: "saturación tras 3 iteraciones" reportada por Yuan está respaldada por tres líneas independientes de crítica empírica + un mecanismo (self-preference bias).

## [2026-05-12] ingest | Bibliografía consolidada → `sources.md`, `index.md`

- `sources.md` poblado con 30+ entradas organizadas por categoría (ancla, RLHF, IRL, post-DPO, adyacentes, multi-agent, LLM-as-Judge, críticas, recursos pedagógicos).
- Top-7 priorizado: Yuan 2024, Rafailov 2023, Wu 2024 Meta-Rewarding, Wang 2025 Temporal SR, Azar 2023 IPO, Ziebart 2008 MaxEnt, Zheng 2023 LLM-as-Judge.
- `index.md` poblado con mapa completo de páginas previstas (tier-1 con resumen propio, tier-2 solo en `sources.md`).
- Próximos pasos: redactar `synthesis/pilar-{1,2,3}` y `synthesis/diagnostico-fallos`, luego summaries tier-1.

## [2026-05-12] ingest | Síntesis de los tres pilares + diagnóstico de fallos

- `synthesis/pilar-1-irl-implicito.md`: argumentación formal IRL ↔ Self-Rewarding vía MaxEnt-IRL/DPO con caveats honestos sobre los límites de la analogía.
- `synthesis/pilar-2-multi-agente-cooperativo.md`: framing como common-payoff cooperative game; contraste con GAN adversarial; respaldo en CORY, Multi-Agent Evolve, Prospector.
- `synthesis/pilar-3-dpo-en-loop.md`: mecánica precisa de la iteración M_t → M_{t+1}; rol de β; por qué DPO y no PPO.
- `synthesis/diagnostico-fallos.md`: integración de Wang 2025 (Temporal SR), Zhou 2025 (SCIR), Zhang 2025 (Process-SRLM), Shafayat 2025, Self-Preference Bias 2024, Liang 2024 (DoT) como cuerpo coherente de crítica.

## [2026-05-12] ingest | Deep research propio (Acosta 2026, RLHF→RLAIF→Self-Rewarding→DPO)

- Fuente: `Investigacion_RL_para_LLMs_RLHF_RLAIF_SelfRewarding_DPO.pdf` (11 pp). Documento de síntesis propia, no peer-reviewed. Carpeta `wiki/raw/` materializada con README; PDF físico pendiente de copia.
- **Páginas nuevas creadas**:
  - `summaries/acosta-2026-deep-research.md` — mapa narrativo del documento maestro.
  - `concepts/grpo.md` — Group Relative Policy Optimization (Shao 2024): PPO sin critic vía baseline grupal.
  - `entities/deepseek-r1.md` — R1-Zero y R1; comportamientos emergentes (auto-verif, reflexión, "aha moment").
  - `concepts/constitutional-ai.md` — CAI con sus dos fases (SL + RL); progenitor estructural de Self-Rewarding.
  - `summaries/joselowitz-2025-irl-llm.md` — IRL recupera reward de LLMs RLHF con 85% precisión.
  - `summaries/sun-vanderschaar-2024-irl-llm.md` — tutorial dedicado a la conexión IRL ↔ alineamiento.
  - `synthesis/comparacion-metodos.md` — tabla canónica de 7 métodos en 2 dimensiones (origen feedback × algoritmo).
- **Páginas sacadas de WIP**: `concepts/rlhf.md` (4 modelos en memoria, PPO en detalle, formulación matemática completa), `concepts/rlaif.md` (pipeline clásico + direct + CAI + relación con Self-Rewarding).
- **Páginas reforzadas**: `synthesis/pilar-1-irl-implicito.md` con §2.5 (validación empírica vía Joselowitz) y nuevas referencias en §5.
- **`sources.md`**: añadidos Shao 2024 (GRPO), DeepSeek-AI 2025 (R1), Joselowitz 2025, Sun-vanderSchaar 2024 (con [VERIFICAR] por arXiv ID incompleto), Acosta 2026 (doc propio).
- **`index.md`**: 6 entradas nuevas en concepts/entities/summaries/synthesis. RLHF/RLAIF/GRPO ya no aparecen como WIP.
- **Matices marcados (no contradicciones netas)**:
  - "DPO converge al óptimo de RLHF-PPO" del PDF (§3.3) requiere matiz: equivalencia teórica bajo Bradley-Terry, pero Azar 2023 muestra colapso bajo preferencias deterministas. Documentado en `summaries/acosta-2026-deep-research.md` §matices.
  - "RLAIF logra auto-mejora con juez = checkpoint" (Lee 2024) desdibuja la frontera RLAIF/Self-Rewarding. Self-Rewarding **formaliza e iteriza** una idea ya presente en RLAIF; no la inventa.
  - "rePIRL (Meta/UC Santa Barbara, 2025)" mencionado en el PDF §5.5 sin arXiv ID — marcado [VERIFICAR].
- **Cobertura ampliada**: la narrativa "evolución del pipeline" ahora incluye la rama algorítmica (PPO → GRPO → DPO) además de la rama de feedback (humanos → LLM externo → mismo modelo → meta-jerarquía).

## [2026-05-12] ingest | Compilación ecosistema Self-Rewarding (raw/self_rewarding.md)

- Fuente: `wiki/raw/self_rewarding.md` — compilación de 22 papers del ecosistema Self-Rewarding 2024-2025 con resúmenes propios.
- **13 papers nuevos añadidos a `sources.md`** (no estaban en la bibliografía previa):
  - **Self-Taught Evaluators** (Tianlu Wang 2024, Meta FAIR, arXiv:2408.02666) — el juez se auto-entrena.
  - **DNO** (Rosset 2024, Microsoft, arXiv:2404.03715) — Nash sobre preferencias generales sin Bradley-Terry.
  - **DICE** (Chen C. 2024, ICLR 2025, arXiv:2406.09760) — bootstrap usando la reward DPO implícita como juez.
  - **CREAM** (Wang Z. 2024, ICLR 2025, arXiv:2410.12735) — regularización por consistencia inter-iteración.
  - **LaTRO** (Salesforce 2024, arXiv:2411.04282) — razonamiento como variable latente.
  - **ScPO** (2024, arXiv:2411.04109) — self-consistency como señal de preferencia.
  - **SER** (Huang 2024, arXiv:2411.00418) — reward model que se auto-mejora.
  - **Self-Rewarding Correction Math** (Xiong 2025, arXiv:2502.19613) — corrección durante inferencia.
  - **SAR** (2025, arXiv:2509.05489) — híbrido verificable + self-aligned.
  - **CoVo** (Zhang K. 2025, arXiv:2506.08745) — reward geométrica de trayectorias.
  - **RLSR** (2025, arXiv:2505.08827) — el modelo genera sus propias tareas.
  - **Multi-Agent Evolve** (2025, arXiv:2510.23595) — multi-instancia co-evolución.
  - **CSR** (Zhou Y. 2024, NeurIPS, arXiv:2405.14622) — vision-language self-rewarding.
  - **Vision-SR1** (2025, arXiv:2508.19652) — VLM por descomposición percepción + razonamiento.
- **Páginas actualizadas**:
  - `synthesis/comparacion-metodos.md`: nueva §4.bis agrupando las 13+ variantes por dimensión del problema (sesgo acumulado, bootstrap implícito, preferencias generales, autonomía extendida, razonamiento, multimodal).
  - `synthesis/diagnostico-fallos.md`: nueva §2.1.bis con sesgo acumulado en modelos ~7B (CREAM) — distinto de saturación geométrica de Wang 2025.
- **Tier-2 ampliado en `sources.md`**: ahora incluye los 13 nuevos como respaldo citable.
- **Cobertura final del ecosistema**: 22 de 22 papers de la compilación del usuario están reflejados en el wiki.
