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
