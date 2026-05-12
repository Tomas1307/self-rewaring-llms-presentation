# Bibliografía

Una entrada por fuente. Organizada por categoría temática. Mantener ordenada por relevancia dentro de cada sección.

**Convención de cita:** `Autores (año). *Título*. Venue. arXiv:ID / DOI.`

**Top-7 priorizado** para la presentación: [[#priorización]] al final.

---

## 1. Papers ancla

### Yuan et al. (2024) — Self-Rewarding Language Models

- **Cita**: Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston (Meta FAIR / NYU). *Self-Rewarding Language Models*. arXiv:2401.10020. ICML 2024 (PMLR vol. 235).
- **URL**: https://arxiv.org/abs/2401.10020
- **Resumen**: [[summaries/yuan-2024-self-rewarding]]
- **Páginas afectadas**: [[concepts/self-rewarding-loop]], [[concepts/llm-as-a-judge]], [[synthesis/pilar-1-irl-implicito]], [[synthesis/pilar-2-multi-agente-cooperativo]], [[synthesis/pilar-3-dpo-en-loop]]
- **Estado**: ingerido 2026-05-12. Paper ancla confirmado.

### Rafailov et al. (2023) — Direct Preference Optimization

- **Cita**: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (Stanford). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. arXiv:2305.18290. NeurIPS 2023.
- **URL**: https://arxiv.org/abs/2305.18290 · [PDF NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf)
- **Resumen**: [[summaries/rafailov-2023-dpo]]
- **Páginas afectadas**: [[concepts/dpo]], [[concepts/kl-beta]], [[synthesis/pilar-3-dpo-en-loop]], [[synthesis/pilar-1-irl-implicito]]
- **Estado**: ingerido 2026-05-12.

### Wu et al. (2024) — Meta-Rewarding Language Models

- **Cita**: Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, Sainbayar Sukhbaatar (Meta FAIR / UC Berkeley / NYU). *Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge*. arXiv:2407.19594.
- **URL**: https://arxiv.org/abs/2407.19594
- **Resumen**: [[summaries/wu-2024-meta-rewarding]]
- **Páginas afectadas**: [[concepts/self-rewarding-loop]], [[synthesis/pilar-2-multi-agente-cooperativo]], [[synthesis/diagnostico-fallos]]
- **Estado**: ingerido 2026-05-12.

---

## 2. RLHF y RL clásico (contexto)

### Christiano et al. (2017) — Deep RL from Human Preferences

- **Cita**: Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei (OpenAI / DeepMind). *Deep Reinforcement Learning from Human Preferences*. arXiv:1706.03741. NeurIPS 2017.
- **URL**: https://arxiv.org/abs/1706.03741
- **Resumen**: [[summaries/christiano-2017-deep-rl-preferences]]

### Ouyang et al. (2022) — InstructGPT

- **Cita**: Long Ouyang, Jeff Wu, Xu Jiang, et al. (OpenAI). *Training language models to follow instructions with human feedback*. arXiv:2203.02155. NeurIPS 2022.
- **URL**: https://arxiv.org/abs/2203.02155
- **Resumen**: [[summaries/ouyang-2022-instructgpt]]

### Schulman et al. (2017) — PPO

- **Cita**: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- **URL**: https://arxiv.org/abs/1707.06347
- **Resumen**: [[summaries/schulman-2017-ppo]]

---

## 3. Inverse Reinforcement Learning (fundamentos)

### Ng & Russell (2000) — Algorithms for IRL

- **Cita**: Andrew Y. Ng, Stuart Russell. *Algorithms for Inverse Reinforcement Learning*. ICML 2000.
- **URL**: https://ai.stanford.edu/~ang/papers/icml00-irl.pdf
- **Resumen**: [[summaries/ng-russell-2000-irl]]

### Abbeel & Ng (2004) — Apprenticeship Learning

- **Cita**: Pieter Abbeel, Andrew Y. Ng. *Apprenticeship Learning via Inverse Reinforcement Learning*. ICML 2004.
- **URL**: https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf
- **Resumen**: [[summaries/abbeel-ng-2004-apprenticeship]]

### Ziebart et al. (2008) — Maximum Entropy IRL

- **Cita**: Brian D. Ziebart, Andrew Maas, J. Andrew Bagnell, Anind K. Dey (CMU). *Maximum Entropy Inverse Reinforcement Learning*. AAAI 2008.
- **URL**: https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf
- **Resumen**: [[summaries/ziebart-2008-maxent-irl]]
- **Importancia**: puente teórico DPO ↔ IRL.

### Arora & Doshi (2021) — IRL Survey

- **Cita**: Saurabh Arora, Prashant Doshi. *A Survey of Inverse Reinforcement Learning: Challenges, Methods and Progress*. **Artificial Intelligence Journal** (Elsevier), vol. 297, art. 103500.
- **URL**: https://arxiv.org/abs/1806.06877 · DOI 10.1016/j.artint.2021.103500
- **Peer-reviewed**: sí (journal).

### Wirth et al. (2017) — Preference-Based RL Survey

- **Cita**: Christian Wirth, Riad Akrour, Gerhard Neumann, Johannes Fürnkranz. *A Survey of Preference-Based Reinforcement Learning Methods*. **Journal of Machine Learning Research (JMLR)** vol. 18, pp. 1–46.
- **URL**: https://jmlr.org/papers/v18/16-634.html
- **Resumen**: [[summaries/wirth-2017-pbrl-survey]]
- **Peer-reviewed**: sí (JMLR).
- **Importancia**: puente formal IRL ↔ RLHF pre-LLM.

### Sadigh, Dragan, Sastry, Seshia (2017) — Active Preference Learning

- **Cita**: Dorsa Sadigh, Anca D. Dragan, Shankar Sastry, Sanjit A. Seshia. *Active Preference-Based Learning of Reward Functions*. Robotics: Science and Systems (RSS) 2017.
- **URL**: https://roboticsproceedings.org/rss13/p53.html · DOI 10.15607/RSS.2017.XIII.053
- **Peer-reviewed**: sí.

### Hejna & Sadigh (2023) — Inverse Preference Learning

- **Cita**: Joey Hejna, Dorsa Sadigh. *Inverse Preference Learning: Preference-based RL without a Reward Function*. arXiv:2305.15363. NeurIPS 2023.
- **URL**: https://arxiv.org/abs/2305.15363 · [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3be7859b36d9440372cae0a293f2e4cc-Abstract-Conference.html)

---

## 4. Variantes post-DPO (análisis teórico y alternativas)

### Azar et al. (2023) — IPO

- **Cita**: Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, Rémi Munos (DeepMind). *A General Theoretical Paradigm to Understand Learning from Human Preferences*. arXiv:2310.12036. AISTATS 2024.
- **URL**: https://arxiv.org/abs/2310.12036
- **Resumen**: [[summaries/azar-2023-ipo]]
- **Importancia**: explica formalmente por qué DPO puede colapsar — relevante para diagnosticar la saturación de Self-Rewarding tras varias iteraciones.

### Ethayarajh et al. (2024) — KTO

- **Cita**: Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela. *KTO: Model Alignment as Prospect Theoretic Optimization*. arXiv:2402.01306. ICML 2024.
- **URL**: https://arxiv.org/abs/2402.01306

### Hong, Lee, Thorne (2024) — ORPO

- **Cita**: Jiwoo Hong, Noah Lee, James Thorne. *ORPO: Monolithic Preference Optimization without Reference Model*. arXiv:2403.07691. EMNLP 2024.
- **URL**: https://arxiv.org/abs/2403.07691 · [ACL Anthology](https://aclanthology.org/2024.emnlp-main.626/)
- **Peer-reviewed**: sí (EMNLP).

---

## 5. Métodos adyacentes (RLAIF, self-play, self-training)

### Bai et al. (2022) — Constitutional AI / RLAIF

- **Cita**: Yuntao Bai, Saurav Kadavath, et al. (Anthropic). *Constitutional AI: Harmlessness from AI Feedback*. arXiv:2212.08073.
- **URL**: https://arxiv.org/abs/2212.08073

### Lee et al. (2023) — RLAIF sistemático

- **Cita**: Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, et al. (Google). *RLAIF vs. RLHF: Scaling RL from Human Feedback with AI Feedback*. arXiv:2309.00267. ICML 2024.
- **URL**: https://arxiv.org/abs/2309.00267

### Chen et al. (2024) — SPIN

- **Cita**: Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu (UCLA). *Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models*. arXiv:2401.01335. ICML 2024.
- **URL**: https://arxiv.org/abs/2401.01335

### Wu et al. (2024) — SPPO

- **Cita**: Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, Quanquan Gu (UCLA / CMU). *Self-Play Preference Optimization for Language Model Alignment*. arXiv:2405.00675.
- **URL**: https://arxiv.org/abs/2405.00675
- **Importancia**: contiene la comparación head-to-head más cercana SPPO vs Self-Rewarding vs SPIN vs DPO/IPO iterativos bajo setup uniforme (Mistral-7B-Instruct-v0.2, UltraFeedback 60k prompts).

### Zelikman et al. (2022) — STaR

- **Cita**: Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman. *STaR: Bootstrapping Reasoning With Reasoning*. arXiv:2203.14465. NeurIPS 2022.
- **URL**: https://arxiv.org/abs/2203.14465
- **Peer-reviewed**: sí (NeurIPS).

### Huang et al. (2023) — Self-Improve

- **Cita**: Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, Jiawei Han. *Large Language Models Can Self-Improve*. arXiv:2210.11610. EMNLP 2023 (DOI 10.18653/v1/2023.emnlp-main.67).
- **URL**: https://arxiv.org/abs/2210.11610

### Madaan et al. (2023) — Self-Refine

- **Cita**: Aman Madaan, Niket Tandon, Prakhar Gupta, et al. *Self-Refine: Iterative Refinement with Self-Feedback*. arXiv:2303.17651. NeurIPS 2023.
- **URL**: https://arxiv.org/abs/2303.17651

### Saunders et al. (2022) — Self-critique

- **Cita**: William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, Jan Leike (OpenAI). *Self-critiquing models for assisting human evaluators*. arXiv:2206.05802.
- **URL**: https://arxiv.org/abs/2206.05802

---

## 6. Cooperative multi-agent y debate

### Du et al. (2024) — Multi-Agent Debate

- **Cita**: Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch. *Improving Factuality and Reasoning in Language Models through Multiagent Debate*. arXiv:2305.14325. ICML 2024.
- **URL**: https://arxiv.org/abs/2305.14325

### Liang et al. (2024) — MAD

- **Cita**: Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Zhaopeng Tu, Shuming Shi. *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate*. arXiv:2305.19118. EMNLP 2024.
- **URL**: https://arxiv.org/abs/2305.19118 · [ACL Anthology](https://aclanthology.org/2024.emnlp-main.992/)
- **Aporte**: introduce el concepto **Degeneración del Pensamiento (DoT)**, mecanismo explicativo de la saturación de self-reflection.

### Kim et al. (2024) — Prospector

- **Cita**: Byoungjip Kim, et al. *Prospector: Improving LLM Agents with Self-Asking and Trajectory Ranking*. EMNLP 2024 Findings.
- **URL**: https://aclanthology.org/2024.findings-emnlp.879/
- **Aporte**: arquitectura Actor-Critic con dos LLMs explícita — respaldo formal del framing cooperative multi-agent.

### Goodfellow et al. (2014) — GANs

- **Cita**: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, et al. *Generative Adversarial Networks*. arXiv:1406.2661. NeurIPS 2014.
- **URL**: https://arxiv.org/abs/1406.2661
- **Uso**: solo como contraste estructural (adversarial ≠ cooperativo).

---

## 7. LLM-as-a-Judge (paradigma y sesgos)

### Zheng et al. (2023) — Judging LLM-as-a-Judge

- **Cita**: Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. arXiv:2306.05685. NeurIPS 2023 Datasets & Benchmarks.
- **URL**: https://arxiv.org/abs/2306.05685
- **Resumen**: [[summaries/zheng-2023-llm-as-judge]]

### Li et al. (2025) — LLM-as-a-Judge Survey

- **Cita**: Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, Kai Shu, Lu Cheng, Huan Liu. *From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge*. arXiv:2411.16594. EMNLP 2025.
- **URL**: https://aclanthology.org/2025.emnlp-main.138/

### Self-Preference Bias paper (2024)

- **Cita**: *Self-Preference Bias in LLM-as-a-Judge*. arXiv:2410.21819.
- **URL**: https://arxiv.org/abs/2410.21819
- **Aporte**: evidencia cuantitativa de que GPT-4 prefiere sus propios outputs; vincula el sesgo a baja perplexity sobre texto familiar.

---

## 8. Críticas y limitaciones de Self-Rewarding (post-Yuan 2024)

### Wang et al. (2025) — Temporal Self-Rewarding

- **Cita**: Yidong Wang, Xin Wang, Cunxiang Wang, Junfeng Fang, Qiufeng Wang, Jianing Chu, Xuran Meng, Shuxun Yang, Libo Qin, Yue Zhang, Wei Ye, Shikun Zhang. *Temporal Self-Rewarding Language Models: Decoupling Chosen-Rejected via Past-Future*. arXiv:2508.06026.
- **URL**: https://arxiv.org/abs/2508.06026
- **Resumen**: [[summaries/wang-2025-temporal-sr]]
- **Importancia**: diagnóstico cuantitativo del modo de fallo central. **Crítica imprescindible para la presentación.**

### Zhou et al. (2025) — SCIR

- **Cita**: *Self-Consistency of the Internal Reward Models Improves Self-Rewarding Language Models*. arXiv:2502.08922.
- **URL**: https://arxiv.org/abs/2502.08922
- **Aporte**: muestra que los dos rewards internos de Self-Rewarding (juez generativo + DPO implícito) son inconsistentes entre sí.

### Zhang et al. (2025) — Process-SRLM

- **Cita**: *Process-based Self-Rewarding Language Models*. arXiv:2503.03746. Findings of ACL 2025.
- **URL**: https://arxiv.org/abs/2503.03746
- **Aporte**: documenta que Self-Rewarding tradicional **falla en razonamiento matemático**; propone extensión step-wise.

### Shafayat et al. (2025) — Can LRMs Self-Train?

- **Cita**: Sheikh Shafayat, Joey Hejna, Adam Foster, Heyang Sun, Dorsa Sadigh. *Can Large Reasoning Models Self-Train?*. arXiv:2505.21444.
- **URL**: https://arxiv.org/abs/2505.21444
- **Aporte**: muestra **colapso súbito de performance** por reward hacking en self-training sostenido.

---

## 9. Recursos pedagógicos (no para citar en slides, sí para el wiki)

| Recurso | Tipo | URL |
|---|---|---|
| Lambert, N. (2024-2026). *RLHF Book* | Libro open-access | https://rlhfbook.com/ |
| Stanford CS224N Lecture 10 (Spring 2024) — Prompting, Instruction Finetuning, DPO/RLHF | Slides curso | https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture10-prompting-rlhf.pdf |
| Stanford CS224R (2026) Lecture 9 — Post-Training Frontier: RLHF, DPO, Modern Preference Opt. | Slides curso | https://cs224r.stanford.edu/slides/09_cs224r_rlhf_2026.pdf |
| Stanford CS324 (Winter 2022) — Large Language Models | Notas curso | https://stanford-cs324.github.io/winter2022/lectures/ |

---

## Priorización

**Top-7 confirmado para la presentación** (orden de centralidad):

1. **Yuan et al. 2024** — paper ancla.
2. **Rafailov et al. 2023 (DPO)** — pilar técnico de optimización.
3. **Wu et al. 2024 (Meta-Rewarding)** — extensión y refuerzo del framing multi-agente.
4. **Wang et al. 2025 (Temporal Self-Rewarding)** — crítica empírica obligatoria.
5. **Azar et al. 2023 (IPO)** — diagnóstico teórico del colapso DPO.
6. **Ziebart et al. 2008 (MaxEnt IRL)** — puente formal IRL ↔ DPO.
7. **Zheng et al. 2023 (LLM-as-a-Judge)** — fundamento + sesgos del paradigma.

**Tier 2** (citables como respaldo, sin slide propia): Ouyang 2022, Christiano 2017, Schulman 2017 (PPO), Bai 2022 (Constitutional AI), Lee 2023 (RLAIF), Chen 2024 (SPIN), Wu 2024 (SPPO), Wirth 2017 (JMLR survey), Hejna-Sadigh 2023 (IPL), Ng-Russell 2000, Abbeel-Ng 2004, Madaan 2023 (Self-Refine), Zelikman 2022 (STaR), Huang 2023 (Self-Improve), Saunders 2022, Du 2024 (debate), Liang 2024 (MAD/DoT), Kim 2024 (Prospector), Zhou 2025 (SCIR), Zhang 2025 (Process-SRLM), Shafayat 2025, Li 2025 (judge survey), Ethayarajh 2024 (KTO), Hong 2024 (ORPO), Goodfellow 2014 (GAN — solo contraste).

## Hiperparámetros verificados

Datos confirmados en los apéndices de los papers originales (importantes para slides técnicas):

- **Yuan 2024 (Self-Rewarding)**: DPO con β = 0.1, LR 1e-6 → 1e-7 con decay; SFT LR 5.5e-6 → 1.1e-6 cosine decay; batch size 16; dropout 0.1; early stopping cada 200 steps; validación con Claude 2 sobre 253 ejemplos.
- **Wu 2024 (Meta-Rewarding)**: DPO 10 épocas, LR 5e-6, β = 0.1; SFT 10 épocas LR 5e-8, batch 32, cosine schedule, checkpoint época 5; cuatro iteraciones; 20.000 prompts generados por Llama-2-70B-Chat con 8-shot.
- **SPPO**: comparación head-to-head sobre Mistral-7B-Instruct-v0.2, 60k prompts UltraFeedback.
