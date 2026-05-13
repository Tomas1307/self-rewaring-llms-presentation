# Self-Rewarding Models (2024–2025)
---

## 1. Self-Rewarding Language Models *(paper seminal)*

**Enlace:** https://arxiv.org/abs/2401.10020  
**Autores:** Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston (Meta / NYU). ICML 2024.

**Abstract:**
> "We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human preferences, which may then be bottlenecked by human performance level, and secondly these separate frozen reward models cannot then learn to improve during LLM training. In this work, we study Self-Rewarding Language Models, where the language model itself is used via LLM-as-a-Judge prompting to provide its own rewards during training. We show that during Iterative DPO training that not only does instruction following ability improve, but also the ability to provide high-quality rewards to itself. Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613."

**Resumen propio:**
Este es el paper seminal del paradigma. Un único LLM asume simultáneamente el rol de generador de respuestas y de juez (LLM-as-a-Judge) que asigna sus propias recompensas durante el entrenamiento iterativo con DPO. En cada iteración el modelo genera pares de preferencias evaluando sus propias respuestas y se actualiza, mejorando tanto la calidad de sus respuestas como su capacidad de juzgar. Tras tres iteraciones con Llama 2 70B supera a Claude 2, Gemini Pro y GPT-4 0613 en AlpacaEval 2.0, abriendo la puerta a modelos que se auto-mejoran indefinidamente sin un reward model externo congelado.

---

## 2. Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (SPIN)

**Enlace:** https://arxiv.org/abs/2401.01335  
**Autores:** Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu (UCLA). ICML 2024.

**Abstract:**
> "We propose a new fine-tuning method called Self-Play fIne-tuNing (SPIN), which starts from a supervised fine-tuned model. At the heart of SPIN lies a self-play mechanism, where the LLM refines its capability by playing against instances of itself. More specifically, the LLM generates its own training data from its previous iterations, refining its policy by discerning these self-generated responses from those obtained from human-annotated data. Our method progressively elevates the LLM from a nascent model to a formidable one, unlocking the full potential of human-annotated demonstration data for SFT. Theoretically, we prove that the global optimum to the training objective function of our method is achieved only when the LLM policy aligns with the target data distribution. Empirically, we evaluate our method on several benchmark datasets including the HuggingFace Open LLM Leaderboard, MT-Bench, and datasets from Big-Bench."

**Resumen propio:**
SPIN introduce un mecanismo de self-play donde el LLM actual (el "jugador principal") aprende a distinguir respuestas humanas auténticas de respuestas generadas por su propia iteración anterior (el "oponente"). Este enfoque extrae el máximo valor de los datos de SFT existentes sin necesidad de nuevas anotaciones humanas ni modelos más fuertes. Los autores demuestran teóricamente que el óptimo global se alcanza sólo cuando la política coincide con la distribución de los datos humanos, y empíricamente SPIN supera al DPO suplementado con preferencias de GPT-4 en múltiples benchmarks.

---

## 3. Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge

**Enlace:** https://arxiv.org/abs/2407.19594  
**Autores:** Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, Sainbayar Sukhbaatar (Meta FAIR / UC Berkeley / NYU). 2024.

**Abstract:**
> "Large Language Models (LLMs) are rapidly surpassing human knowledge in many domains. While improving these models traditionally relies on costly human data, recent self-rewarding mechanisms have shown that LLMs can improve by judging their own responses instead of relying on human labelers. However, existing methods have primarily focused on improving model responses rather than judgment capabilities, resulting in rapid saturation during iterative training. To address this issue, we introduce a novel Meta-Rewarding step to the self-improvement process, where the model judges its own judgements and uses that feedback to refine its judgment skills. Surprisingly, this unsupervised approach improves the model's ability to judge and follow instructions, as demonstrated by a win rate improvement of Llama-3-8B-Instruct from 22.9% to 39.4% on AlpacaEval 2, and 20.6% to 29.1% on Arena-Hard."

**Resumen propio:**
Meta-Rewarding extiende el self-rewarding clásico añadiendo un tercer rol: el de "meta-juez" que evalúa la calidad de los propios juicios del modelo. Mientras el self-rewarding sólo refuerza la generación de respuestas, este paso adicional entrena explícitamente la capacidad evaluadora, rompiendo la saturación observada en iteraciones sucesivas. El resultado es una mejora notable sin supervisión humana (22.9% → 39.4% en AlpacaEval 2 con Llama-3-8B-Instruct), demostrando que pulir las habilidades de evaluación es tan importante como mejorar las respuestas.

---

## 4. CREAM: Consistency Regularized Self-Rewarding Language Models

**Enlace:** https://arxiv.org/abs/2410.12735  
**Autores:** Zhaoyang Wang, Weilei He, Zhiyuan Liang, Xuchao Zhang, Chetan Bansal, Ying Wei, Weitong Zhang, Huaxiu Yao. ICLR 2025.

**Abstract:**
> "Recent self-rewarding large language models (LLM) have successfully applied LLM-as-a-Judge to iteratively improve the alignment performance without the need of human annotations for preference data. These methods commonly utilize the same LLM to act as both the policy model (which generates responses) and the reward model (which scores and ranks those responses). The ranked responses are then used as preference pairs to train the LLM via direct alignment technologies (e.g. DPO). However, it is noteworthy that throughout this process, there is no guarantee of accuracy in the rewarding and ranking, which is critical for ensuring accurate rewards and high-quality preference data. Empirical results from relatively small LLMs (e.g., 7B parameters) also indicate that improvements from self-rewarding may diminish after several iterations in certain situations, which we hypothesize is due to accumulated bias in the reward system."

**Resumen propio:**
CREAM identifica un problema crítico en los modelos auto-recompensados pequeños (≈7B): el sesgo del sistema de recompensa se acumula, provocando etiquetas de preferencia sobre-confiadas y retroceso del rendimiento tras unas pocas iteraciones. Los autores introducen un término de regularización basado en la consistencia de los rankings entre iteraciones consecutivas: cuando la consistencia es baja, la señal de preferencia se atenúa. Esto produce datos de preferencia más confiables y entrenamiento más estable, con mejoras demostrables en consistencia de la recompensa y alineamiento general.

---

## 5. Process-based Self-Rewarding Language Models

**Enlace:** https://arxiv.org/abs/2503.03746  
**Autores:** Shimao Zhang, Xiao Liu, Xin Zhang, Junxiao Liu, Zheheng Luo, Shujian Huang, Yeyun Gong (Nanjing University / Microsoft Research Asia). Findings of ACL 2025.

**Abstract:**
> "Large Language Models have demonstrated outstanding performance across various downstream tasks and have been widely applied in multiple scenarios. Human-annotated preference data is used for training to further improve LLMs' performance, which is constrained by the upper limit of human performance. Therefore, Self-Rewarding method has been proposed, where LLMs generate training data by rewarding their own outputs. However, the existing self-rewarding paradigm is not effective in mathematical reasoning scenarios and may even lead to a decline in performance. In this work, we propose the Process-based Self-Rewarding pipeline for language models, which introduces long-thought reasoning, step-wise LLM-as-a-Judge, and step-wise preference optimization within the self-rewarding paradigm."

**Resumen propio:**
Este trabajo aborda una limitación crítica del self-rewarding clásico: su inefectividad en razonamiento matemático complejo, donde evaluar holísticamente soluciones multi-paso resulta insuficiente. El pipeline propuesto descompone tanto la evaluación como la optimización en pasos individuales, combinando razonamiento de cadena larga con un juez paso a paso. La contribución principal es trasladar el paradigma de auto-recompensa al dominio del razonamiento matemático de manera efectiva, logrando mejoras consistentes en GSM8K, MATH y otros benchmarks matemáticos tras iteraciones sucesivas.

---

## 6. Calibrated Self-Rewarding Vision Language Models (CSR)

**Enlace:** https://arxiv.org/abs/2405.14622  
**Autores:** Yiyang Zhou, Zhiyuan Fan, Dongjie Cheng, Sihan Yang, Zhaorun Chen, Chenhang Cui, Xiyao Wang, Yun Li, Linjun Zhang, Huaxiu Yao. NeurIPS 2024.

**Abstract:**
> "Large Vision-Language Models (LVLMs) have made substantial progress by integrating pre-trained large language models (LLMs) and vision models through instruction tuning. Despite these advancements, LVLMs often exhibit the hallucination phenomenon, where generated text responses appear linguistically plausible but contradict the input image, indicating a misalignment between image and text pairs. [...] In the reward modeling, we employ a step-wise strategy and incorporate visual constraints into the self-rewarding process to place greater emphasis on visual input. Empirical results demonstrate that CSR enhances performance and reduces hallucinations, achieving substantial improvements across ten benchmarks and tasks."

**Resumen propio:**
CSR extiende el paradigma de auto-recompensa a modelos de visión-lenguaje con el objetivo de reducir alucinaciones visuales. La innovación clave es incorporar restricciones visuales explícitas (puntuaciones de alineación imagen-texto combinadas con scores lingüísticos) durante la auto-evaluación paso a paso, evitando que el modelo ignore la entrada visual. Los autores demuestran teórica y empíricamente que esta calibración visual mejora hasta un 7.62% en diez benchmarks multimodales, manteniendo la capacidad de mejora iterativa sin supervisión externa.

---

## 7. Self-Rewarding Vision-Language Model via Reasoning Decomposition (Vision-SR1)

**Enlace:** https://arxiv.org/abs/2508.19652  
**Autores:** Investigadores de NTU/NUS (2025).

**Abstract:**
> "Vision-Language Models (VLMs) often suffer from visual hallucinations, saying things that are not actually in the image, and language shortcuts, where they skip the visual part and just rely on text priors. [...] In this paper, we introduce Vision-SR1, a self-rewarding method that improves visual reasoning without relying on external visual supervisions via reinforcement learning. Vision-SR1 decomposes VLM reasoning into two stages: visual perception and language reasoning. The model is first prompted to produce self-contained visual perceptions that are sufficient to answer the question without referring back the input image. To validate this self-containment, the same VLM model is then re-prompted to perform language reasoning using only the generated perception as input to compute reward."

**Resumen propio:**
Vision-SR1 descompone el razonamiento multimodal en percepción visual y razonamiento lingüístico, construyendo una señal de auto-recompensa novedosa: el modelo genera una descripción visual "auto-contenida" y luego, sin acceder a la imagen, intenta responder usando sólo esa descripción. Si lo logra, la percepción es fiel y recibe una recompensa interna. Combinada con GRPO y supervisión de la respuesta final, esta estrategia reduce alucinaciones visuales y atajos lingüísticos de manera autónoma, sin supervisión externa ni labels costosos.

---

## 8. Self-Taught Evaluators

**Enlace:** https://arxiv.org/abs/2408.02666  
**Autores:** Tianlu Wang et al. (Meta FAIR). 2024.

**Abstract:**
> "Model-based evaluation is at the heart of successful model development -- as a reward model for training, and as a replacement for human evaluation. To train such evaluators, the standard approach is to collect a large amount of human preference judgments over model responses, which is costly and the data becomes stale as models improve. In this work, we present an approach that aims to improve evaluators without human annotations, using synthetic training data only. Starting from unlabeled instructions, our iterative self-improvement scheme generates contrasting model outputs and trains an LLM-as-a-Judge to produce reasoning traces and final judgments, repeating this training at each new iteration using the improved predictions. Without any labeled preference data, our Self-Taught Evaluator can improve a strong LLM (Llama3-70B-Instruct) from 75.4 to 88.3 (88.7 with majority vote) on RewardBench."

**Resumen propio:**
Self-Taught Evaluators construye un evaluador LLM-as-a-Judge sin ninguna etiqueta humana: parte de instrucciones sin anotar, genera pares de respuestas contrastantes sintéticamente y entrena el juez de manera iterativa con sus propias predicciones mejoradas. Cada iteración usa las cadenas de razonamiento del juez actual para producir datos de entrenamiento para el siguiente, creando un ciclo de auto-mejora puro. El resultado mejora Llama3-70B-Instruct de 75.4 a 88.3 en RewardBench, superando a GPT-4 como evaluador sin ninguna anotación humana.

---

## 9. Self-Play Preference Optimization for Language Model Alignment (SPPO)

**Enlace:** https://arxiv.org/abs/2405.00675  
**Autores:** Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, Quanquan Gu (UCLA). NeurIPS 2024.

**Abstract:**
> "In this paper, we propose a self-play-based method for language model alignment, which treats the problem as a constant-sum two-player game aimed at identifying the Nash equilibrium policy. Our approach, dubbed Self-Play Preference Optimization (SPPO), utilizes iterative policy updates to provably approximate the Nash equilibrium. [...] Using only 60k prompts (without responses) from the UltraFeedback dataset and without any prompt augmentation, by leveraging a pre-trained preference model PairRM with only 0.4B parameters, SPPO can obtain a model from fine-tuning Mistral-7B-Instruct-v0.2 that achieves the state-of-the-art length-controlled win-rate of 28.53% against GPT-4-Turbo on AlpacaEval 2.0."

**Resumen propio:**
SPPO formaliza el alineamiento como un juego de suma constante de dos jugadores y busca la política de equilibrio de Nash, que es preferida sobre cualquier otra política de manera consistente. El algoritmo actualiza iterativamente la política usando únicamente prompts sin respuestas y un modelo de preferencias pequeño (PairRM, 0.4B), sin supervisión de GPT-4. Logra un win-rate LC de 28.53% frente a GPT-4-Turbo en AlpacaEval 2.0 con Mistral-7B y 38.77% con Llama-3-8B, superando a DPO e IPO con garantías teóricas de convergencia.

---

## 10. Bootstrapping Language Models with DPO Implicit Rewards (DICE)

**Enlace:** https://arxiv.org/abs/2406.09760  
**Autores:** Changyu Chen, Zichen Liu, Chao Du, et al. (Sea AI Lab). ICLR 2025.

**Abstract:**
> "Human alignment in large language models (LLMs) is an active area of research. A recent groundbreaking work, direct preference optimization (DPO), has greatly simplified the process from past work in reinforcement learning from human feedback (RLHF) by bypassing the reward learning stage in RLHF. DPO, after training, provides an implicit reward model. In this work, we make a novel observation that this implicit reward model can by itself be used in a bootstrapping fashion to further align the LLM. Our approach is to use the rewards from a current LLM to construct a preference dataset, which is then used in subsequent DPO rounds. [...] Our approach, named self-alignment with DPO ImpliCit rEwards (DICE), shows great improvements in alignment. It achieves an increase of more than 8% in length-controlled win rate on AlpacaEval 2 for all the different base models that we tried, without relying on external feedback."

**Resumen propio:**
DICE observa que un modelo entrenado con DPO contiene un reward model implícito (la diferencia log π_θ − log π_ref) y propone reutilizar esas recompensas para construir nuevos pares de preferencias de manera iterativa, sin LLM-as-a-Judge externo. Para mitigar sesgos conocidos incorpora regularización de longitud y experience replay. El resultado es un esquema elegante de self-rewarding basado en la propia aritmética de DPO, obteniendo >8% de mejora en AlpacaEval 2 length-controlled sin feedback externo alguno.

---

## 11. Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences (DNO)

**Enlace:** https://arxiv.org/abs/2404.03715  
**Autores:** Corby Rosset, Ching-An Cheng, Arindam Mitra, Michael Santacroce, Ahmed Awadallah, Tengyang Xie (Microsoft Research). 2024.

**Abstract:**
> "This paper studies post-training large language models (LLMs) using preference feedback from a powerful oracle to help a model iteratively improve over itself. [...] We introduce Direct Nash Optimization (DNO), a provable and scalable algorithm that marries the simplicity and stability of contrastive learning with theoretical generality from optimizing general preferences. Because DNO is a batched on-policy algorithm using a regression-based objective, its implementation is straightforward and efficient. Moreover, DNO enjoys monotonic improvement across iterations which helps it improve even over a strong teacher (such as GPT-4). In our experiments, a resulting 7B parameter Orca-2.5 model aligned by DNO achieves the state-of-the-art win-rate against GPT-4-Turbo of 33% on AlpacaEval 2.0."

**Resumen propio:**
DNO formaliza la auto-mejora iterativa de LLMs como la búsqueda de un equilibrio de Nash sobre preferencias generales, evitando las limitaciones del modelo Bradley-Terry (que no puede capturar preferencias intransitivas o cíclicas). El algoritmo es batched on-policy y se reduce a un objetivo de regresión simple, garantizando mejora monotónica entre iteraciones. Un Orca-2.5 de 7B alcanza 33% de win-rate contra GPT-4-Turbo en AlpacaEval 2.0, un avance absoluto de 26 puntos desde el modelo inicial.

---

## 12. Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding (LaTRO)

**Enlace:** https://arxiv.org/abs/2411.04282  
**Autores:** Investigadores de Salesforce AI Research. 2024.

**Abstract:**
> "Large language models (LLMs) have shown impressive capabilities, but solving complex problems often requires multi-step reasoning. [...] Our findings suggest that pre-trained LLMs are not only capable reasoners but also possess the potential to act as explicit reward models for evaluating reasoning paths. We term this approach of utilizing explicit reward functions induced by LLMs themselves as 'self-rewarding.' We propose LaTRO (Latent Reasoning Optimization), which reformulates reasoning as sampling from a latent distribution and optimizes it jointly with a self-rewarding objective derived from the same LLM. Empirically, LaTRO outperforms both baseline models and supervised fine-tuning approaches on reasoning tasks like GSM8K, while also demonstrating the capacity to compress reasoning processes and shift computational burdens from inference to training time."

**Resumen propio:**
LaTRO conceptualiza las trayectorias de razonamiento como variables latentes y optimiza simultáneamente la política y la función de recompensa, ambas derivadas del propio LLM. La intuición central es que los modelos pre-entrenados poseen "razonadores ocultos" capaces de juzgar la calidad de cadenas de pensamiento sin verificadores externos. La metodología combina muestreo variacional de trayectorias con auto-evaluación, permitiendo comprimir el coste de inferencia y aportando un fundamento variacional riguroso para el self-rewarding en razonamiento.

---

## 13. Self-Consistency Preference Optimization (ScPO)

**Enlace:** https://arxiv.org/abs/2411.04109  
**Autores:** Investigadores de múltiples instituciones. 2024.

**Abstract:**
> "We introduce Self-consistency Preference Optimization (ScPO). ScPO is an approach to self-train LLMs for complex problem-solving tasks without access to gold solutions or final answers in the training data. Our approach leverages the concept of self-consistency, an inference technique that consolidates multiple sampled reasoning chains through majority voting. By leveraging self-consistency, we construct preference pairs without any external reward model: consistent answers are treated as positive signals and inconsistent ones as negative signals. ScPO addresses a core limitation of self-rewarding approaches: that LLMs struggle at evaluating the correctness of their own responses on complex problem-solving tasks which have an unambiguous correct answer."

**Resumen propio:**
ScPO propone una solución elegante al problema de auto-evaluación en tareas con respuesta única correcta, donde el LLM no puede evaluar fiablemente sus propias respuestas mediante prompting de juez. En su lugar, usa la consistencia entre múltiples cadenas de razonamiento como señal de auto-recompensa: las respuestas que emergen por mayoría se consideran positivas y las discordantes negativas. Este mecanismo construye pares de preferencias sin reward model externo ni etiquetas, siendo especialmente efectivo en tareas de razonamiento matemático y lógico.

---

## 14. Self-Evolved Reward Learning for LLMs (SER)

**Enlace:** https://arxiv.org/abs/2411.00418  
**Autores:** Chenghua Huang et al. 2024.

**Abstract:**
> "Reinforcement Learning from Human Feedback (RLHF) is a crucial technique for aligning language models with human preferences, playing a pivotal role in the success of conversational models like GPT-4, ChatGPT, and Llama 2. A core challenge in employing RLHF lies in training a reliable reward model (RM), which relies on high-quality labels typically provided by human experts or advanced AI system. [...] In this paper, we propose Self-Evolved Reward Learning (SER), a novel approach where the RM generates additional training data to iteratively improve itself. We conducted extensive experiments on multiple datasets such as HH-RLHF and UltraFeedback, using models like Mistral and Llama 3, and compare SER against various baselines. Our results demonstrate that even with limited human-annotated data, learning from self-feedback can robustly enhance RM performance."

**Resumen propio:**
SER aborda el cuello de botella de los reward models supervisados proponiendo un ciclo iterativo en el que el propio RM genera datos de entrenamiento adicionales para mejorarse a sí mismo. El RM etiqueta muestras sobre las que tiene alta confianza y las usa para refinarse, con garantías teóricas de convergencia hacia una política cuasi-óptima incluso con errores de estimación de recompensa. Los experimentos con Llama 3 y Mistral en HH-RLHF y UltraFeedback muestran que el self-feedback robusto mejora al RM incluso con datos iniciales limitados.

---

## 15. Self-Consistency of the Internal Reward Models Improves Self-Rewarding Language Models (SCIR)

**Enlace:** https://arxiv.org/abs/2502.08922  
**Autores:** Xin Zhou, Yiwen Guo, Ruotian Ma, Tao Gui, Qi Zhang, Xuanjing Huang. 2025.

**Abstract:**
> "Recent research has shown that the self-rewarding language model (SRLM) is a promising approach to address these challenges. The core idea behind SRLM is to use LLMs themselves to generate preference data, reducing the need for human annotation or external reward models. [...] [We propose] SCIR [...] Through comprehensive empirical evaluation, we demonstrate that SCIR can improve both alignment performance and reward modeling ability compared to the baselines, validating that self-consistency of internal reward models can improve SRLM."

**Resumen propio:**
SCIR identifica que un LLM contiene múltiples señales internas de recompensa (prompting de juez, recompensa implícita DPO, likelihood de secuencias) y propone utilizar la consistencia entre ellas como criterio de fiabilidad para filtrar pares de preferencias. Los pares donde las distintas señales internas coinciden se ponderan más, mientras que los inconsistentes se descartan o atenúan. Esta perspectiva de "ensemble interno" mejora la robustez del self-rewarding sin modelos externos, mitigando el problema de sesgo acumulado que afecta a métodos como CREAM.

---

## 16. Self-rewarding correction for mathematical reasoning

**Enlace:** https://arxiv.org/abs/2502.19613  
**Autores:** Wei Xiong, Hanning Zhang, Chenlu Ye, Lichang Chen, Nan Jiang, Tong Zhang (UIUC / Maryland). 2025.

**Abstract:**
> "We study self-rewarding reasoning large language models (LLMs), which can simultaneously generate step-by-step reasoning and evaluate the correctness of their outputs during the inference time—without external feedback. This integrated approach allows a single model to independently guide its reasoning process, offering computational advantages for model deployment. We particularly focus on the representative task of self-correction, where models autonomously detect errors in their responses, revise outputs, and decide when to terminate iterative refinement loops. To enable this, we propose a two-staged algorithmic framework for constructing self-rewarding reasoning models using only self-generated data."

**Resumen propio:**
Este trabajo propone un framework de dos etapas para entrenar LLMs que simultáneamente razonan paso a paso y evalúan la corrección de sus propias salidas durante la inferencia. La primera fase sintetiza trayectorias de cadena larga con patrones de auto-recompensa y auto-corrección mediante rejection sampling secuencial; la segunda fase refuerza estas habilidades con RL basado en reglas. Sin reward models externos, Llama-3 y Qwen-2.5 entrenados con este método igualan el desempeño de sistemas con verificadores externos en razonamiento matemático.

---

## 17. Process-based Reward Model Meets Process-based Self-Rewarding / Self-Aligned Reward (SAR)

**Enlace:** https://arxiv.org/abs/2509.05489  
**Autores:** Investigadores de instituciones académicas. 2025.

**Abstract:**
> "Recently, reinforcement learning (RL) with verifiable rewards has attracted broad attention in LLM training, demonstrating remarkable improvements in reasoning skills. However, such verifiable signals—especially in domains like math—are inherently coarse, providing only a binary correct/incorrect signal that can lead to overlong or shallow reasoning. We propose Self-Aligned Reward, a complementary reward signal that the model generates for itself and combines with the verifiable reward during RL post-training. Empirically, training with self-aligned reward enhances both efficiency (significant drops in average response length) and accuracy on math reasoning benchmarks."

**Resumen propio:**
SAR hibrida las recompensas verificables (binarias, basadas en corrección de la respuesta final) con una señal de auto-recompensa complementaria que el propio modelo genera durante el post-training con RL. Las recompensas verificables son gruesas y propensas a generar respuestas innecesariamente largas; la señal auto-alineada aporta granularidad de proceso. La combinación produce reasoners más eficientes (respuestas más cortas) y más precisos en benchmarks matemáticos como MATH-500 y AMC/AIME.

---

## 18. Temporal Self-Rewarding Language Models: Decoupling Chosen-Rejected via Past-Future

**Enlace:** https://arxiv.org/abs/2508.06026  
**Autores:** Investigadores de diversas instituciones. 2025.

**Abstract:**
> "Self-Rewarding language models demonstrate an alternative paradigm to self-improvement, where language models serve dual roles as both response generators and evaluators. [...] Despite the success of Self-Rewarding language models on benchmarks like AlpacaEval and Arena-Hard, our theoretical analysis reveals a critical limitation: when the representational similarity between chosen and rejected responses increases, the DPO gradient vanishes, causing training collapse. To address this, we introduce a Temporal Self-Rewarding framework that decouples chosen and rejected responses along a temporal axis: rejected responses are 'Anchored Rejection' generated from a past (initial SFT) version of the model, while chosen responses are 'Future-Guided Chosen' obtained from an EMA-tracked future version."

**Resumen propio:**
Este trabajo diagnostica matemáticamente por qué el self-rewarding estándar se degrada: cuando chosen y rejected provienen del mismo modelo en el mismo instante, sus representaciones convergen y el gradiente de DPO se desvanece. La solución introduce un eje temporal: los rejected se anclan a una versión pasada del modelo (SFT inicial) mientras los chosen se derivan de una versión futura trazada con EMA. Esta separación temporal preserva un margen de preferencia útil a lo largo de múltiples iteraciones, logrando ganancias sostenidas en benchmarks de instruction-following y razonamiento.

---

## 19. Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning (CoVo)

**Enlace:** https://arxiv.org/abs/2506.08745  
**Autores:** Kongcheng Zhang, Qi Yao, Shunyu Liu, Yingjie Wang, Baisheng Lai, Jieping Ye, Mingli Song, Dacheng Tao (Zhejiang University / Alibaba / NTU). 2025.

**Abstract:**
> "Recent advances of Reinforcement Learning (RL) have highlighted its potential in complex reasoning tasks, yet effective training often relies on external supervision, which limits the broader applicability. In this work, we propose a novel self-rewarding reinforcement learning framework to enhance Large Language Model (LLM) reasoning by leveraging the consistency of intermediate reasoning states across different reasoning trajectories. Our key insight is that correct responses often exhibit consistent trajectory patterns in terms of model likelihood: their intermediate reasoning states tend to converge toward their own final answers (high consistency) with minimal deviation toward other candidates (low volatility)."

**Resumen propio:**
CoVo deriva recompensas intrínsecas a partir de patrones geométricos en las trayectorias de razonamiento: las respuestas correctas muestran estados intermedios consistentes (convergentes hacia el answer final) con baja volatilidad, mientras las incorrectas oscilan. Estos dos features se combinan vectorialmente para producir una señal de recompensa robusta a outliers, complementada con un bono de curiosidad basado en divergencia KL para fomentar exploración diversa. El método permite a Qwen2.5 y Llama3.2 (3B-7B) alcanzar el rendimiento de GRPO con verificadores externos sin reward models ni etiquetas externas.

---

## 20. Can Large Reasoning Models Self-Train?

**Enlace:** https://arxiv.org/abs/2505.21444  
**Autores:** Investigadores de múltiples instituciones. 2025.

**Abstract:**
> "Online reinforcement learning with verifiable reward (RLVR) has emerged as a new paradigm of LLM post-training especially for enhancing math, coding and reasoning performances. Despite the success of the reasoning models, it is still unclear to what extent they can generalize beyond the difficulty of their training data distribution, a problem termed easy-to-hard generalization. In this work we systematically study whether large reasoning models can improve through pure self-training without any ground-truth labels, using only internally derived signals such as majority voting, semantic clustering and self-judging. We document both the gains in easy-to-hard generalization and the pathologies that arise, including reward hacking, diversity collapse and degradation after several iterations."

**Resumen propio:**
Este estudio empírico sistemático plantea la pregunta central: ¿hasta dónde pueden los grandes reasoning models auto-entrenarse con señales internas? Los autores evalúan varias variantes de auto-recompensa (mayoría, clustering semántico, self-judging) y documentan tanto ganancias en generalización fácil-a-difícil como las patologías: colapso de diversidad, reward hacking y degradación tras varias rondas. Sus resultados aportan una visión matizada y crítica que complementa los trabajos optimistas sobre self-rewarding, identificando las condiciones bajo las cuales el auto-entrenamiento funciona o no en modelos de razonamiento a gran escala.

---

## 21. RLSR: Reinforcement Learning from Self Reward

**Enlace:** https://arxiv.org/abs/2505.08827  
**Autores:** Investigadores de 2025.

**Abstract:**
> "Our approach reveals that models can generate synthetic tasks and reliably evaluate their own performance in certain domains, pointing toward a new paradigm of autonomous self-improvement. [...] Models can generate synthetic tasks and reliably evaluate their own performance in certain domains, pointing toward a new paradigm of autonomous self-improvement. This has profound implications for AI development, potentially enabling systems that continuously identify weaknesses, generate appropriate practice, and improve through self-directed learning cycles. Such autonomy could dramatically accelerate progress in domains where traditional supervised learning is limited by data availability or annotation costs."

**Resumen propio:**
RLSR propone un ciclo completo de auto-mejora autónoma donde el modelo no sólo evalúa sus respuestas (como en el self-rewarding estándar) sino que también genera las tareas de práctica más adecuadas para sus propias debilidades. El ciclo identifica dominios deficientes, sintetiza tareas dirigidas a ellos, se auto-evalúa y actualiza su política, todo sin supervisión humana. La propuesta apunta a un paradigma de agentes que se auto-dirigen curricularmente, similar al concepto de práctica deliberada aplicado a LLMs.

---

## 22. Self-Play Fine-Tuning (SPIN) + iteraciones con Self-Rewarding: Multi-Agent Evolve — LLM Self-Improve through Co-evolution

**Enlace:** https://arxiv.org/abs/2510.23595  
**Autores:** Investigadores de diversas instituciones. 2025.

**Abstract:**
> "We design domain-agnostic self-rewarding mechanisms, including Judge-based evaluation, difficulty-aware rewards, and format rewards, which eliminate the reliance on human-labeled ground truth or external verifiers. We empirically demonstrate the effectiveness and scalability of Multi-Agent Evolve on Qwen2.5-3B-Instruct across mathematics, coding, reasoning, and general knowledge benchmarks, achieving improvements over both base and supervised fine-tuning baselines."

**Resumen propio:**
Multi-Agent Evolve propone un framework multi-agente donde diferentes instancias del LLM co-evolucionan: un agente genera respuestas, otro actúa como juez con auto-recompensa, y el sistema retroalimenta dinámicamente la dificultad de los prompts. Se introducen tres tipos de recompensas auto-generadas: basada en juez, consciente de dificultad y de formato. El sistema es domain-agnostic, funciona en matemáticas, código, razonamiento y conocimiento general, y escala efectivamente en modelos pequeños (3B) sin necesidad de etiquetas humanas ni verificadores externos.

---

## Tabla de referencia rápida

| # | Título (abreviado) | arXiv ID | Año | Venue |
|---|-------------------|----------|-----|-------|
| 1 | Self-Rewarding Language Models | 2401.10020 | 2024 | ICML 2024 |
| 2 | SPIN | 2401.01335 | 2024 | ICML 2024 |
| 3 | Meta-Rewarding LMs | 2407.19594 | 2024 | — |
| 4 | CREAM | 2410.12735 | 2024 | ICLR 2025 |
| 5 | Process-based Self-Rewarding | 2503.03746 | 2025 | ACL 2025 |
| 6 | CSR (visión-lenguaje) | 2405.14622 | 2024 | NeurIPS 2024 |
| 7 | Vision-SR1 | 2508.19652 | 2025 | — |
| 8 | Self-Taught Evaluators | 2408.02666 | 2024 | — |
| 9 | SPPO | 2405.00675 | 2024 | NeurIPS 2024 |
| 10 | DICE | 2406.09760 | 2024 | ICLR 2025 |
| 11 | DNO | 2404.03715 | 2024 | — |
| 12 | LaTRO | 2411.04282 | 2024 | — |
| 13 | ScPO | 2411.04109 | 2024 | — |
| 14 | SER | 2411.00418 | 2024 | — |
| 15 | SCIR | 2502.08922 | 2025 | — |
| 16 | Self-rewarding correction (math) | 2502.19613 | 2025 | — |
| 17 | Self-Aligned Reward (SAR) | 2509.05489 | 2025 | — |
| 18 | Temporal Self-Rewarding | 2508.06026 | 2025 | — |
| 19 | CoVo | 2506.08745 | 2025 | — |
| 20 | Can LRMs Self-Train? | 2505.21444 | 2025 | — |
| 21 | RLSR | 2505.08827 | 2025 | — |
| 22 | Multi-Agent Evolve | 2510.23595 | 2025 | — |