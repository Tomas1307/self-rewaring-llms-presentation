---
name: rafailov-2023-dpo
description: Resumen del paper DPO — Rafailov et al. 2023, "Direct Preference Optimization", NeurIPS 2023, arXiv:2305.18290.
metadata:
  type: summary
  tags: [dpo, preferencias, rlhf, optimizacion, kl, bradley-terry]
  last_updated: 2026-05-12
---

# Rafailov et al. (2023) — Direct Preference Optimization

## Cita

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (Stanford). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. arXiv:2305.18290, mayo 2023. **NeurIPS 2023**.

- **arXiv**: https://arxiv.org/abs/2305.18290
- **NeurIPS proceedings**: https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf

## Resumen técnico

DPO reemplaza el pipeline RLHF de tres etapas (SFT → reward model → PPO con KL) por una **pérdida de clasificación binaria directa** sobre pares de preferencia. La intuición central — y el subtítulo del paper — es que **el propio LLM, parametrizado vía log-ratio respecto a un modelo de referencia, ya codifica una recompensa implícita**.

**Derivación en tres pasos**:

1. **Problema KL-restringido de RLHF**:
   ```
   max_π  E_{x,y~π} [ r(x,y) ]  −  β · KL( π ‖ π_ref )
   ```

2. **Solución cerrada** (multiplicadores de Lagrange):
   ```
   π*(y|x) = (1 / Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
   ```

3. **Inversión**: despejar `r`:
   ```
   r(x,y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
   ```

Insertando esta `r` implícita en el **modelo Bradley-Terry**:
```
P(y_w ≻ y_l | x) = σ( r(x, y_w) − r(x, y_l) )
```

la constante de partición `Z(x)` se **cancela en la diferencia**, dejando una pérdida que solo depende de `π_θ` y `π_ref`.

## Aporte clave para la presentación

**Sin DPO no hay Self-Rewarding viable.** DPO permite:

1. Optimizar política directamente sobre pares auto-generados.
2. Sin entrenar reward model separado.
3. Sin sampling on-policy (offline-friendly).
4. Sin infraestructura PPO.

Es el sustrato que cierra el loop de auto-mejora — generar → juzgar → DPO → repetir. Ver [[synthesis/pilar-3-dpo-en-loop]].

Además, la equivalencia DPO ↔ MaxEnt-IRL es el **puente formal del Pilar 1** ([[synthesis/pilar-1-irl-implicito]]): la `r̂_θ` implícita que DPO recupera es estructuralmente la solución MaxEnt-IRL con prior `π_ref`.

## Fórmulas centrales

### Pérdida DPO

```
L_DPO(π_θ; π_ref) = − E_{(x, y_w, y_l) ~ D} [
    log σ(
        β · log( π_θ(y_w | x) / π_ref(y_w | x) )
      − β · log( π_θ(y_l | x) / π_ref(y_l | x) )
    )
]
```

**Símbolos**:
- `π_θ`: política actual (modelo en entrenamiento).
- `π_ref`: política de referencia, típicamente el modelo SFT inicial, congelado.
- `β`: coeficiente que regula la divergencia KL respecto a `π_ref`. Típico: β ∈ [0.05, 0.5]; Yuan 2024 usa β = 0.1.
- `σ`: sigmoide logístico.
- `(x, y_w, y_l)`: prompt, respuesta preferida ("chosen") y rechazada ("rejected").
- `D`: dataset de pares de preferencias.

### Recompensa implícita aprendida

```
r̂_θ(x, y) = β · log( π_θ(y | x) / π_ref(y | x) )
```

Esta es la "reward function" que el modelo está aprendiendo *implícitamente*. No hay reward model separado; la política *es* el reward model.

### Equivalencia con RLHF KL-restringido

La política óptima de DPO es la misma que la del objetivo:
```
max_π  E_{x, y~π} [ r(x,y) ]  −  β · D_KL( π(·|x) ‖ π_ref(·|x) )
```

con `r(x,y) = r̂_θ(x,y)` (recuperada implícitamente del modelo de Bradley-Terry).

## Resultados empíricos clave

**Control de sentimiento (IMDb)**: DPO supera a PPO de forma robusta a lo largo del frontier *reward vs. KL*.

**TL;DR summarization**:
- DPO: ~61 % win rate (evaluado por GPT-4).
- PPO en su mejor temperatura: ~57 %.
- DPO es además **más robusto al sampling temperature** durante inferencia.

**Anthropic HH (single-turn dialogue)**: DPO es el único método que mejora consistentemente sobre las respuestas "chosen" del propio test set, con **58 %** de preferencia humana sobre PPO.

## Limitaciones conocidas

1. **Asume Bradley-Terry.** Ignora preferencias intransitivas o estocásticas no-BT. [[summaries/azar-2023-ipo]] (IPO) muestra que esto causa colapso bajo preferencias cuasi-deterministas.
2. **Depende fuertemente de `π_ref`.** Si la referencia es pobre, DPO hereda el sesgo. En Self-Rewarding la `π_ref` se mueve cada iteración (`π_ref = M_t`), lo que acumula drift respecto al SFT inicial.
3. **Sensible a la calidad de los pares.** Pares ruidosos o casi idénticos colapsan la señal (gradiente → 0). Relevante para [[synthesis/diagnostico-fallos]] §2.1.
4. **No realiza sampling on-policy** durante optimización. Genera *distribution drift* respecto a `π_θ` en regímenes muy off-policy.
5. **β requiere ajuste.** Valores incorrectos producen sobre/sub-ajuste a las preferencias. Yuan 2024 usa β = 0.1; literatura general: 0.05–0.5.

## Variantes posteriores (ver también)

- **IPO** ([[summaries/azar-2023-ipo]], Azar 2023): regulariza con ψ = identidad, evita colapso bajo preferencias deterministas.
- **KTO** (Ethayarajh 2024, arXiv:2402.01306): elimina la necesidad de pares — usa solo etiquetas binarias.
- **ORPO** (Hong 2024, arXiv:2403.07691): elimina `π_ref` y combina SFT + alineación en un paso.

Ver [[synthesis/comparacion-metodos]] para la tabla comparativa (WIP).

## Conexiones en el wiki

- [[concepts/dpo]] — derivación detallada y notación expandida.
- [[concepts/kl-beta]] — análisis del coeficiente β (WIP).
- [[concepts/preference-pair]] — modelo Bradley-Terry y construcción de pares.
- [[synthesis/pilar-3-dpo-en-loop]] — cómo DPO encaja en el loop de Self-Rewarding.
- [[synthesis/pilar-1-irl-implicito]] — equivalencia con MaxEnt-IRL.
- [[summaries/yuan-2024-self-rewarding]] — paper ancla que usa DPO.
- [[summaries/ouyang-2022-instructgpt]] — RLHF clásico (PPO) que DPO reemplaza.
