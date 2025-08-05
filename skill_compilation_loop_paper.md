
# Towards Human‑Centric Skill Compilation in Multi‑Agent Systems  
*D. Visca & A. Colleague, 2025 (position paper)*  

---
## Abstract  
We hypothesise that a multi‑agent architecture which **compiles frequently‑successful action traces into inexpensive “skills”** will (i) cut inference cost and latency without hurting task accuracy and (ii) improve robustness under resource stressors that mimic human fatigue. Grounded in cognitive‑science models of skill acquisition and relevance realisation, we outline an eight‑phase skill‑compilation loop and propose a three‑part experimental programme with crisp quantitative gates.

---
## 1 Background & Motivation  
Humans convert declarative instructions into rapid, automatic routines via basal‑ganglia “chunking” (ACT‑R). Astronaut training accelerates this by relentless simulation under sleep loss and equipment faults. Current LLM‑centric agents rarely compile such routines; they regenerate entire tool chains, inflating cost and fragility. Hierarchical RL and agent‑distillation results hint at feasibility but lack a closed cognitive loop. We therefore test:  

> **H1** — Skill compilation, guided by human cognitive principles, yields a Pareto improvement in **accuracy, $‑cost, and latency** over naïve recursive agents.

---
## 2 Skill‑Compilation Loop (Eight Phases)  

| # | Phase | Action | Artefact | Success gate |
|---|-------|--------|----------|--------------|
| 1 | Declarative scaffold | SME records ≤ 20 canonical demos; auto‑transcribe to task graph. | `graph_v0.json` | Graph solves ≥ 90 % of demos. |
| 2 | Trace logging | Append `{ctx_emb, tool, args, result, cost, success}` for every call. | `trace_…parquet` | < 0.1 % trace loss. |
| 3 | Pattern mining | PrefixSpan on last 10 k traces; keep subseqs with support ≥ 2 %, success ≥ 95 %, cost ≤ 0.6×baseline. | `C_t.csv` | ≥ 85 % candidates pass unit tests. |
| 4 | Compilation / distillation | Freeze into LoRA adapter (text) **or** Python macro (I/O). | `skills.db` + binary | Byte‑equivalent on 98/100 validation samples. |
| 5 | Validation harness | 100 unit + fuzz tests per skill; record pass‑rate, latency, Δcost. | `test_report.html` | Unit ≥ 95 %, fuzz ≥ 85 %, Δcost ≤ 0. |
| 6 | Retrieval & salience gate | Small encoder ranks skills; call top‑1 if confidence ≥ τ (0.8). | `gate_model.onnx` | Precision@1 ≥ 90 % on 500 held‑out queries. |
| 7 | Perturbation training | Self‑play with 404s, latency spikes, context truncation, budget cuts. | `sim_trace.parquet` | Compiled agent ≥ 85 % success vs vanilla ≤ 65 %. |
| 8 | Skill registry & lifecycle | Evict if usage < 5/wk or two CI fails; quarterly SME audit. | `eviction_log.csv` | ≤ 500 active skills, ≥ 95 % call coverage. |

---
## 3 Experimental Plan  

| Exp | Objective | Setting & Metrics | Success threshold |
|-----|-----------|------------------|-------------------|
| **E1 Efficiency** | Validate Pareto gain | 50 Tool‑Bench tasks; compare cost, latency, accuracy. | Cost ↓ ≥ 40 %, latency ↓ ≥ 30 %, accuracy loss ≤ 1 pp. |
| **E2 Robustness** | Stress‑tolerance | Inject 10 % tool outages + 20 % context loss. | Compiled ≥ 85 % success; baseline ≤ 65 %. |
| **E3 Learning curve** | Human‑like power law | Plot success vs traces compiled. | Linear log‑log trend (R² ≥ 0.9). |

---
## 4 System‑Level KPIs (continuous)  

| KPI | Target |
|-----|--------|
| Task accuracy (50‑task set) | ≥ Baseline – 1 pp |
| Mean $‑cost / task | ≤ 0.6×Baseline |
| P95 latency | ≤ 0.7×Baseline |
| Slow‑path escalations | ≤ 10 % of requests |
| CI critical regressions | 0 per cycle |

Seven consecutive days within bounds ⇒ **95 % confidence** improvements are stable.

---
## 5 Expected Contributions  
1. A closed, cognitive‑inspired compilation loop bridging symbolic task graphs and statistical adapters.  
2. An open evaluation protocol for **cost‑aware agent competence**.  
3. Empirical evidence that human‑centred design principles—relevance filtering, deliberate perturbation, and chunking—translate into tangible gains for multi‑agent systems.

---
## References (abridged)  
* ACT‑R Workshop 2024.  
* HRL‑MA: Macro‑Action Hierarchical RL, NIPS 2025.  
* Agent Distillation for Tool Use, MLSys 2025.  
* REPLUG: Retrieval‑Augmented Prompting, EMNLP 2024.  
* NASA Sleep Operations Manual, 2023.  
* Vervaeke, J. Relevance Realisation Lectures, 2025.  
* Power‑Law of Practice Revisited, Cognition 2024.  
