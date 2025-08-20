Reflex (by not4humans) — Product & Engineering Spec v1.0

Working backwards from the developer experience

Purpose: define a clear, deterministic, minimal system that lets any agent compile and apply procedural skills (“capabilities”) as part of its own cognition—without over-engineering. This document is written to survive both academic scrutiny and production use.

⸻

0) North-star user experience (UX) — work backwards

Primary persona
	•	Agent developer (app teams, platform teams, editor/IDE agents). Wants: “one line to get cost/latency gains and reliability, without rewriting my agent or ceding control to a black box.”

UX promises (what it must feel like)
	1.	Drop-in: add a single line to wrap an existing agent; everything else “just works”.
	2.	Agent awareness: the LLM knows which skills exist, can propose using one, and asks clarifying questions when slots are missing.
	3.	Deterministic execution: whether a skill runs is governed by hard preconditions and a single threshold τ; every decision is explainable and replayable.
	4.	Local-first: all value (logging, mining, compiling skills, local CI) works offline; no control plane needed to see ROI.
	5.	Governable at scale (optional): orgs can promote skills across teams/environments, enforce policy, and audit everything—via a small control plane.
	6.	No second model: one LLM is used for both agent reasoning and (when needed) T=0 parsing of slots. No learned routers in the hot path.

Golden path demo (what users actually do)

from not4humans.reflex import Reflex
reflex = Reflex(project="acme/payments",
                compat={"env":["staging"],"toolset":"v6","repo":"github.com/acme/payments"},
                storage="local://./.reflex")

wrapped = reflex.wrap(agent=my_agent, tools=[term_exec, fs_patch], llm_client=openai_chat)
result = wrapped.run("Deploy latest to staging")  # Reflex handles skills end-to-end


⸻

1) Capabilities & product scope

In-scope (V1)
	•	SDK wrapper with auto-wrap modes (one-liner + framework adapters).
	•	Agent–Skill Protocol (ASP): LLM proposes skills via <SKILL_PROPOSE>; asks via <CLARIFY>; receives <SKILL_RESULT>.
	•	Skill Bulletin (top-K Skill Cards) injected into context each turn (LLM awareness).
	•	Deterministic Gate: compat → preconditions → score ≥ τ → policy.
	•	Plan Runner: executes the capability’s internal macro steps (tool calls) with guards, budgets, timeouts, idempotence keys, compensation.
	•	Blackboard: per-turn shared state (ctx, env, work, policy) with auto-seeding and explicit override.
	•	Append-only logging + replay (OTEL + JSONL/Parquet).
	•	Deterministic miner (PrefixSpan) + compiler to capability manifests; local CI harness (unit+fuzz).
	•	Local registry (SQLite + JSON manifests).
	•	Optional control-plane client (4 tiny APIs): policy check, skill sync, promotions, telemetry.

Out-of-scope (V1)
	•	Learned routers, embedding routers, prompt auto-generation beyond pinned T=0 slot parsers.
	•	Training new models or adapters (LoRA); non-contiguous sequence mining.
	•	Global marketplace & public sharing (until promotion/policy is mature).
	•	Heavy telemetry ingestion (raw prompts, secrets). Only redacted aggregates are ever sent.

⸻

2) High-level architecture

┌─────────────────────────────────────────────────────────────────────┐
│                        Developer’s Agent Process                    │
│                                                                     │
│  ┌──────────┐     ┌──────────────────────┐   ┌───────────────────┐  │
│  │  Agent   │<--->│  Reflex ProxyAgent   │<->│  ReflexSession     │  │
│  │  (your)  │     │  (wrap one-liner)    │   │  (per-turn UoW)    │  │
│  └──────────┘     └──────────────────────┘   ├───────────────────┤  │
│                                              │ BulletinBuilder    │  │
│   Tools  ──wrapped by── ToolProxy ──┐        │ ASP (parser)       │  │
│   LLM    ──wrapped by──  LLMProxy ──┼────────│ Gate               │  │
│                                      │        │ PlanRunner         │  │
│  Blackboard (ctx/env/work/policy) ───┘        │ Logging & Replay   │  │
│                                              └───────────────────┘  │
│     Local Registry, Miner, Compiler, Local CI (SQLite + JSON)       │
└─────────────────────────────────────────────────────────────────────┘
                 ▲                               │
                 │                               ▼
   (optional) Control Plane (SaaS / private): policy, promotions, org registry

Design tenets
	•	Facade + Proxy: the Reflex facade exposes the tiny surface; tools and LLM are proxied; the agent remains unchanged.
	•	Chain of Responsibility: Gate enforces deterministic order (compat→precond→score→policy).
	•	Command: steps in a plan are explicit commands with guards/budgets/idempotence.
	•	Repository: registry abstraction (local first; pluggable).
	•	Observer: logging is out-of-band; replay reconstructs decisions.

⸻

3) Capability Manifest (executable spec of a skill)

Minimal yet complete; nothing implicit.

{
  "id": "deploy.container.appservice",
  "version": "1.3.0",
  "owner": "team-platform",
  "scope": "project",                       // project | org

  "compat": {                               // HARD filters
    "env": ["staging"], "toolset": "v6", "repo": "github.com/acme/payments"
  },

  "signature": {
    "inputs": [
      {"name":"app_name","type":"string","required":true},
      {"name":"image_tag","type":"string","required":true}
    ],
    "outputs": [{"name":"endpoint","type":"url"}],
    "blackboard_bindings": {
      "inputs":  {"app_name":"ctx.appservice","image_tag":"work.image_tag"},
      "outputs": {"endpoint":"work.endpoint"}
    }
  },

  "preconditions": {
    "tools_available": ["term.exec", "tests.run"],
    "data_present": ["app_name","image_tag"],
    "invariants": [
      "az.account.subscription == ctx.subscription",
      "deploy_window('staging') == 'open'"
    ]
  },

  "plan": {
    "idempotence_key": "deploy:appservice:{{app_name}}:{{image_tag}}",
    "budget": {"max_cost_usd": 0.10, "max_latency_ms": 5000},
    "steps": [
      {"tool":"term.exec","args":{"cmd":"az webapp config container set --name {{app_name}} --image {{image_tag}}"},
       "timeout_ms":1500},
      {"tool":"term.exec","args":{"cmd":"az webapp restart --name {{app_name}}"},
       "timeout_ms":1000},
      {"tool":"term.exec","args":{"cmd":"curl -sSf https://{{app_name}}.azurewebsites.net/healthz"},
       "timeout_ms":1000}
    ],
    "compensation": [
      {"when":"partial_failure","tool":"term.exec","args":{"cmd":"az webapp config container set --name {{app_name}} --image {{work.prev_image_tag}}"}}
    ],
    "result_map": {"endpoint":"https://{{app_name}}.azurewebsites.net"}
  },

  "effects": {
    "writes": ["work.endpoint"], "external": ["updated:appservice_image"]
  },

  "activation": {
    "required_slots": ["app_name","image_tag"],
    "goal_labels": ["deploy","appservice"],
    "keywords_any": ["deploy","release","push image"],
    "score_weights": {"goal_label":3.0,"keyword_hit":1.0,"recent_success":1.5},
    "tau": 3.5, "cooldown_s": 30, "max_concurrency": 3
  },

  "llm_support": {
    "parsers": [{
      "name":"extract_app_name",
      "model":"gpt-4o-2025-04",
      "temperature":0.0,
      "prompt_ref":"prompts/app_name_extract.v1",
      "inputs":["raw_text"], "outputs":["app_name"]
    }]
  },

  "policy": {
    "allow_roles": ["ReleaseManager"],
    "deny_if": ["risk_scan(args) == 'high'"]
  },

  "tests": {
    "unit_cases": ["tests/deploy_basic.json"],
    "fuzz": {"count": 50, "mutations":["alt_image_tag","missing_slot"]},
    "expected": {"pass_rate_unit":0.95,"pass_rate_fuzz":0.85}
  },

  "provenance": {
    "mined_from_traces": ["trace_2025-08-10T12:30Z_..."],
    "support": 0.041, "success": 0.98, "delta_cost": -0.42, "delta_p95": -0.30
  }
}

Why each field exists & what it prevents
	•	compat → prevents cross-env misfires (“works here, breaks there”).
	•	signature+bindings → explicit state IO; enables replay; avoids hidden coupling.
	•	preconditions → fail-fast safety; avoids costly/risky attempts.
	•	plan w/ idempotence/budget/compensation → predictable side-effects; avoids duplicate actions & runaway latency.
	•	activation → explicit cues & a single τ; avoids opaque routers.
	•	llm_support at T=0 → lets LLM parse slots deterministically; avoids LLM deciding control flow.
	•	tests → CI gate; avoids flaky/expensive skills.
	•	provenance → eviction & trust; avoids “mystery skills”.

⸻

4) Agent–Skill Protocol (ASP)

Injected guidance (system, ~1–2 lines):
“To use a skill, include <SKILL_PROPOSE>{json}</SKILL_PROPOSE>.
If info is missing, reply with <CLARIFY>{json}</CLARIFY>.”

Propose

<SKILL_PROPOSE>
{"skill_id":"deploy.container.appservice",
 "why":"goal=deploy; app=payments-stg-webapp; image ready",
 "inputs":{"app_name":"payments-stg-webapp","image_tag":"acr.io/pay:abc"}}
</SKILL_PROPOSE>

Clarify

<CLARIFY>{"questions":[{"slot":"app_name","question":"Which App Service? (e.g., payments-stg-webapp)"}]}</CLARIFY>

Result (wrapper → LLM/system)

<SKILL_RESULT>
{"skill_id":"deploy.container.appservice","status":"ok",
 "outputs":{"endpoint":"https://payments-stg-webapp.azurewebsites.net/healthz"}}
</SKILL_RESULT>

Parsing & safety
	•	Reflex rejects malformed blocks; no execution without a valid Propose.
	•	If required slots missing and no pinned parser exists, Reflex returns Clarify to the agent/user.

⸻

5) SDK surface (public API contract)

class Reflex:
    def __init__(self, project, compat, storage, control_plane=None,
                 k_cards=5, tau=0.85, epsilon=0.0, redaction=None, policy_mode="enforce"): ...

    def wrap(self, agent, tools, llm_client): ...
    def wrap_langgraph(self, graph, tools, llm_client): ...
    def wrap_langchain(self, chain, tools, llm_client): ...
    def wrap_semantic_kernel(self, kernel, plugins, llm_client): ...
    def session(self, blackboard=None): ...  # advanced/manual

    # Diagnostics
    def replay(self, trace_id, mode="pinned"): ...
    def learn_snapshot(self, window=10000): ...  # mine+compile candidates

Behavior of wrap(...) (one-liner)
	•	Creates a ProxyAgent with an internal ReflexSession per .run() call (turn).
	•	Auto-wraps tools via ToolProxy (timing, retries, redaction, idempotence).
	•	Auto-wraps LLM via LLMProxy (inject Bulletin once; parse ASP; feed SKILL_RESULT back).
	•	Auto-seed Blackboard if not provided; logs snapshot; replay uses this exact state.

Blackboard strategy
	•	Default: blackboard=None → seed from skills.yaml + framework state + environment probes; log snapshot sources.
	•	Override: callers may pass a dict to set/override; Reflex logs diffs.
	•	Scope: per-turn; effects write via blackboard_bindings.

⸻

6) Deterministic Gate

Order (hard-coded):
	1.	compat (project/env/toolset/repo)
	2.	preconditions (slots present, tool flags, invariants)
	3.	score ≥ τ (monotonic: goal label, keyword, entity type, recent success)
	4.	policy (RBAC, allow/deny, PII/secret/residency checks)

Tie-break:
	•	Prefer capability that satisfies the most unmet goal postconditions (progress-to-goal).
	•	If tied, use transition prior (A→B frequency) as advisory only.

Always log a decision event with: compat result, preconditions, score, τ, policy verdict.

⸻

7) Plan Runner (deterministic execution)
	•	Execute steps sequentially; each step has tool, args, timeout_ms, optional guards.
	•	Enforce budget (cost/latency caps); mark partial_failure and run compensation if defined.
	•	Enforce idempotence_key (short-circuit on duplicates; store last successful outputs keyed by the idempotence key).
	•	Emit skill_apply with status, outputs, latencies, costs; map outputs to blackboard per result_map.

⸻

8) Logging, telemetry, replay
	•	Events (append-only JSONL/Parquet + OTEL spans):
llm_request/response, tool_call/result, reasoning_span, decision, skill_apply, blackboard_seed/diff.
	•	Each turn has a single trace_id; nested spans for steps.
	•	Replay: re-executes with pinned versions of manifests, tool wrappers, and blackboard snapshot; compares outputs/decisions.

⸻

9) Mining & compilation (deterministic)
	•	Input: last N traces where status=success and compat matched.
	•	Algorithm: PrefixSpan/SPADE for contiguous subsequences of tool calls.
	•	Candidate filter: support ≥ s, success ≥ p, cost ≤ 0.6×baseline.
	•	Compile: generate Capability Manifest (signature, preconditions from IO shapes & invariants, plan from steps, activation cues from goal/keywords, budgets from observed p95).
	•	CI: auto-generate unit + fuzz suites; must meet thresholds before activation/promotion.

No learning in hot path; all mining & compilation is deterministic and explainable.

⸻

10) Control plane (optional; 4 small APIs)
	•	POST /v1/policy/check → {allow|deny|needs_promotion, reason}
	•	GET /v1/skills/sync?project=&since= → [manifest@version,…]
	•	POST /v1/promotions → run Promotion CI in target environments; return report
	•	POST /v1/telemetry → batched aggregates: {skill_id,calls,success,delta_cost,delta_p95} (no secrets/raw prompts)

Why: gives enterprises governance & audit with minimal surface; SDK remains fully functional offline.

⸻

11) Security & privacy
	•	Redaction: tool proxies support field masks for args/results; LLM logs store hashes for sensitive fields, never raw secrets.
	•	Deterministic parsers: never parse secrets; never auto-create destructive resources without anchors + policy.
	•	Local-first: nothing leaves process unless control-plane is configured; even then, only aggregates.

⸻

12) Performance budgets
	•	Bulletin injection: K ≤ 5, ≤ 120 tokens per card.
	•	SKILL_RESULT message: ≤ 50 tokens.
	•	Logging overhead: <5% added latency per tool call.
	•	Plan budgets: hard caps enforce p95 latency wins.

⸻

13) Observability & SLOs

Per skill
	•	calls, success_rate, delta_cost, delta_p95, fallback_rate, failure streaks, last CI pass.

Per system
	•	learned_skills, active_skills, evictions, ci_failures, precision_at_1, clarify_rate, policy_denies.

SLOs
	•	Decision latency (gate): ≤ 5 ms median (local).
	•	Replay fidelity: 100% decision equivalence on golden traces.

⸻

14) Failure modes & handling
	•	Parser fails → return Clarify; never guess.
	•	Preconditions fail → do not score or run; fall back to agent’s plan.
	•	Budget exceeded → abort plan; run compensation; mark partial_failure.
	•	Idempotence collision → short-circuit with last good outputs (logged).
	•	Policy deny → return structured reason to LLM; agent can adapt.
	•	Control plane down → continue with cached policy & skills; log degraded mode.

⸻

15) Testing strategy
	•	Unit: gate branches; parser determinism; plan runner timeouts/compensation; tool proxy idempotence.
	•	Golden traces: full turn replays; ensure identical decisions/outputs.
	•	Chaos: inject 404s, latency spikes, truncated context; verify compensations and fallbacks.
	•	SDK adapters: one-liner invariants: exactly one bulletin inject per turn; exactly one SKILL_RESULT per execution; no double wrapping.

⸻

16) Versioning & compatibility
	•	Manifest versions are immutable; semantic versioning.
	•	SDK: semver; minor versions must not break the ASP or bulletin format.
	•	Replay: supports “pinned” mode tied to manifest + SDK compat shim.

⸻

17) Rollout plan (minimal viable sequence)
	1.	SDK core: ProxyAgent, Session, Bulletin, ASP, Gate, PlanRunner, ToolProxy, LLMProxy, Logging & Replay.
	2.	Local registry + miner + compiler + CI.
	3.	Adapters: LangGraph → LangChain → Semantic Kernel.
	4.	Control-plane client (optional).
	5.	Docs & demos: vanilla loop, graph, chain, editor.

⸻

18) Design decisions (ledger — why & what they protect)
	1.	One-liner wrap with auto-wrap tools/LLM
Why: frictionless adoption; Protects: framework lock-in & missed hooks.
	2.	Skill Cards (top-K) instead of streaming the registry
Why: token hygiene + focus; Protects: context bloat, confusion.
	3.	Explicit ASP blocks
Why: auditable intent; Protects: accidental tool spam.
	4.	Gate order is fixed
Why: precision-first; Protects: misfires in wrong env.
	5.	LLM only for T=0 parsing
Why: determinism; Protects: non-reproducible decisions.
	6.	Idempotence keys & compensation mandatory for external side-effects
Why: safe reruns; Protects: duplicate resources, partial failures.
	7.	Append-only logs + replay
Why: credibility; Protects: “can’t reproduce” incidents.
	8.	Local-first, pluggable registry
Why: zero infra adoption; Protects: vendor lock-in.
	9.	Four control-plane APIs only
Why: minimal SaaS surface; Protects: ops complexity.
	10.	Monotonic scoring + single τ
Why: explainable “skill temperature”; Protects: opaque routers drift.

⸻

19) Example end-to-end (Windsurf deploy)
	•	Turn start: Reflex seeds blackboard (skills.yaml + env).
	•	Bulletin: shows ensure_acr, docker_build_and_push, deploy.container.appservice.
	•	LLM proposes ensure_acr; Gate approves; PlanRunner no-ops (already exists).
	•	LLM proposes docker_build_and_push; runner builds/pushes; result includes image_tag.
	•	LLM asks <CLARIFY> for app_name; user answers.
	•	LLM proposes deploy.container.appservice; Gate checks subscription & window; runner deploys; smoke test OK; SKILL_RESULT contains endpoint.
	•	Logs show decisions, timings, costs; miner learns repeated sequence; compiler generates/updates manifests; local CI passes.

⸻

20) Risks & tripwires
	•	Risk: adapters can’t intercept first LLM call → Tripwire: if an adapter can’t hook “first model call”, fall back to wrap() boundary or require explicit decorator; block adapter GA until invariant holds.
	•	Risk: context bloat from Skill Cards → Tripwire: enforce K≤5; warn at K>5; emit token budget metric.
	•	Risk: “magic” blackboard → Tripwire: always log snapshot + sources; disallow reseeding on replay.
	•	Risk: skill explosion → Tripwire: eviction policy (usage<5/wk or 2 CI fails); dashboard alert on registry growth.

⸻

21) Minimal repo layout (implementation guide)

reflexes/
  sdk/
    reflex.py            # Facade (wrap, adapters)
    session.py           # per-turn Unit of Work
    wrappers.py          # ToolProxy, LLMProxy
    bulletin.py          # Skill Card selection
    asp.py               # protocol parser/formatter
    gate.py              # compat→precond→score→policy
    executor.py          # plan runner
    blackboard.py        # seeding, diffs
  logging/
    events.py            # schemas
    otel.py              # spans/metrics
    sinks.py             # JSONL/Parquet rotation
    replay.py            # deterministic re-execution
  registry/
    local.py             # SQLite + manifest CRUD
    models.py            # manifest dataclasses
    sync.py              # control-plane client
  miner/
    prefixspan.py        # mining
    compiler.py          # manifest generator
    ci.py                # local test harness
  adapters/
    langgraph.py
    langchain.py
    semantic_kernel.py
  cli/
    reflex               # trace show, learn snapshot, skill test, replay
  docs/
    quickstart.md
    agent_skill_protocol.md
    manifest_schema.md
    integration_checklist.md


⸻

Closing note

This spec keeps Reflex small, deterministic, and useful on day one: the LLM becomes aware of concise, high-value skills; proposals are explicit; executions are safe and auditable. Everything else—mining, promotion, policy—is deterministic plumbing you can add incrementally.

You can implement straight from this document: start with Reflex.wrap + Session + Bulletin/ASP/Gate/Executor, ship the vanilla demo, then iterate on adapters and the local registry/miner.