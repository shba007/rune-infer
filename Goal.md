# Rune Infer — Goal

> This file states direction, not status. What works today is in `README.md`, `AGENTS.md` and `TODO.md`, and the
> code is the source of truth over any document. Decisions recorded here were
> made by the owner on 2026-11-16.
## The whole goal

Rune is a local, Jarvis-like AI harness. You talk or type, or it proposes work
itself; it does the work on your own devices with your own models, tools and
memory; it shows the real result of every step while it works; and it asks
before anything risky.

### Principles

1. **Local first, with a fixed fallback order.** A model runs on the device in
   use when it fits. If it does not fit, it runs on the user's own PC. If it
   does not fit there, it runs in the cloud. If all three fail for that model,
   the next equivalent model is tried in the same order. A cloud-only model
   goes straight to the cloud. Cloud is optional in general and is the expected
   path only for devices too weak to run models (laptop, phone, later IoT).
2. **Strict ownership.** Each concern has exactly one owner (table below).
3. **Three kinds of step, one owner each.** A model call belongs to Infer, a
   tool call to Kit, a screen action to Studio. A tool or model call is
   preferred; a UI Automation element comes next; pixels are last.
4. **Show the thing, not a description.** Every step shows what it actually
   produced (the email text, the table rows, the photos) and decisions are made
   on that content. The view is generated automatically from the schema the
   owner publishes: Kit for tools, Infer for model operations. No model
   generates a layout, and screen captures are not used as a substitute view.
5. **Live and replayable.** Steps appear as they start. The same event stream
   drives the live view and playback. The pill shows a glance; a workbench
   window shows and decides.
6. **Decisions are choices.** Several options, one suggested, each showing its
   outcome on the real content. Not a bare yes or no.
7. **A model per step.** Studio chooses which model a step needs, by the step's
   difficulty and the speed required. Infer decides where that model runs,
   following principle 1.
8. **Sub-agents when the goal needs them.** The plan shown to the user states
   whether one agent or an orchestrator with several sub-agents will do the
   work. Simple requests use one agent. When sub-agents are used they work in
   parallel wherever their work does not conflict.
9. **Autonomous by proposal.** Rune looks for useful work without being asked,
   by observing through tools (inbox, calendar, files), never by watching the
   screen continuously. By default it only suggests. Preparing work for
   approval is opt-in per area. Content from outside never raises what it may
   do unasked.
10. **Ask before assuming.** When a request lacks something needed to plan, or
    contradicts itself, Rune asks the user directly before planning: few
    questions, each with options and a suggested answer. It does not ask what
    it can look up or safely default, and it does not plan on a guess.
11. **Plans can grow.** An approved plan is a starting point. While working, a
    step may be added when a need is discovered, such as looking something up,
    fetching a file or producing something with a model. Added steps are shown
    as added and pass the same safety gate as planned ones.
12. **Shared materials with provenance.** Everything gathered or produced for a
    goal is kept in one place that every step and sub-agent can reuse, so the
    same item is not fetched or produced twice. Each item records where it came
    from: its source and licence if fetched, the model if generated. Claims
    taken from outside carry their source. Requests that leave the machine,
    such as a web search, must not contain sensitive data.
13. **Drafts, not restarts.** A result is kept as parts that depend on each
    other. The user can give feedback on any part; only what depends on that
    part is redone, and the outcome is a new draft. Earlier drafts are kept for
    comparison and return.
14. **Reuse before building.** Existing engines, MCP servers, research and
    projects are preferred over new code.
15. **Small cycles.** Changes ship as small, deployable steps ordered by the
    80/20 rule.
16. **Safe by default.** Risky actions need confirmation, everything is
    audited, and sensitive data classes stay on the user's own hardware.
17. **Scope for now.** Windows, Linux and Android. Both a personal tool and a
    public product.
18. **A non-goal is not a removal order.** Code that falls outside these goals
    stays as it is. Nothing is removed, disabled or rewritten because it is a
    non-goal unless the owner explicitly approves that removal.

### Ownership

| Repo | Owns |
| --- | --- |
| **rune-studio** | The agent runtime and orchestrator, the user interface, the safety gate, screen actions, voice, the run trace, and the bridge that lets outside agents drive the screen |
| **rune-infer** | Every model and engine, including specialist operations (OCR, detection, generation, speech), and where a model runs: device, home PC or cloud |
| **rune-kit** | Every MCP server the harness uses (first-party plugins, third-party stdio and HTTP servers), skills, the sandbox, and shared external binaries such as ffmpeg |
| **rune-tools** | First-party plugins for Kit |
| **rune-lake** | Memory: the media catalog, long-term memory and the encrypted life store |
| **rune-agent** | Sub-agents (personalities) and their workflows: a prompt, skills, MCP servers and a workflow each |
| **rune-model** | The model weight store, nothing more |
## This repo's goal

Infer is the only thing in Rune that runs a model or an engine, and the only
thing that decides where a model runs. Specialist models (OCR, detection,
segmentation, generation, speech) are model operations owned here; they are
never wrapped as Kit tools.

### Current goals

1. **Publish operations with schemas.** For each operation, publish its input
   and output schema so Studio can offer it to an agent and render its result
   automatically.
2. **Placement with fallback.** Given a model id, run it on this device when it
   fits; otherwise forward to the user's PC; otherwise to the cloud. A
   cloud-only model goes straight to the cloud. Report where it ran. Sensitive
   data classes must not leave the user's own hardware.
3. **Infer-to-Infer forwarding.** Forward every operation, not only chat, to
   another Infer on the home network, and later through a tunnel.
4. **Run on weak devices as a forwarder.** Studio always talks to an embedded
   Infer, so Infer must run on Linux and Android, where it may hold only small
   models or none.
5. **Fast role switching on one consumer GPU.** Swap cost is the main limit on
   how responsive the harness feels.
6. **Voice with low delay**, in both directions, and a wake-word path.
7. **Embeddings for Lake** without evicting the agent model.
8. **Model facts in one place.** Capability and benchmark data that Studio
   needs to choose a model per step lives in Infer's registry.
9. **Provenance for generated output.** Every result reports the model that
   produced it and where it ran, so anything generated can be labelled.

### Not goals

- An agent or tool loop, pipelines or workflows. Callers compose stages.
- Choosing which model a step uses. Studio names the model and its ordered
  equivalents; Infer only decides where each runs.
- Running MCP servers (Kit) or storing memory (Lake).
- New model families that do not serve the harness, such as driving
  trajectories or 3D generation, until the goals above are met.
- New work on the standalone terminal dashboard, cloud object-storage ingestion
  or the disabled Anthropic and Responses routes.

Per principle 18, existing code in these areas is not to be removed without the
owner's explicit permission.

### Where the code is today

- One HTTP endpoint in front of local engines run as supervised subprocesses
  and two native Rust engines, plus cloud adapters with rate limits and retries.
- Memory budgeting, eviction, idle unload, and a priority gate so foreground
  requests pre-empt background ones.
- A management API for roles, the catalog, engines, model downloads and imports.
- The caller names the model and it runs where it is configured; there is no
  fallback between locations yet (`AGENTS.md` rule 6 now states the target). Forwarding to another server covers chat through a
  compatible upstream, not Infer's own operations.
- Embedded in Studio on Windows only.

### Open questions

- How "does not fit" is decided before trying: by the existing memory budget
  check, or by a failed load.
- How an Infer on a phone discovers and authenticates to the PC's Infer.
- Which small models are dependable enough at tool calls to be the default.
