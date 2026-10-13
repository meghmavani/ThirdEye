# ThirdEye — Requirements Specification

**Project:** ThirdEye
**Version:** 2.0 · **Date:** 2026-10-12 · **Status:** Draft for team review · **Supersedes:** v1.0 (2026-09-18, LEGO-first)

**Source of truth:** `ThirdEye_Project_Plan.md` (plan revised 2026-10-12). Every requirement below traces to a plan section; where this document and the plan disagree, the plan wins and this document is a bug.

**What changed in v2.0.** The project pivoted from LEGO-first to **Jenga + jigsaw**; LEGO is parked. Requirement IDs are stable and are cited in the code, so nothing was renumbered. Requirements that no longer apply are marked **superseded** or **parked** in place, requirements that moved procedure were re-worded and re-prioritised, and new requirements take the next free number in their class (`FR-JIG`, `FR-DEP`, `DR-JIG`, …). The milestone gates keep their IDs but their content was reset for the new schedule (CC-5).

---

## 1. Purpose and scope

### 1.1 Purpose

This document turns the ThirdEye project plan into a numbered, testable set of requirements. It exists so that (a) each module owner knows exactly what "done" means for their module, (b) the evaluation in week 11 has pre-agreed pass/fail criteria rather than post-hoc ones, and (c) procurement and environment setup is a checklist rather than a scramble.

### 1.2 Product scope

ThirdEye observes a person performing a physical procedure through fixed cameras and raises real-time alerts when a step is missed, performed out of order, or performed incorrectly. The procedure itself is not hard-coded: an LLM compiles a written rulebook or instruction sheet into a machine-readable **procedure spec**, and a deterministic checker compares observed world state against that spec.

| Procedure | Spec form | State model | Cameras | Role |
| --- | --- | --- | --- | --- |
| **Jenga** (game rules) | Constraint set, no fixed order | Tower lattice (18 × 3) | iPhone LiDAR (colour + depth) at one corner, a second camera at the opposite corner | Must-have; the generality result and live adversarial demo |
| **Jigsaw** (instruction sheet) | Dependency graph (DAG): corners → edges → interior | Piece grid | One overhead webcam over a board with four ArUco markers | Must-have; the ordered-procedure counterpart |
| LEGO assembly (Model A, Model B) | Dependency graph (DAG) | Stud grid | Top-down camera | **Parked.** Code and tests kept green; no new work |

### 1.3 Document scope

Covered: functional, non-functional, data, hardware, software, interface, evaluation and milestone-gate requirements, plus the open items that block requirements from being finalised.

Not covered: implementation design (plan §2 and §4), work allocation (plan §9), and the report/poster content (plan §11).

### 1.4 Definitions

| Term | Meaning |
| --- | --- |
| **Procedure spec** | JSON document compiled from a rulebook or instruction sheet; either DAG form (plan §4.2) or constraint form (§4.3) |
| **World state** | Current physical configuration as tracked by the system: the Jenga 18 × 3 lattice, the jigsaw piece grid, or (parked) the LEGO stud grid |
| **Event** | A discrete observed change (`ADD`, `REMOVE`, `MOVE`, `STEP_START`, `STEP_END`, `COLLAPSE`) emitted by a state tracker or the temporal model |
| **Checker** | Deterministic engine that evaluates spec preconditions against events and raises alerts |
| **Hard dependency** | Physically necessary precondition (support). A violation indicates *our* perception failed, never a user error |
| **Soft dependency** | Order stated by the instructions. A violation is a genuine out-of-order user error |
| **Occlusion window** | Interval during which a hand covers the workspace and the state tracker is frozen |
| **Diagonal-corner mode** | Jenga viewing geometry: each camera sits at a corner of the tower and sees two faces at an angle; two cameras are merged |
| **Unidentified piece** | A jigsaw piece whose best match against the box picture is not clearly better than the second-best. It is never alerted on |
| **Tier 1 / 2 / 3** | The generality ladder of plan §1.2: Jenga constraints / jigsaw as a new state model / an unseen source with no code change |

### 1.5 Identifier and priority scheme

Requirement IDs are `<CLASS>-<MODULE>-<n>` and are stable — never renumber, mark superseded instead.

| Class | Meaning |
| --- | --- |
| `FR` | Functional requirement |
| `NFR` | Non-functional requirement |
| `DR` | Data requirement |
| `HW` | Hardware requirement |
| `SW` | Software / environment requirement |
| `IF` | Interface / contract requirement |
| `EV` | Evaluation acceptance criterion |
| `MG` | Milestone gate |

Priority uses MoSCoW, aligned to plan §6:

| Priority | Meaning |
| --- | --- |
| **M** | Must have — the minimum viable project. Failure here fails the project |
| **S1**–**S6** | Should have, in the plan's priority order. Delivered only once all **M** items work end to end |
| **C** | Could have / stretch. Attempted only if week 13 slack is genuinely clear |
| **P** | Parked. Built or partly built for the earlier LEGO plan; the code and its tests stay green but no new work is planned |
| **W** | Won't have this semester — recorded so it is not silently re-scoped in |
| ~~Superseded~~ | Replaced by a named newer requirement; kept for stable numbering |

Verification method per requirement: **T** = automated test, **D** = live demonstration, **A** = analysis / measurement against recorded data, **I** = inspection or review.

**Built** in the *Status* sense of this document means implemented and covered by the 229-test suite as of 2026-10-12; it does not mean evaluated against real video.

---

## 2. Stakeholders

| Stakeholder | Interest | Requirements they sign off |
| --- | --- | --- |
| Course panel / evaluators | Working live demo, defensible results | FR-UI, EV-*, MG-* |
| Perception lead | Calibration, depth input, slot reading, piece matching, state trackers | FR-CAL, FR-PER, FR-STA, FR-DEP, FR-JIG |
| Temporal model lead | Features, MS-TCN++, benchmark scripts | FR-TMP, EV-SEG |
| Reasoning lead | Schemas, compiler, validator, checker, explainer | FR-CMP, FR-CHK, FR-EXP, IF-* |
| App lead | Streaming backend, UI, integration, latency | FR-UI, NFR-PERF, IF-WS |
| Demo participants (panel member, teammates) | Can operate it without training | NFR-USE |
| Recorded subjects | Consent and release terms for published data | NFR-PRIV, DR-REL |

---

## 3. Functional requirements

### 3.1 FR-CMP — LLM procedure compiler (plan §2.1, §4.2, §4.3)

Status: both forms are **built** for the LEGO and Jenga instances; the jigsaw schema extension and the real source texts are to do.

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-CMP-1 | The compiler shall accept a plain-text instruction sheet (one instruction per step) and emit a procedure spec in **DAG form** conforming to the §4.2 schema, for the `piece_grid` state model (jigsaw) and, as built, the `stud_grid` model (LEGO). | M | T |
| FR-CMP-2 | For each step the compiler shall emit: step id, verbatim instruction, part descriptor (a piece id for jigsaw; colour + size for LEGO), target pose (cell and rotation for jigsaw; x, y, layer, rotation for LEGO), `hard_depends_on`, `soft_depends_on`, `optional`, and an expected duration range. | M | T |
| FR-CMP-3 | The compiler shall distinguish **hard** (physical support) from **soft** (stated order) dependencies and record both independently. For jigsaw, `hard_depends_on` is normally empty. | M | T |
| FR-CMP-4 | The compiler shall accept a rulebook and emit a procedure spec in **constraint form**: one or more action schemas, each with preconditions (`expr`, human-readable `text`, `violation` code) and effects, plus terminal conditions. | M | T |
| FR-CMP-5 | Precondition expressions shall be drawn from a **fixed predicate vocabulary supplied in the prompt**, checked against a whitelist when compiled, and evaluated by a small safe interpreter. Python `eval` or any equivalent arbitrary-code path shall not be used. | M | I, T |
| FR-CMP-6 | The validator shall reject a spec that fails any of: schema conformance, dependency-cycle check, collision check (two parts occupying the same cell), support check (no floating parts, for stacked parts only) and — for constraint specs — a satisfiability check that at least one legal action exists from the initial state. | M | T |
| FR-CMP-7 | On a validation failure the system shall report the failing check and the offending step/constraint id, and shall permit **one automated repair round** through the LLM before requiring human intervention. | M | T |
| FR-CMP-8 | The system shall provide a **teach-by-demonstration** path: record one correct performance, capture the world state after each action, and use the LLM only to name the resulting steps. | C | D |
| FR-CMP-9 | The UI shall allow a human to edit and re-validate a compiled spec before a session runs. | S6 | D |
| FR-CMP-10 | Compilation shall be an offline, once-per-procedure operation. No LLM call shall occur on the per-frame or per-event path. | M | I |
| FR-CMP-11 | The LLM shall never be given raw video or video-derived imagery. Its inputs are text only. | M | I |
| FR-CMP-12 | The compiler shall be exercised on a **Jenga rulebook** (`manuals/jenga_rules.txt`) and a **jigsaw instruction sheet**, each compiled to a validated spec that is committed under `specs/`. | M | T, D |

### 3.2 FR-CAL — Calibration and cameras (plan §2.2)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-CAL-1 | The system shall detect four printed ArUco markers at the board corners and compute a homography from image pixels to **board-cell coordinates** (stud coordinates for the parked LEGO plate). *Built.* | M | T, D |
| FR-CAL-2 | Calibration shall succeed from any reasonable overhead camera angle and shall not require the camera to be perfectly perpendicular. | M | D |
| FR-CAL-3 | The homography shall be recomputed every 10 frames, or immediately when marker positions shift beyond a threshold. | M | T |
| FR-CAL-4 | A manual four-corner click fallback shall be available when marker detection fails. | S1 | D |
| FR-CAL-5 | ~~LEGO side camera for height.~~ **Parked** — see FR-CAL-8. The LEGO height path is not extended. | P | — |
| FR-CAL-6 | ~~Two side cameras at 90° for Jenga.~~ **Superseded by FR-CAL-8** (diagonal-corner geometry replaces orthogonal straight-on views). | Superseded | — |
| FR-CAL-7 | The system shall detect and report loss of calibration (markers occluded or out of frame) rather than silently emitting wrong coordinates. *Built.* | M | T |
| FR-CAL-8 | For Jenga the system shall support **two cameras at opposite corners of the tower**, each seeing two faces at an angle, calibrated against table markers around the base plus known block dimensions (75 × 25 × 15 mm) for scale, with the two views merged into one lattice. | M | D |
| FR-CAL-9 | The system shall ingest an **iPhone LiDAR** colour + depth stream (Record3D) and shall also accept recorded RGB-D files so that development does not need the phone. | M | D, T |
| FR-CAL-10 | The time offset between the iPhone and the second camera shall be measured and corrected, so that a single move is attributed to one timestamp within one frame interval. | M | A |

### 3.3 FR-PER — Perception (plan §2.3)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-PER-1 | The system shall detect hand keypoints every frame and derive: hand-over-workspace, grasp, and **number of hands in contact**. *Built.* | M | T |
| FR-PER-2 | ~~Fine-tuned YOLO detector for LEGO parts by colour and size.~~ **Parked.** The detector stays in `perception/detect.py`. | P | — |
| FR-PER-3 | ~~Zero-shot open-vocabulary detector as fallback and ablation baseline.~~ **Parked** with FR-PER-2. | P | — |
| FR-PER-4 | The system shall maintain stable part identities across frames using a multi-object tracker, so that part motion can be attributed. | M | T |
| FR-PER-5 | The Jenga perception shall use a **single block class**, so that a Jenga failure is attributable to state or reasoning logic rather than to detection. | M | I |
| FR-PER-6 | While a hand occludes the workspace the world state shall be **frozen**; updates resume only after the hand has left. *Built.* | M | T |

### 3.3a FR-DEP — Jenga depth and slot reading (plan §2.3, §4.6)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-DEP-1 | The system shall derive the occupancy of each of the 54 slots from depth, with a colour cue, in diagonal-corner mode. | M | T, A |
| FR-DEP-2 | The system shall merge the readings of the two cameras into a single lattice, and shall mark a slot **unobserved** when neither camera can see it, rather than assuming empty. | M | T |
| FR-DEP-3 | Where a slot's reading is unobserved or contradictory the checker shall wait for a clearer view and not raise an alert (FR-CHK-7). | M | T |
| FR-DEP-4 | The system shall detect a block **pushed out** of its slot from depth. | S2 | A |
| FR-DEP-5 | If depth thresholds prove insufficient, a small slot-crop classifier (~150 labelled frames) shall be provided as the fallback. | C | A |

### 3.3b FR-JIG — Jigsaw piece identification (plan §2.3, §4.6)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-JIG-1 | The spec schema shall support a `piece_grid` state model: grid dimensions, piece ids taken from the box picture, and a pose of cell + rotation in {0, 90, 180, 270}. | M | T |
| FR-JIG-2 | The system shall segment pieces on the board plane from the overhead view. | M | T, A |
| FR-JIG-3 | The system shall identify each segmented piece by **matching it against the box picture**: which cell it belongs to and at what rotation. No detector shall need training. | M | T, A |
| FR-JIG-4 | Matching shall report the margin between the best and second-best match. A piece below the margin threshold is **unidentified**, is logged, and shall never produce an alert. | M | T |
| FR-JIG-5 | A wrong cell and a wrong rotation shall be distinguishable: right piece in the wrong cell; right cell with the wrong rotation. | M | T |
| FR-JIG-6 | The matcher shall be developed and evaluated on at least two distinct puzzles, one of them held out. | S4 | A |

### 3.4 FR-STA — World-state tracker (plan §2.4)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-STA-1 | The world-state tracker shall be an **interface** exposing `current()` and `diff()`, with one independent implementation per procedure. No checker code shall branch on procedure type. *Built.* | M | I, T |
| FR-STA-2 | The LEGO tracker shall maintain the model as a list of placed bricks `(type, x, y, layer, rotation)` in stud coordinates. *Built.* | P | T |
| FR-STA-3 | A part (brick or piece) shall be counted as placed only after it has remained still for ≈0.5 s following a hand exit. | M | T |
| FR-STA-4 | The tracker shall emit a state diff after each placement or removal as an event, e.g. `ADD piece r0c1 @ (1,0,rot0)`. | M | T |
| FR-STA-5 | ~~LEGO layer inference with a side camera.~~ **Parked** with LEGO; jigsaw has no height. | P | — |
| FR-STA-6 | The tracker shall support a **forced re-verification** request for named prior steps, resolved once the workspace view is clear (see FR-CHK-7). *Built.* | M | T |
| FR-STA-7 | The Jenga tracker shall maintain an 18 × 3 occupancy array plus per-layer orientation, filled from the camera views (diagonal-corner mode, FR-DEP-1/2). The lattice and its occupancy snapshot are *built*; filling them from depth is to do. | M | T |
| FR-STA-8 | The Jenga tracker shall emit `ADD (layer, slot)` and `REMOVE (layer, slot)` events, pair them into one `MOVE`, and emit `COLLAPSE` as a terminal event detected from a large single-frame drop in tower height. *Built.* | M | T, D |
| FR-STA-9 | Jenga preconditions shall be judged against the tower **as it stood before the move**, advancing the baseline only from a reading with no block in hand. *Built.* | M | T |
| FR-STA-10 | The jigsaw tracker shall maintain a `cols × rows` occupancy of cells, each holding the identified piece and rotation, or *unidentified*, or empty. | M | T |

### 3.5 FR-TMP — Step recognizer (plan §2.5)

The step recognizer is a **benchmark-only track**. It is not on the live critical path for either demo and is the first module cut if the schedule slips. Any requirement here that threatens a Must-have requirement is cut, not rescheduled.

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-TMP-1 | The system shall extract clip-level features from a pretrained video encoder plus perception features (hand positions, part-in-hand), and cache them per video. | S5 | T |
| FR-TMP-2 | A temporal action segmentation model shall label each moment with a generic action class: `search`, `grasp`, `place`, `adjust`, `remove`, `idle`. | S5 | T |
| FR-TMP-3 | The model shall predict **generic verbs, not step ids**, so that it transfers between Jenga and jigsaw without retraining. Step identity comes from the world-state tracker. | S5 | I |
| FR-TMP-4 | The temporal model shall run asynchronously at ≈2 Hz and shall never block a video frame. | S5 | T, A |
| FR-TMP-5 | The temporal model shall supply the action signal during occlusion windows, and its events shall be consumable by the checker on the same interface as state events. | S6 | T |
| FR-TMP-6 | The system shall run the step recognizer standalone on a public benchmark that has no tracked grid. | S5 | A |

### 3.6 FR-CHK — Checker (plan §2.6, §4.4)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-CHK-1 | The checker shall maintain per-step status in {`pending`, `active`, `done`, `error`} and evaluate preconditions on every received event. *Built.* | M | T |
| FR-CHK-2 | The checker's native interface shall be the **constraint form**; DAG steps shall be lowered into it so both procedures execute identical engine code. *Built.* | M | I, T |
| FR-CHK-3 | The checker shall raise **missed step**: a required, non-optional step still `pending` at session end, or implied by a soft-dependency violation. | M | T, D |
| FR-CHK-4 | The checker shall raise **out of order**: a placement matching step *k* while a step it softly depends on is still `pending`. | M | T, D |
| FR-CHK-5 | The checker shall raise **wrong part**: a placed part (jigsaw piece, or LEGO brick) occupying a cell that belongs to a different part. | M | T, D |
| FR-CHK-6 | The checker shall raise **wrong position / rotation** (right part, wrong cell or turned wrong) and **extra / removed part**. | M | T, D |
| FR-CHK-7 | An unmet **hard** dependency, or an event the tracker cannot match, shall **never** produce a user-facing alert. It shall defer the decision, request state re-verification, and resolve as either a logged **perception miss** or a genuine alert. | M | T |
| FR-CHK-8 | Perception misses recorded by the re-verification path shall be logged separately and counted against perception recall in the evaluation, not against the user. | M | A, I |
| FR-CHK-9 | Events below a configurable confidence threshold shall produce a **soft (yellow) warning** rather than a red alert. | S2 | T, D |
| FR-CHK-10 | For Jenga the checker shall raise: **illegal source layer**, **layer started early**, **two-handed move**, **wrong replacement**, and **collapse** (terminal). Every rule shall live in the compiled spec, not in checker code. *Built.* | M | T, D |
| FR-CHK-11 | Every alert shall be attributable to one explicit rule or precondition id. No alert shall originate from a learned model's output alone. | M | I |
| FR-CHK-12 | The checker shall write a complete, timestamped event and alert log for every session. *Built.* | M | T |

### 3.7 FR-UI — Interface, streaming and explainer (plan §2.7, §4.5)

A camera window with overlays satisfies the **M** items below; the web UI is the first should-have.

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-UI-1 | The live view shall show the video with a lattice overlay (Jenga) or a grid overlay with piece boxes (jigsaw). | M | D |
| FR-UI-2 | The live view shall show a **checklist** that ticks steps off as they complete (jigsaw) or a **live legality panel** (Jenga). | M | D |
| FR-UI-3 | Alerts shall appear in the live view distinguishing red (error) from yellow (soft warning), with the human-readable rule text. | M | D |
| FR-UI-4 | The demo process shall read the cameras **locally in the Python process** and push only overlays and alerts over the WebSocket. | M | I, A |
| FR-UI-5 | A **ghost overlay** shall indicate the next expected piece. | S6 | D |
| FR-UI-6 | Session start, stop and procedure selection shall be operable from the UI without editing code or config files. | S1 | D |
| FR-UI-7 | A FastAPI + WebSocket server and web UI shall present FR-UI-1..3 in a browser. | S1 | D |
| FR-EXP-1 | At session end (or on `COLLAPSE`) the system shall pass the event log to the LLM and produce a readable session report citing timestamps, the step or move involved and the consequence. | S3 | D |
| FR-EXP-2 | The session report shall be generated from the event log only — never from video. | S3 | I |

### 3.8 FR-SYS — System-level behaviour

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-SYS-1 | The end-to-end path camera → perception → state → checker → alerts shall run as a single live session with no manual step between stages, for **both** Jenga and jigsaw. | M | D |
| FR-SYS-2 | Procedure, camera and model choices shall be set by YAML config; switching between Jenga and jigsaw shall require no code change. | S1 | I, D |
| FR-SYS-3 | A session shall recover from transient failure (dropped frames, a lost depth packet, momentary marker loss) without terminating. | S1 | T |
| FR-SYS-4 | The system shall run entirely offline except for the offline compile and post-session explain calls; a local LLM fallback shall be documented. | S1 | I |
| FR-SYS-5 | A session shall be replayable from a recorded log or a scripted scene through the same tracker and checker code a live camera uses. *Built (`replay.py`).* | M | T |

---

## 4. Non-functional requirements

### 4.1 NFR-PERF — Performance and latency (plan §3.1)

Budget is measured on the demo laptop in week 6 and re-measured in week 11. The Jenga path is the budgeted one; jigsaw matching runs on hand-exit only and has its own bound.

| ID | Requirement | Target | Pri | Verify |
| --- | --- | --- | --- | --- |
| NFR-PERF-1 | End-to-end live frame rate | ≥ 15 FPS sustained | M | A |
| NFR-PERF-2 | Per-frame critical path total (Jenga) | ≤ 40 ms (66 ms available) | M | A |
| NFR-PERF-3 | Jenga slot reading, two cameras, per frame (replaces the parked LEGO detector budget) | ≤ 15 ms | M | A |
| NFR-PERF-4 | Hand keypoints, per frame | 5–10 ms | M | A |
| NFR-PERF-5 | Tracker, per frame | < 2 ms | M | A |
| NFR-PERF-6 | Homography refresh, amortised | < 1 ms | M | A |
| NFR-PERF-7 | State update + diff, per hand-exit | < 1 ms | M | A |
| NFR-PERF-8 | Checker, per event | < 1 ms | M | A |
| NFR-PERF-9 | Temporal model, async, off critical path | ≤ 200 ms per 0.5 s window | S5 | A |
| NFR-PERF-10 | WebSocket overlay push | ≤ 10 ms per frame | S1 | A |
| NFR-PERF-11 | Median delay from error occurrence to alert shown | < 2 s | M | A |
| NFR-PERF-12 | Frame + depth decode (Record3D), per frame | ≤ 8 ms | M | A |
| NFR-PERF-13 | Jigsaw piece matching, per hand-exit | ≤ 100 ms | M | A |

### 4.2 NFR-* — Other quality attributes

| ID | Attribute | Requirement | Pri | Verify |
| --- | --- | --- | --- | --- |
| NFR-REL-1 | Reliability | A full jigsaw run and a full Jenga game shall each complete without a crash or a manual restart in ≥ 9 of 10 consecutive runs. | M | A |
| NFR-REL-2 | Quietness | ≤ 1.0 false alerts per fully correct jigsaw run or Jenga game. | M | A |
| NFR-USE-1 | Usability | A person who has not used the system before shall complete a guided jigsaw run using only on-screen information. | S1 | D |
| NFR-USE-2 | Usability | Alert text shall name the step or move and the consequence in plain language, not a rule id alone. | S1 | I |
| NFR-MNT-1 | Maintainability | Adding a new procedure of an existing state model shall require only a new source text and a compiled spec — no code change. | M | D |
| NFR-MNT-2 | Maintainability | Adding a new **state model** shall require implementing the `WorldState` interface and a rule policy only; checker engine and compiler code shall be untouched. | M | I |
| NFR-PORT-1 | Portability | Training (step recognizer) shall be runnable on a free-tier cloud GPU via precomputed features and small model variants. | S5 | D |
| NFR-PORT-2 | Portability | Inference shall run on a single consumer machine; a documented CPU-only degraded mode is a stretch goal. | M / C | A |
| NFR-COST-1 | Cost | Total LLM API spend shall stay negligible (≈20 calls across the project); a local model fallback shall be documented. | M | A |
| NFR-PRIV-1 | Privacy | Recorded subjects shall give written consent covering research use and, separately, public release. Frame the workspace, not the person. | M | I |
| NFR-PRIV-2 | Privacy | Secrets (LLM API keys) shall live in environment variables or an untracked `.env`; no key shall be committed. | M | I, T |
| NFR-REP-1 | Reproducibility | Every reported number shall be regenerable from a committed script, a fixed data split and a recorded config. | M | I |
| NFR-REP-2 | Reproducibility | Dataset splits shall be fixed in week 8 and never adjusted after a result is reported from them. | M | I |
| NFR-TRC-1 | Traceability | Every experiment run shall be logged to the experiment tracker with its config hash and git commit. | S | I |
| NFR-DOC-1 | Documentation | The README, plan and requirements shall describe the current direction; a change of direction shall update `docs/` in the same week. | M | I |

---

## 5. Data requirements (plan §5)

### 5.1 Public datasets

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| DR-PUB-1 | Access, licence and storage size for every candidate public dataset shall be **verified** — the plan's descriptions are from memory and are unverified. | S5 | I |
| DR-PUB-2 | One public procedural-error dataset with mistake labels shall be secured as the primary benchmark for the step recognizer. | S5 | I |
| DR-PUB-3 | At least one alternative benchmark shall be identified and kept viable in case access is denied or slow. | S5 | I |
| DR-PUB-4 | Any third benchmark is a stretch item and shall not be scheduled. | C | — |

### 5.2 ThirdEye-LEGO (own recordings) — parked

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| DR-LEG-1 … DR-LEG-9 | v1.0 LEGO data requirements (Model A and B designs, brick vocabulary, 40–60 + 15–20 runs, planted errors, labels, 30 h budget). **Parked**: no LEGO recording or labelling is scheduled. The scripted replay scenarios (`scripts/scenarios/model_a_*.yaml`) remain as tests. | P | — |

### 5.3 ThirdEye-JENGA (own recordings)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| DR-JEN-1 | 15–20 games from 3–4 people, recorded as colour + depth from the iPhone and colour from a second camera at the opposite corner. | M | I |
| DR-JEN-2 | Planted violations shall cover: pull from the top complete layer; start a new layer early; two-handed pull; return a block to an illegal slot. Collapses occur naturally and need no planting. | M | I |
| DR-JEN-3 | Labels: move start/end, source and destination slot, violation type, and the lattice state after each move. Budget **10 person-hours**. | M | A |
| DR-JEN-4 | If the fallback slot classifier is needed (FR-DEP-5): ~150 labelled slot-crop frames against a dark backdrop. | C | I |

### 5.3a ThirdEye-JIGSAW (own recordings)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| DR-JIG-1 | One development puzzle (about 12–24 pieces, so a run is short) and one held-out puzzle, each with its box picture. Pieces shall differ visibly; avoid large uniform regions. | M | I |
| DR-JIG-2 | 20–30 runs from 3–4 people, roughly half containing planted errors, from the overhead view. | M | I |
| DR-JIG-3 | Planted errors shall cover: skip a piece; interior piece before the border; piece in the wrong cell; piece turned 90/180/270°; piece removed after placement. | M | I |
| DR-JIG-4 | Labels: per placement, the cell, rotation and time; error type per run. Budget **10 person-hours**. | M | A |
| DR-JIG-5 | The held-out puzzle shall be used only for testing, never for tuning the matcher. | S4 | I |

### 5.4 Compiler eval corpus

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| DR-CRP-1 | **15–20 short written sources** for procedures that are never physically recorded, each with a hand-authored ground-truth spec. *Not started.* | M | I |
| DR-CRP-2 | At least 3–4 corpus entries shall be rulebook-style sources (constraint form) and several shall be jigsaw-style instruction sheets (dependency form). | M | I |
| DR-CRP-3 | Compiler errors shall be reported under an explicit taxonomy: wrong pose, missed dependency, hallucinated step, wrong constraint. | M | A |
| DR-CRP-4 | This corpus is the one result that survives independently of the vision stack, and shall be completed even if perception slips. | M | I |

### 5.5 Release

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| DR-REL-1 | The compiler eval corpus is text-only and shall be released unconditionally with the code. | M | I |
| DR-REL-2 | Video datasets shall be released only with the consent of everyone recorded (NFR-PRIV-1). | S | I |

---

## 6. Hardware requirements (plan §3.1, §2.2)

Procurement checklist. Items marked M block a must-have demo.

| ID | Item | Spec | Qty | Pri |
| --- | --- | --- | --- | --- |
| HW-1 | Overhead camera | 1080p webcam with a stable USB/IP feed, for the jigsaw | 1 | M |
| HW-2 | Overhead camera stand / rig | Rigid, repeatable position; must not drift during a session | 1 | M |
| HW-3 | ~~LEGO baseplate~~ | Parked with LEGO | — | P |
| HW-4 | Printed ArUco markers | 4, mounted at the jigsaw board corners, rigid and non-glossy; plus a set around the Jenga base | 2 sets | M |
| HW-5 | ~~LEGO brick set~~ | Parked with LEGO | — | P |
| HW-6 | Plain mat and even lighting | Matte, non-reflective; consistent illumination across the board | 1 | M |
| HW-7 | Laptop | Runs the live pipeline within the §4.1 budget; GPU recommended | 1 | M |
| HW-8 | ~~LEGO side camera~~ | Parked with LEGO | — | P |
| HW-9 | **iPhone Pro with LiDAR** | Streams colour + depth (Record3D) from one corner of the Jenga tower, on a stand | 1 | M |
| HW-10 | Jenga set | Standard 54 blocks, 75 × 25 × 15 mm | 1 | M |
| HW-11 | Dark backdrop | High contrast behind the Jenga tower | 1 | M |
| HW-12 | Table markers for Jenga | Around the tower base, for calibration and scale | 1 set | M |
| HW-13 | **Second camera** | Webcam or phone at the opposite corner of the Jenga tower, on a stand | 1 | M |
| HW-14 | Jigsaw puzzles | One development puzzle and one held-out puzzle, with box pictures | 2 | M |
| HW-15 | Jigsaw board | A flat, matte surface sized for the puzzle, with the HW-4 markers | 1 | M |

| ID | Constraint | Pri |
| --- | --- | --- |
| HW-C1 | A minimum of **2 cameras** is required for Jenga (the iPhone and one more); the jigsaw rig adds a third device. The achievable mount and sync count is an open item (§13). | M |
| HW-C2 | All Jenga cameras shall be time-aligned well enough that a single move is attributable to one timestamp within one frame interval (FR-CAL-10). | M |
| HW-C3 | Camera positions shall be reproducible between sessions, or recalibration shall be part of the documented session startup. | M |

---

## 7. Software and environment requirements (plan §3)

### 7.1 Module → technology commitments

| ID | Module | Primary | Fallback | Trained? |
| --- | --- | --- | --- | --- |
| SW-1 | Calibration | OpenCV ArUco + homography | Manual 4-corner click | No |
| SW-2 | Hands | MediaPipe Hands | MMPose hand models | No |
| SW-3 | Part detection | Jenga: depth thresholds + colour cue. Jigsaw: matching against the box picture. (The LEGO YOLO detector is parked.) | Jenga: small slot-crop classifier | Jenga fallback only |
| SW-4 | Tracking | ByteTrack | SORT | No |
| SW-5 | World state | Own lattice / grid modules (NumPy) | — | No |
| SW-6 | Video features | VideoMAE or I3D, precomputed | DINOv2 per-frame | No |
| SW-7 | Step segmentation | MS-TCN++ | ASFormer | **Yes** — benchmark track |
| SW-8 | LLM (compiler, explainer) | Claude via API | Local Qwen or Llama via Ollama | No — prompt + JSON schema |
| SW-9 | Checker | Plain Python engine | — | No |
| SW-10 | Backend | FastAPI + WebSocket streaming | — | — |
| SW-11 | Frontend | React | Streamlit for early versions | — |
| SW-12 | Labelling | CVAT (step segments, slot states) | Label Studio | — |
| SW-13 | Experiment tracking | Weights & Biases | TensorBoard | — |
| SW-14 | Colour + depth input | Record3D stream | Recorded RGB-D files | No |

### 7.2 Environment requirements

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| SW-E1 | The project shall pin a reproducible Python environment (lockfile committed); the versions in §7.3 are indicative floors, not the final pins. | M | I |
| SW-E2 | Python 3.11+ with PyTorch and CUDA matched to the available GPU driver, where a GPU is used. | M | T |
| SW-E3 | `pip install` (or the chosen lock tool's equivalent) on a clean machine shall bring up a working inference environment. Verified once per semester on a second machine. | M | T |
| SW-E4 | Training-only and live-camera dependencies shall live in separate extras groups so the test machine installs the minimum. *Built (`pyproject.toml`).* | S | I |
| SW-E5 | The repository shall follow the directory layout of plan §4.1. | M | I |
| SW-E6 | Unit tests shall exist for the Jenga lattice, the piece grid, the state diff and every checker rule. | M | T |
| SW-E7 | Data, weights and videos shall not be committed to git; a documented storage location and a fetch script shall be provided instead. | M | I |

### 7.3 Dependency groups (indicative; pin properly)

The groups below match `pyproject.toml`: core, `live`, `llm`, `train` and `dev`.

**Core — tests and replay**

```
numpy
opencv-contrib-python        # ArUco lives in contrib
pydantic                     # procedure-spec schemas
pyyaml                       # scenario scripts and configs
```

**`live` — camera path and server**

```
torch
ultralytics                  # tracking (and the parked YOLO detector)
mediapipe                    # hand keypoints
fastapi
uvicorn[standard]            # WebSocket streaming
```

Record3D access and any RGB-D reader are not yet chosen (OQ-8).

**`llm` — offline compile and explain**

```
anthropic
python-dotenv                # API key, never committed
```

**`train` — step recognizer and evaluation**

```
torchvision
transformers                 # VideoMAE features
scikit-learn
pandas
matplotlib
wandb
tqdm
```

**`dev`**

```
pytest
ruff
mypy
```

**Frontend:** Node 20+, React, a WebSocket client. Streamlit is acceptable for internal views only.

**External services:** an LLM API key (≈20 calls total) with Ollama + a local model as the offline fallback; a labelling server (CVAT) reachable by the whole team.

---

## 8. Interface and contract requirements (plan §4.2–§4.4)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| IF-SPEC-1 | Both spec forms shall be defined as Pydantic models and validated on load. A spec that does not validate shall not run. *Built.* | M | T |
| IF-SPEC-2 | The DAG form shall carry: `procedure`, `kind: "dag"`, `state_model` (`stud_grid` or `piece_grid`), the grid geometry, and a `steps` array as specified in FR-CMP-2. | M | T |
| IF-SPEC-3 | The constraint form shall carry: `procedure`, `kind: "constraints"`, `state_model`, geometry (e.g. `tower: {layers, slots_per_layer}`), an `actions` array with `preconditions` and `effects`, and a `terminal` array. *Built.* | M | T |
| IF-SPEC-4 | Every precondition shall carry a machine `expr`, a human `text` and a `violation` code; alerts shall surface the `text` and log the `violation`. *Built.* | M | T |
| IF-SPEC-5 | Specs shall be versioned; a spec file shall record the compiler version and model that produced it. *Built (provenance).* | S | I |
| IF-EVT-1 | Events shall conform to a single schema: `t` (seconds since session start), `source` (`state` \| `temporal`), `kind`, optional `part`, optional `pose`, and `confidence`. | M | T |
| IF-EVT-2 | The checker shall accept events from the state tracker and the temporal model through the same entry point, distinguished only by the `source` field. | M | T |
| IF-STA-1 | Every state model shall implement `current()` and `diff()` and nothing procedure-specific shall leak past that boundary. | M | I, T |
| IF-WS-1 | The WebSocket shall carry overlays, checklist/legality status and alerts **outbound only** on the demo path; video frames shall not travel inbound during a demo. | S1 | I, A |
| IF-WS-2 | The WebSocket message schema shall be documented and versioned so the UI and backend can be developed independently. | S1 | I |
| IF-LOG-1 | The session log shall be a machine-readable file sufficient to regenerate the alert timeline and the LLM report without the video. *Built.* | M | T |

---

## 9. Evaluation acceptance criteria (plan §7)

These are the pass/fail gates for week 11. They are fixed **now**, before results exist.

| ID | Question | Metric | Target | Pri |
| --- | --- | --- | --- | --- |
| EV-DET-1 | ~~Are parts detected where they are? (LEGO, centroid error in studs)~~ | — | Parked with the LEGO detector | P |
| EV-DET-2 | ~~p95 centroid error~~ | — | Parked | P |
| EV-DET-3 | ~~mAP@0.5~~ | — | Parked | P |
| EV-STA-1 | Is the jigsaw state right? | Placement accuracy, piece + cell + rotation exact, per step | ≥ 90% | M |
| EV-STA-2 | ~~LEGO Model B placement accuracy~~ | — | Parked | P |
| EV-STA-3 | Is the Jenga lattice right? | Slot occupancy accuracy vs hand-labelled state | ≥ 95% | M |
| EV-STA-4 | Are jigsaw pieces identified? | Cell and rotation correct among identified pieces; unidentified rate reported | ≥ 90% correct; unidentified ≤ 10% | M |
| EV-SEG-1 | Are steps recognized? | Frame accuracy, Edit score, F1@{10, 25, 50} | Within reach of published numbers on the chosen benchmark | S5 |
| EV-ERR-1 | Are jigsaw errors caught? | Precision at fixed recall, **per error type** | Precision ≥ 0.8 at recall ≥ 0.8 | M |
| EV-ERR-2 | ~~LEGO Model B error detection~~ | — | Parked | P |
| EV-ERR-3 | Is it quiet when nothing is wrong? | False alerts per fully correct run, **each** procedure | ≤ 1.0 | M |
| EV-ERR-4 | How early? | Median delay from error to alert | < 2 s | M |
| EV-LAT-1 | Is it usable live? | Per-stage latency vs the §4.1 budget | ≥ 15 FPS, per-frame ≤ 40 ms | M |
| EV-CMP-1 | Is the compiler right? | Step accuracy over the eval corpus (n = 15–20) | ≥ 0.9 | M |
| EV-CMP-2 | " | Dependency accuracy | ≥ 0.8 | M |
| EV-CMP-3 | " | Pose and constraint accuracy | Reported with the §5.4 error taxonomy | M |
| EV-JEN-1 | Do Jenga rules hold? | Violation precision/recall **per rule** | Precision ≥ 0.8 at recall ≥ 0.8 | M |
| EV-JEN-2 | " | Collapse detection | Reported | M |

### 9.1 Required baselines

| ID | Baseline | Pri |
| --- | --- | --- |
| EV-BL-1 | **Temporal model only** — no world-state tracker; errors decided from recognized actions alone. | S5 |
| EV-BL-2 | **End-to-end VLM** — sampled frames plus the rulebook or instructions, asked to list errors. Reported **as measured**; no outcome is predicted in advance, and a split result (VLM better on missed step, worse on exact slot and timing) is a legitimate and interesting finding. | S3 |
| EV-BL-3 | **Published results** from the benchmark dataset's own paper. | S5 |

### 9.2 Required ablations

| ID | Ablation | Pri |
| --- | --- | --- |
| EV-AB-1 | Without the world-state tracker (actions only). | S5 |
| EV-AB-2 | Without the temporal model (state only). | S5 |
| EV-AB-3 | **Occlusion-window slice** — errors that begin and end inside an occlusion window, state-only vs fused. The only place the temporal model is expected to win. | S6 |
| EV-AB-4 | LLM-generated spec vs hand-written spec. | M |
| EV-AB-5 | ~~Zero-shot vs fine-tuned detector~~ — replaced by EV-AB-9 and EV-AB-10. | Superseded |
| EV-AB-6 | With vs without soft warnings, measured on false-alarm rate. | S2 |
| EV-AB-7 | ~~Top-down vs top-down + side camera on Model B height~~ — replaced by EV-AB-9. | Superseded |
| EV-AB-8 | Tier 1 vs tier 2 vs tier 3 generality (plan §1.2). | S3 |
| EV-AB-9 | Jenga: colour only vs colour + depth; one camera vs two cameras merged. | S1 |
| EV-AB-10 | Jigsaw: matching with and without rotation search; effect of the unidentified-wait rule on false alerts. | S1 |

### 9.3 Generality ladder — reporting requirement

| ID | Requirement | Pri |
| --- | --- | --- |
| EV-GEN-1 | **Tier 1** — Jenga *game*: a procedure of constraints, not order, through the same checker interface. Supports: the spec format itself is general. **This is the claim worth making.** | M |
| EV-GEN-2 | **Tier 2** — jigsaw: a new state model and a new identification method (matching, not training), same checker and same spec lowering. Supports: adding a procedure means writing a state tracker. | M |
| EV-GEN-3 | **Tier 3** — an unseen source compiled with no code change (a second jigsaw and a second rulebook). Supports: the compiler and checker generalise across instances. | S3 |
| EV-GEN-4 | The report shall state the ladder honestly, shall not present one tier alone as the generality result, and shall describe LEGO as a prior result and not a rung. | M |

---

## 10. Milestone gates (plan §8)

Weeks count from the week of 2026-09-14. MG-1 to MG-4 are met; MG-5 to MG-14 were reset in v2.0 for the Jenga + jigsaw schedule (CC-5).

| ID | Week | Gate — the requirement is met when… | Pri |
| --- | --- | --- | --- |
| MG-1 | 1–2 | Plan and requirements written; spec schema (both forms) committed; compiler v1 + validator running. **Met.** | M |
| MG-2 | 2 | Checker engine and Jenga rule set running against scripted scenarios. **Met.** | M |
| MG-3 | 3–4 | Jenga lattice and collapse detection; calibration and hand tracking; session log and replay. **Met** (229 tests). | M |
| MG-4 | 4 | LEGO state tracker and live camera path working. **Met, then parked.** | P |
| MG-5 | 5 | Plan and requirements rewritten; Jenga rulebook written; Record3D input spike; jigsaw schema extension committed. | M |
| MG-6 | 6 | **Jenga end-to-end v0**: RGB-D in → lattice → checker → alerts from one corner; latency table measured (NFR-PERF-*); jigsaw board calibrated. | M |
| MG-7 | 7 | Diagonal-corner views with the second camera merged (FR-DEP-2); jigsaw piece-grid tracker; Jenga recording started. | M |
| MG-8 | 8 | Jigsaw piece matching; both sources compiled; eval corpus committed; Jenga labels exported; **splits frozen** (NFR-REP-2). | M |
| MG-9 | 9 | **Jigsaw end-to-end v0**; Jenga violation P/R measured on recorded games. | M |
| MG-10 | 10 | Jigsaw runs recorded and labelled; soft warnings and false-alert target met; server running. | S2 |
| MG-11 | 11 | Full evaluation complete: both procedures, compiler corpus, baselines, ablations, latency. | M |
| MG-12 | 12 | UI polished (checklist, legality panel, alerts); LLM report; someone outside the team can operate it; demo video recorded. | S1 |
| MG-13 | 13 | **Reserved slack — not allocated to features.** Absorbs overruns from any earlier week. | — |
| MG-14 | 14 | Report, poster, demo video and final presentation submitted. | M |

**Gate rule:** the cut order of plan §8 applies — step recognizer, then occlusion fusion, then the LLM report, then the second jigsaw. Jenga and the first jigsaw are not cut.

---

## 11. Deliverable requirements (plan §11)

| ID | Deliverable | Requirement | Pri |
| --- | --- | --- | --- |
| DL-1 | Code | Public repository with setup instructions and evaluation scripts. | M |
| DL-2 | Dataset | ThirdEye-JENGA and ThirdEye-JIGSAW videos, specs and labels, released subject to DR-REL-2. Compiler eval corpus released unconditionally. | S |
| DL-3 | Report | Written as a short paper: problem, related work, method, results, ablations, the generality ladder, limitations. | M |
| DL-4 | Demo video | 2–3 min: a Jenga game with violations called in real time; a jigsaw with the checklist ticking and each error type caught; both procedures compiled from their text alone. | M |
| DL-5 | Live demo | A Jenga tower with the iPhone and a second camera, and a jigsaw board under the webcam. A panel member plays Jenga and tries to cheat, then builds the jigsaw and tries to skip or swap pieces. | M |
| DL-6 | Publication | Submission to a student research track or a workshop on egocentric/procedural video. | C |

---

## 12. Constraints, assumptions and exclusions

### 12.1 Constraints

| ID | Constraint |
| --- | --- |
| CON-1 | 14 calendar weeks, with week 13 reserved as slack and not available for features. |
| CON-2 | Team of 3 or 4 (open item OQ-1). With 3, the Perception and App roles merge. |
| CON-3 | One consumer machine, or free-tier cloud GPUs. All model choices must respect this. |
| CON-4 | Negligible budget: a Jenga set, puzzles, cameras, stands, printed markers, and ≈20 LLM API calls. The iPhone with LiDAR is assumed to be available. |
| CON-5 | Fixed cameras only. No head-mounted or moving camera for our own recordings. |
| CON-6 | All visual judgment comes from CV models; every alert comes from an explicit rule. The LLM is text-in, text-out. |

### 12.2 Assumptions

| ID | Assumption | If false |
| --- | --- | --- |
| ASM-1 | Lighting is even and the board is matte during recording and demo. | Segmentation and matching degrade; add lighting control or a contrast mat. |
| ASM-2 | One person works in the workspace at a time. | Out of scope (§12.3); state attribution breaks. |
| ASM-3 | Public dataset access is granted within a few weeks. | Fall back to an alternative benchmark (DR-PUB-3); own data alone still carries the demos. |
| ASM-4 | Two cameras at opposite corners of the tower, each seeing two faces, give enough observability of all 54 Jenga slots, using depth. *Replaces v1.0's two orthogonal straight-on views. The current lattice code still assumes straight-on views and must be extended (plan §2.4).* | Some slots become unobserved (FR-DEP-2); tier-1 result at risk. Fall back to straight-on views. |
| ASM-5 | Standard 75 × 25 × 15 mm Jenga blocks provide adequate scale reference. | Add an explicit scale marker to the rig. |
| ASM-6 | Record3D delivers colour + depth with latency and drop-outs small enough for 15 FPS, and can be time-aligned with the second camera. | Develop on recorded RGB-D; reduce to one corner for the live demo. |
| ASM-7 | A box picture and a puzzle with visibly distinct pieces make matching reliable without training. | Choose a different puzzle; report the unidentified rate honestly. |
| ASM-8 | The jigsaw's instruction sheet is a fixed order (corners → edges → interior). | A looser "any border-first order" would need a constraint-form jigsaw spec (open question). |

### 12.3 Explicitly out of scope

| ID | Excluded | Pri |
| --- | --- | --- |
| OOS-1 | Jigsaw pieces that are stacked, tilted or off the board plane. | W |
| OOS-2 | Multiple people working simultaneously. | W |
| OOS-3 | Head-mounted (moving) cameras for our own data. | W |
| OOS-4 | Jenga stability or lean estimation — legality is checked, physics is not. | W |
| OOS-5 | VLM reading picture-only official LEGO instruction pages — a separate research problem with its own failure modes. Revisit only if week 13 is clear. | C |
| OOS-6 | New LEGO work of any kind (detector, side camera, Model B, recordings). | P |

### 12.4 Stretch items

| ID | Item | Pri |
| --- | --- | --- |
| STR-1 | Synthetic training data (only relevant if the Jenga slot classifier fallback is used). | C |
| STR-2 | Laptop-CPU or edge deployment. | C |
| STR-3 | Voice alerts. | C |

---

## 13. Open items blocking requirement finalisation (plan §12)

| ID | Open question | Blocks | Due |
| --- | --- | --- | --- |
| OQ-1 | Team size: 3 or 4? | CON-2, role assignment | Week 5 |
| OQ-2 | ~~Which bricks do we already have?~~ **Closed** — LEGO is parked, no bricks needed. | HW-5 | Closed |
| OQ-3 | ~~Final designs for Model A and Model B.~~ **Closed** — parked with LEGO. | DR-LEG-1, DR-LEG-2 | Closed |
| OQ-4 | What GPU is available: local card, college lab, or Colab/Kaggle only? | NFR-PORT-1, NFR-PORT-2, SW-E2 | Week 5 |
| OQ-5 | Course review dates, to line up the timeline (§10). | all MG | Week 5 |
| OQ-6 | How many cameras can we actually mount and synchronise? Jenga needs the iPhone plus one more; the jigsaw adds an overhead webcam. | HW-C1, HW-C2, FR-CAL-8, FR-CAL-10 | Week 5 |
| OQ-7 | ~~Is an LLM API allowed or budgeted?~~ **Closed** — ≈20 calls, cost negligible, Ollama fallback exists. | NFR-COST-1, SW-8 | Closed |
| OQ-8 | Which Record3D access route (USB or Wi-Fi) and which RGB-D reader library? | FR-CAL-9, SW-14, NFR-PERF-12 | Week 6 |
| OQ-9 | Which jigsaw puzzles (piece count, distinctive pieces), and which is held out? | DR-JIG-1, HW-14, DR-JIG-5 | Week 6 |
| OQ-10 | Is the jigsaw one fixed instruction sheet, or also a looser border-first variant (constraint form)? | ASM-8, FR-CMP-12 | Week 6 |

---

## 14. Traceability

### 14.1 Requirement class → plan section

| Requirement class | Plan section |
| --- | --- |
| FR-CMP | §2.1, §4.2, §4.3 |
| FR-CAL | §2.2 |
| FR-PER, FR-DEP, FR-JIG | §2.3, §4.6 |
| FR-STA | §2.4 |
| FR-TMP | §2.5, §4.7 |
| FR-CHK | §2.6, §4.4 |
| FR-UI, FR-EXP | §2.7, §4.5 |
| NFR-PERF | §3.1 |
| DR-* | §5 |
| HW-* | §3.1, §2.2 |
| SW-* | §3, §4.1 |
| IF-* | §4.2, §4.3, §4.4 |
| EV-* | §7 |
| MG-* | §8 |
| DL-* | §11 |
| CON, ASM, OOS, STR | §6, §10 |
| OQ-* | §12 |

### 14.2 Risk → mitigating requirement (plan §10)

| Risk | Likelihood | Mitigated by |
| --- | --- | --- |
| Diagonal-corner views make slot reading harder than straight-on | **High** | FR-DEP-2 (unobserved, not guessed), FR-DEP-3, ASM-4 (fallback), EV-AB-9 |
| iPhone depth stream unreliable | Medium | FR-CAL-9 (recorded RGB-D), FR-SYS-3, ASM-6 |
| Depth too noisy to see a pushed-out block | Medium | FR-DEP-4 is S2; no rule depends on it |
| Jigsaw pieces confused by the matcher | High | FR-JIG-4 (unidentified, never alerted), DR-JIG-1, EV-STA-4, EV-AB-10 |
| Piece segmentation fails | Medium | FR-JIG-2, FR-CAL-4, HW-6 |
| Hands block the view during placement | High | FR-PER-6 (freeze), FR-STA-3 (stability filter), FR-TMP-5, EV-AB-3 |
| Too many false alarms | Medium | FR-CHK-7, FR-CHK-9, NFR-REL-2 and EV-ERR-3 |
| Step recognition too inaccurate | Medium | FR-TMP scope note (benchmark-only, cut first) |
| Dataset access slow or denied | Medium | DR-PUB-1, DR-PUB-3 |
| Scope creep: two procedures, two rigs, a UI and a benchmark | **High** | §10 gate rule (cut order), priority scheme (§1.5) |
| LLM specs wrong | Low | FR-CMP-6, FR-CMP-7, EV-CMP-* |
| Jenga tower collapses mid-run | Certain | FR-STA-8 (terminal event) — a demo highlight, not a failure |
| Docs drift from the code | Medium | NFR-DOC-1, CC-4 |

### 14.3 Must-have set (the minimum viable project)

The project passes its minimum bar when every **M** requirement is verified. In plan terms (§6): the Jenga rulebook and a jigsaw instruction sheet compile to validated specs; the iPhone depth stream and the second camera fill the tower lattice; the overhead camera and box-picture matching fill the piece grid; the checker catches the Jenga fouls live and the jigsaw's out-of-order, missed, wrong-part and wrong-position/rotation errors live; the live path runs at 15+ FPS within the latency budget; and compiler accuracy is measured on the eval corpus.

---

## 15. Change control

| ID | Requirement |
| --- | --- |
| CC-1 | Requirement IDs are stable. A requirement that is dropped is marked **superseded** or **parked** with a reason; it is never deleted or renumbered. IDs cited in the code keep their meaning. |
| CC-2 | Any change to an **EV-** target after week 11 results exist must be recorded as a change with a date and a rationale, and the original target reported alongside. |
| CC-3 | Promotion of any **S** or **C** item to **M** requires the must-have set to be green first. |
| CC-4 | This document is regenerated from the plan whenever the plan changes materially; the plan is the source of truth. The PDFs are regenerated from the markdown after every change (`docs/README.md`). |
| CC-5 | **v2.0 re-baseline (2026-10-12).** The milestone gates MG-5 to MG-14 keep their ID and week number but their content was reset for the Jenga + jigsaw schedule; this is the one allowed exception to CC-1. v1.0's text is in git history. |

---

*ThirdEye Requirements Specification v2.0 — 2026-10-12. Derived from `ThirdEye_Project_Plan.md`. Supersedes v1.0 (2026-09-18).*
