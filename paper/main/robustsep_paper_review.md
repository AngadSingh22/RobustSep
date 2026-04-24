# RobustSep Paper Review: Verifiability, Scientific Framing, and Diagram Plan

Source reviewed: `paper/main/main.tex`  
Review focus: academic ML paper readiness, internal consistency, verifiability, and figures that communicate the system.

## Executive Verdict

The paper has a strong systems idea: it frames RGB-to-CMYK/CMYKOGV separation as a constrained, stochastic, drift-aware decision problem rather than a single deterministic color transform. The architecture is coherent and paper-worthy if presented as a reproducible robust-separation pipeline with explicit press priors, feasibility projection, candidate generation, surrogate scoring, and tail-risk selection.

The current draft is not yet submission-ready because the trust layer is incomplete. The Evaluation section still contains placeholders, some claims are phrased as completed results while the abstract says evaluation is planned, and several citations are being used as placeholders for unrelated claims. The biggest improvement is to make the paper verifiable: every dataset, split, manifest, PPP, seed, target-generation step, and evaluation report should be hash-linked and described as part of the experimental protocol.

## Highest-Priority Changes

1. **Resolve proposal-vs-result language**

   The abstract says evaluation is planned, while the Evaluation section says "We evaluate" and "All experiments are conducted using pretrained models." Pick one stance:
   - If results are not final, frame this as a system/protocol paper and remove result-style claims.
   - If results are now available, replace all placeholders with measured values, split sizes, confidence intervals, and report hashes.

   A paper with empty tables and future-tense evaluation will read as unfinished.

2. **Make the dataset section concrete**

   Replace "1,000 RGB images sampled from \_\_\_" with a reproducible dataset paragraph. At minimum, report:
   - dataset families: RobustSep staged corpus, DocLayNet/layout data, SKU-110K/product photos, RGB cube/color ramps;
   - split policy and seed;
   - patch size, stride/grid policy, alpha policy, corrupt image exclusions;
   - train/val/test patch or surrogate-example counts;
   - manifest paths and hash policy.

   Suggested wording direction: "We evaluate on the held-out test split emitted by the v1.1 split manifest, with all examples addressable through manifest records containing source path, patch coordinates, target hash, PPP hash, drift hash, and shard hash."

3. **Add a reproducibility subsection**

   Add `\subsection{Reproducibility and Artifact Integrity}` under Evaluation. This is essential because RobustSep's strongest claim is not just accuracy; it is auditability under stochastic generation.

   Include:
   - exact PPP base family and overrides;
   - deterministic root seed;
   - drift sample count and finite quantile definition;
   - manifest hash verification;
   - target hash and shard hash verification;
   - software commit hash;
   - GPU/model checkpoint hash;
   - whether surrogate metrics came from train, val, or test gates.

4. **Define the feasible set fully**

   The current refinement section only shows clipping and TAC scaling. The implemented/architectural feasible set is richer and should be formalized:
   - per-channel caps;
   - TAC cap;
   - OGV total cap;
   - pair caps: C+O, M+G, Y+V;
   - neutral/dark OGV cap based on Lab chroma and L\*;
   - projection after every solver/refiner step.

   The paper should define `K(PPP)` once, then state that `\Pi_K` is the deterministic projection operator enforcing all components.

5. **Fix citation mismatches**

   Several citations appear to support unrelated claims:
   - `\cite{icc}` is used for a CMAC neural-network paper, but the bibliography entry is the ICC profile standard.
   - `\cite{kingma}` is used for Zhang/polynomial neural-network work and for surrogate context windows, but Kingma is the VAE paper.
   - `\cite{marcu}` is used to justify PPP externalization and ICC rendering details; this needs either a more precise citation or a wording change.
   - The bibliography entry title "RGB-YMCK" should be verified; if it is a typo, fix to CMYK.

   These are reviewer-visible problems. They weaken trust even if the method is sound.

6. **Clarify intent maps**

   The text says brand pixels cover logos, wordmarks, and spot-color regions, but the method derives intent from local statistics and edge analysis with no learned components. A local heuristic cannot reliably know something is a brand mark unless metadata or annotation is provided.

   Safer framing:
   - "brand-critical proxy" or "high-salience flat/edge regions";
   - reserve "brand" for annotated/metadata-driven cases;
   - explicitly state the fallback and priority order.

7. **State the teacher/surrogate truth carefully**

   The paper says the surrogate is trained entirely under ICC teacher supervision with nominal and drifted ICC targets. If the current implementation uses a calibrated teacher proxy for CMYKOGV appearance in some paths, state that precisely. Do not imply measured press Lab supervision unless actual measurements exist.

   Suggested distinction:
   - "ICC/proxy-supervised surrogate" for current experiments;
   - "measured press supervision" as future work, if not used.

8. **Strengthen evaluation metrics around the actual claims**

   The metrics list includes MSE, PSNR, and SSIM, but the paper's core claim is robust print feasibility. Put primary metrics first:
   - mean, median, q75, q90, q95 DeltaE00 under drift;
   - percentage of patches exceeding DeltaE00 thresholds;
   - TAC violation rate;
   - channel-cap, OGV-cap, neutral/dark-cap, and pair-cap violation counts;
   - surrogate candidate ranking: Spearman and top-1 agreement;
   - escalation rate and fallback rate;
   - runtime per megapixel or per patch.

   PSNR and SSIM can remain secondary, but they should not carry the paper.

## Verifiability Plan to Add to the Paper

Add a table named "Artifact and Protocol Traceability" with columns:

| Artifact                 | What It Verifies                      | Required Field                                                      |
| ------------------------ | ------------------------------------- | ------------------------------------------------------------------- |
| Split manifest           | Dataset membership and split identity | `split_manifest_version`, family list, alpha policy, source weights |
| Surrogate shard manifest | Training/eval examples                | total examples, shard count, SHA256 per shard                       |
| Target records           | Deterministic target generation       | `target_hash`, `initial_hash`, `ppp_hash`                           |
| Drift bank               | Robustness evaluation replay          | `drift_hash`, root seed, sample count                               |
| PPP                      | Feasibility envelope                  | canonical PPP JSON hash                                             |
| Checkpoint               | Model identity                        | checkpoint SHA256, model config, training config                    |
| Evaluation report        | Metric reproducibility                | report hash, commit hash, command line                              |

Add a short protocol paragraph:

> All stochastic components are derived from deterministic seeds scoped by source identifier, patch coordinate, PPP hash, and stage name. We report canonical JSON hashes for PPPs, split manifests, target records, surrogate shards, drift samples, checkpoints, and evaluation reports. A result is considered reproducible only if manifest hashes, shard hashes, and report hashes validate before metric aggregation.

This directly supports the paper's auditability claim.

## Suggested Evaluation Section Rewrite Outline

Use this structure instead of the current placeholder-heavy version:

1. **Datasets and Splits**
   - list source families and counts;
   - document alpha filtering and corrupt file exclusions;
   - give train/val/test counts.

2. **Baselines**
   - ICC CMYK baseline;
   - projected ICC baseline with `\Pi_K`;
   - learned deterministic baseline, if available;
   - ablations: without refiner, without surrogate, without PPP constraints.

3. **Protocol**
   - patch size, context size, drift sample count, q-risk definition;
   - PPP base family and overrides;
   - deterministic seeds;
   - checkpoint/report hashes.

4. **Primary Metrics**
   - DeltaE00 drift percentiles;
   - feasibility violation counts;
   - TAC/ink usage;
   - surrogate ranking metrics;
   - escalation/fallback rates.

5. **Secondary Metrics**
   - MSE/PSNR/SSIM;
   - edge error;
   - qualitative panels.

6. **Ablation and Failure Analysis**
   - compare components;
   - report where RobustSep fails: saturated neons, dark neutrals, thin text, high OGV constraints, severe drift.

## Specific Text-Level Issues

- The abstract uses "amalgamates"; replace with simpler academic wording such as "combines."
- "Non-deterministic architecture" may sound uncontrolled. Prefer "stochastic candidate-generation architecture with deterministic replay."
- "All color errors are assumed to carry equal perceptual weight" should be reframed as "standard aggregate color metrics weight all pixels uniformly"; commercial print does not treat all errors equally.
- The method should introduce CMYKOGV explicitly before using it heavily: define O, G, V as expanded-gamut Orange, Green, Violet channels and state that CMYK export is optional/downstream.
- The three lambda values are described as `{0.1, 0.5, 0.9}` in the paper, while candidate probe code also uses `{0.0, 0.3, 0.6, 0.9, 1.0}` for quality gates. Clarify training/inference lambda schedule versus probe lambda schedule.
- The paper should define the finite quantile operator: `Q_q(S) = sort(S)[ceil(q|S|)-1]`.
- The refiner formula currently shows proportional TAC scaling, but the implementation uses a capped-simplex-style downward projection. Align the paper with the actual operator.
- The Evaluation tables should not have blank cells. Use "Pending" only in an internal draft, never in a submission PDF.

## Diagram Plan and Generation Prompts

Use at most three figures. The goal is not to draw every module as a box; the goal is to make RobustSep feel like a serious, mathematically grounded ML system for physical production. These prompts intentionally ask for premium 3D/isometric polish, but the diagrams must remain readable in a paper PDF and must keep the math visible.

### Figure 1: RobustSep System Manifold

**Placement:** End of Introduction or beginning of Methodology, immediately before `Problem Formulation`.

**Purpose:** Replace the ordinary pipeline diagram with a visually memorable view of the full system: RGB patches enter a probabilistic/semi-physical decision machine; CMYKOGV candidates are projected into a feasible print manifold; drift-risk selects the output.

**Prompt:**

```text
Create a premium academic ML systems figure for a paper titled "RobustSep". The image should be a polished semi-3D/isometric technical diagram on a clean white or very light warm-gray background, suitable for NeurIPS/ICLR/CVPR. Visual metaphor: RGB image patches flow into a translucent mathematical "separation manifold" where multiple CMYKOGV candidate sheets are generated, projected into a constrained feasible region K(PPP), evaluated under drift, and one robust candidate exits as the selected CMYKOGV separation plus audit report.

Composition: left-to-right but with depth. Left: small RGB patch grid tiles entering the system. Center-left: a sleek PPP conditioning console showing a JSON-like card with "PPP", "caps", "TAC", "OGV", "drift". Center: conditional VAE represented as a luminous latent sphere or Gaussian cloud z with three emerging CMYKOGV candidate planes labeled lambda=0.1, 0.5, 0.9. Center-right: a translucent 3D feasible envelope labeled K(PPP), with candidates snapped/projected onto it by Pi_K. Right: drift sampler fans out N perturbed copies through a compact forward surrogate block, then a q90 tail-risk selector chooses the final output.

Include mathematical labels directly in the figure: y in K(PPP), Pi_K(y), epsilon ~ Pi(PPP), R_k = Q_0.90({d_k,i}), DeltaE00, CMYKOGV. Use a restrained but high-end palette: charcoal text, deep teal, cyan, controlled orange/violet accents for OGV, light gray glass surfaces, subtle shadows. Make it look expensive and precise, not cartoonish. Use crisp typography, thin technical arrows, subtle 3D depth, no decorative blobs, no photorealistic printing press, no clutter. The figure should communicate stochastic generation + physical feasibility + drift-tail selection in one glance.
```

### Figure 2: PPP Feasibility Geometry

**Placement:** Inside `Press Prior Package (PPP)`, after the PPP paragraph and before the drift distribution paragraph.

**Purpose:** Make the mathematical heart of the paper attractive and concrete: PPP is not just config; it defines a feasible geometric region with hard constraints and a projection operator.

**Prompt:**

```text
Create a beautiful, professional mathematical figure explaining the Press Prior Package and feasibility envelope in RobustSep. Style: semi-3D scientific visualization mixed with clean vector annotations, white background, journal-quality. Show a left panel as an elegant PPP object card, like a floating glass JSON schema, with fields: base_family, channel caps kappa_c, TAC_max, OGV_max, pair caps rho_CO rho_MG rho_YV, neutral/dark thresholds tau_chroma tau_dark, drift distribution Pi(PPP), override mask.

Right panel: a stylized 3D constrained feasible region K(PPP), like a clipped simplex/polytope inside a coordinate space. Use translucent planes for constraints: per-channel caps, TAC plane, OGV cap, pair-cap planes C+O, M+G, Y+V, and a highlighted neutral/dark OGV cap region. Show an infeasible candidate y outside the polytope and a clean projection arrow Pi_K moving it to y' on/in the feasible region. Include small formulas: 0 <= y_c <= kappa_c, sum_c y_c <= TAC_max, y_O+y_G+y_V <= OGV_max, y_C+y_O <= rho_CO, Pi_K: R^7 -> K(PPP).

Visual tone: mathematical, premium, precise, slightly futuristic but sober. Use translucent glass geometry, thin charcoal labels, muted teal feasible region, orange/violet OGV accents, gray constraint planes, subtle shadows. No cartoon icons, no busy pipeline boxes, no random gradients. The figure should make reviewers immediately understand that feasibility is a formal projected constraint set, not a heuristic postprocess.
```

### Figure 3: Drift-Risk Selection and Reproducible Evidence Chain

**Placement:** In `Refinement and Assembly`, immediately after the risk-score equation; optionally reference it again in Evaluation/Reproducibility.

**Purpose:** Combine the robustness criterion and verifiability story in one strong figure: candidates are judged by drift-tail distributions, and the whole decision is replayable via hashes/seeds.

**Prompt:**

```text
Create a high-end academic ML figure showing RobustSep drift-risk candidate selection with reproducibility evidence. White background, premium semi-3D/vector hybrid style, crisp labels and mathematical annotations. Left side: three elegant CMYKOGV candidate tiles y1, y2, y3 emerging from the feasible set, each with a small SHA/hash tag. From each candidate, show a fan of drift perturbation rays epsilon_1 ... epsilon_N sampled from Pi(PPP), rendered as translucent copies moving through a compact forward surrogate F_theta. Middle/right: for each candidate, show a sleek 3D mini distribution plot or violin/ridge plot of DeltaE00 errors under drift, with q90 marked by a vertical glowing line. Highlight the selected candidate as the one with lowest q90 tail risk, not necessarily lowest mean.

At the bottom, include a thin reproducibility chain as an integrated audit strip: split manifest hash -> target hash -> PPP hash -> drift hash -> checkpoint hash -> evaluation report hash. Do not make this a separate boring flowchart; make it feel like the evidence trail attached to the selected candidate. Include formulas: d_k,i = DeltaE00(F_theta(pi_i(y_k)), A*), R_k = Q_0.90({d_k,i}_{i=1}^N), k* = argmin_k R_k.

Visual style: professional ML paper, mathematically sexy, restrained colors, charcoal text, teal selected path, orange/violet drift accents, muted gray unselected candidates, subtle depth and shadows. Avoid decorative gradients, cartoons, or excessive icons. The figure should make the reviewer understand both why tail-risk is necessary and why the stochastic process is reproducible.
```

## Recommended Figure Order

1. **Figure 1** should be the main architecture figure and should appear early.
2. **Figure 2** should be the mathematical feasibility figure inside the PPP/method section.
3. **Figure 3** should carry both the drift-risk decision logic and the reproducibility/hash story.

If space is tight, keep Figures 1 and 3 in the main paper and move Figure 2 to the appendix only if the formal `K(PPP)` equations remain in the main text. I would not include more than three diagrams; beyond that, tables and quantitative plots should carry the paper.

## Suggested Tables to Add

1. **Dataset and Split Table**

   Columns: source family, raw files, staged patches, train examples, val examples, test examples, notes.

2. **PPP Base Family Table**

   Columns: base family, TAC max, OGV max, O/G/V caps, neutral/dark thresholds, risk threshold.

3. **Surrogate Quality Gate Table**

   Columns: split, mean DeltaE00, q90 DeltaE00, Spearman ranking, top-1 agreement, checkpoint hash, passed.

4. **Feasibility Violation Table**

   Columns: method, below zero, channel cap, TAC, OGV cap, neutral OGV, pair CO, pair MG, pair YV.

5. **Ablation Table**

   Columns: variant, q90 DeltaE00 under drift, TAC violation %, OGV violation %, escalation rate, runtime.

## Suggested Reviewer-Facing Claim Boundaries

Make these claims only if backed by the final reports:

- "guarantees print-feasibility" only after defining exactly which constraints are guaranteed by `\Pi_K`;
- "robust under drift" only with q90/q95 drift metrics and drift distribution definition;
- "generalizes across process families" only if evaluation includes multiple PPP base families or explicit cross-family tests;
- "brand-aware" only if brand regions are annotated or if the text says "brand-proxy intent heuristic";
- "ICC-supervised" only if the exact ICC/proxy path is specified.

## Minimal Edits I Would Apply Before Submission

1. Replace the Evaluation section placeholders with a reproducible protocol, even if final results are not ready.
2. Add the formal `K(PPP)` definition and full `\Pi_K` constraints.
3. Add an artifact-hash reproducibility subsection.
4. Fix citation mismatches.
5. Replace overclaiming language with measured or explicitly planned claims.
6. Add Figure 1 and Figure 3 at minimum.
7. Move any unfinished results tables to an internal draft or appendix marked "protocol template," not the main submission.
