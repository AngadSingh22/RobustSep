

# **RobustSep Architectural Specification**

Angad Singh Ahuja

December 20, 2025

1. # **Objective and scope**

This document specifies RobustSep v1.1 as an end-to-end, PPP\-conditioned pipeline that maps an input RGB image to a single internal CMYKOGV separation, with optional CMYK export, while enforcing print-feasibility and

robustness under process drift. The specification is designed to be implementable, reproducible, and auditable: every stochastic choice is seeded, every decision is logged, and every output is accompanied by a decision report that explains what constraints were tight, where risk concentrated, and which fallbacks were invoked.

As an internal reference, This is RobustSep v1.1 which differs from v1 in the following ways \-

(i) adds deterministic seeding (for all stochastic parts) as a hard requirement, (ii) a multi-shot candidate coverage escalation policy (increase candidate diversity before increasing robustness sampling), (iii) projection-closed refinement and a mandatory global post-blend feasibility projection, which basically means that, after the refiner which applies the constraint-patch to generated combinations of CMYK separations including say, any smoothing or neutral stabilisation edits, it must apply a feasibility projection, which is a set of non-negotiable constraints that must necessarily hold for a viable separation as the final step, so the refiner’s output is guaranteed to satisfy TAC and per-channel caps (and any other hard constraints)  
(iv) an intent-map layer that reweights risk and computes allocation across brand/flat/gradient regions (as was suggested by Shreeya in her modification, thanks for that\!),

2. # **Definitions and notation**

   1. ## **Signals**

Let the input image be *I*RGB ∈ \[0*,* 1\]H*×*W *×*3. The internal output separation is *I*CMYKOGV ∈ \[0*,* 1\]H*×*W *×*7. A downstream CMYK image may be emitted only through an explicitly documented export operator.

RobustSep operates patch-wise. Let the patch size be *s* \= 16\. A patch extracted at spatial coordinate *u* is denoted *x*u ∈ \[0*,* 1\]s*×*s*×*3. The corresponding internal CMYKOGV patch is *y*u ∈ \[0*,* 1\]s*×*s*×*7.

2. ## **PPP (Press Prior Package)**

PPP is an inference-time conditioning object. Conceptually, it provides:

* A feasibility envelope K(PPP) that encodes hard constraints (e.g., TAC limit, per-channel caps).

  * A drift distribution over plausible process perturbations (e.g., TRC shifts and ink-strength and viscosity variation).

    * Defaults and thresholds (e.g., robustness quantile denoted as *q*, drift sample count denoted as *N* , neutral thresholds) that can be overridden.

PPP schema is fundementally two-tier: a base\_family token each of which contains a default bundle of constraints and drift assumptions (TAC limit, channel caps, perturbation ranges, neutral thresholds, and so on) and then, the second tier \- if you know something specific about the job or press, you then override only the relevant numbers (for example, lower TAC-max, tighten K cap, widen TRC drift range) without redefining the whole profile.

Shipped base families are:

* film\_generic\_conservative (default)

  * film\_gravure\_generic

    * film\_flexo\_generic

    * paperboard\_generic

    * label\_stock\_generic

  3. ## **Intent maps**

RobustSep v1.1 uses three intent maps to represent region-specific priorities:

* *B*(*x, y*) ∈ {0*,* 1} for brand-critical regions (logos, brand colors, critical marks).

  * *F* (*x, y*) ∈ {0*,* 1} for flat regions (uniform fields where stability and banding avoidance dominate).

    * *G*(*x, y*) ∈ {0*,* 1} for gradient-sensitive regions (where continuity and smooth transitions dominate).

Intent maps are computed deterministically from the input image using *local statistics and edge- consistency cues*. (**Mathematically TBS**). They induce per-patch weights *w*u \= (*w*B*, w*F *, w*G),  
u	u	u

typically proportional to the fraction of pixels in the patch belonging to each intent class. Those weights then reweigh the risk score so errors in brand or gradient regions count more, and steer compute so important patches can get more candidate search or stricter evaluation than low-stakes flat patches.

4. ## **Models and operators**

   * Proposer *G*ϕ: a conditional VAE that proposes candidate CMYKOGV patches given *x*u, a structure token *τ*u, a style knob *λ*, and an embedding of PPP. (**Mathematically TBS**)

     * Forward surrogate *F*θ: a learned appearance model that predicts a Lab-like perceptual proxy from CMYKOGV under PPP (used for robustness evaluation). (**Mathematically TBS**)

     * Feasibility projection Π*K*(PPP)(·): a deterministic operator that maps any CMYKOGV patch to the feasible envelope K(PPP). (**Mathematically TBS**)

   5. ## **Robustness and risk**

Let *ϵ* ∼ D(PPP) denote a drift perturbation sampled from the PPP drift distribution. For a refined candidate *y*u, RobustSep evaluates appearance under drift, computes a perceptual error,

aggregates by a percentile, and uses this as candidate risk. (**Mathematically TBS**)

Let Err(*y*u*, ϵ*) be the (intent-weighted) perceptual error under drift. For *N* drift samples, define:

Risk(*y*u) \= Percentileq 1{Err(*y*u*, ϵ*i)}N 2 *,*

where *q* ∈ (0*,* 1\) is configured (e.g., *q* \= 0*.*90). As the system is robustness-first, selection minimises Risk(*y*u) among feasible candidates, with a tie-break on nominal (non-drift) error. (**Mathematically TBS**)

3. # **System contract**

   1. ## **Inputs**

RobustSep v1.1 takes:

* *I*RGB: the input image.

  * PPP: base family plus overrides (if any).

    * *λ*: a style knob trading conservative stability against saturation/exploration.

    * Seed policy: deterministic seeding for all stochastic components.

  2. ## **Outputs**

RobustSep v1.1 produces:

* *I*CMYKOGV: a single internal CMYKOGV image, plus optional exported CMYK if requested.

  * A decision report (Section [7](#selected-patches-are-blended-into-the-full-image-using-overlapping,-edge-aware-blending-\(windowed-weighting-with-seam-downweighting-based-on-edges\).-after-blending,-the-system-must-apply-a-full-image-feasibility-check-and-must-apply-pixelwise-πk\(ppp\)-as-a-global-safety-pass-if-any-violations-remain.-\(mathematically-tbs\))) containing seeds, thresholds, failures, hotspots, and applied fallbacks.

  3. ## **Hard guarantees**

The system must satisfy:

* **Feasibility closure:** after the final global safety pass, all pixels comply with K(PPP) (e.g., TAC and channel caps).

  * **Robustness-first selection:** the chosen per-patch candidates minimize percentile risk under

PPP drift among feasible candidates.

* **Reproducibility:** rerunning with the same inputs, PPP, and seeds reproduces identical candidates, selections, and outputs.

4. # **PPP specification**

   1. ## **Required fields for PPP to be parsed**

A PPP instance must necessarily include the following fields; exact JSON keys are fixed by the implementation schema and recorded in the manifest.

* **Feasibility constraints:** TACmax and per-channel caps (*c*max*, m*max*, y*max*, k*max), plus any overprint sanity rules.

  * **Drift model:** ranges/strengths for ink multipliers and TRC perturbations, and sampling policy.

    * **Robustness policy:** default *N* (drift samples), quantile *q*, and any rejection thresholds.

    * **Neutral/dark policy:** thresholds for neutral/dark stabilization (e.g., dark and near-neutral criteria) and any scaling constants.

  2. ## **Overrides**

Overrides may replace defaults from the selected base family. The decision report must record both the base family and the override mask (which fields were overridden).

5. # **Pipeline overview**

RobustSep v1.1 executes the following stages:

1. Pre-run: In this step, we load the two learned models which are *G*ϕ referring to the conditional VAE that proposes 3 CMYKOGV separations per 16x16 RGB patch and *F*θ, referring to the forward surrogate model that predicts appearance in a Lab-like space. Simultaneously, we also load the PPP presets for the chosen family \+ any overrides, and establish deterministic seeds so the same input and PPP produce the same candidates and drift samples across different runs. (**Mathematically TBS**)

2. Preprocess: compute intent maps (*B, F, G*) per pixel using the yet to be specified statistical patch, extract overlapping 16 × 16 patches, alongside this, we also compute structure token *τ*u ∈ {edge, flat, textured} for each 16x16 patch (**Mathematically TBS**) .

3. Patch engine: for each patch *u*, propose candidates, refine with projection-closed constraints, evaluate robustness risk under PPP drift using intent-weighted scoring, select the best candidate, and cache diagnostics. (**Mathematically TBS**)

4. Aggregation: blend selected patches with edge-aware overlap blending, then apply a mandatory global feasibility safety pass. (**Mathematically TBS**)

5. Emit outputs: final *I*CMYKOGV, optional exported CMYK if configured, plus decision report and artifacts.

6. # **Module specifications**

   1. ## **Intent map module**

**Input:** *I*RGB. **Output:** binary maps *B, F, G* plus per-patch weights *w*u.

**Requirement:** intent computation must be deterministic and its parameters (thresholds, kernels) must be recorded in the artifact manifest. (**Mathematically TBS**)

2. ## **Patching and structure token**

**Input:** *I*RGB. **Output:** overlapping patches *x*u and a structure token *τ*u ∈ {edge, flat, textured} for each patch.

**Requirement:** patch extraction must record stride/overlap settings and blending policy in the run manifest. (**Mathematically TBS**)

3. ## **Proposer** *G*ϕ

**Input:** (*x , τ , λ,* PPP*,* seed). **Output:** *K* candidate CMYKOGV patches {*y*(k)}K .  
u	u	u

k\=1

PROCESS \- (**Mathematically TBS**) **Default:** *K*base \= 3; research cap *K*cap \= 5\.  
**Training:** synthetic-first targets derived from PPP\-driven simulation, with ICC/profile-derived  
anchors as regularization to prevent synthetic drift. Training details are out of scope for this

inference-time spec but model checkpoints and configs must be versioned \- (**Implementationally TBS**)

4. ## **Refiner** *R* **(projection-closed)**

**Input:** candidate patch *y*, plus PPP. **Output:** refined patch *y*˜ such that *y*˜ ∈ K(PPP). The refiner (**Mathematically TBS**) applies a constraint stack (**Mathematically TBS**):

* Ink-limit logic: enforce TAC and per-channel caps; apply overprint sanity checks.

  * Patch smoothness and separation regularity: reduce spatial and inter-channel discontinuities that cause artifacts.

    * Neutral/dark stability: ensure stable behavior in dark and near-neutral regions based on configured thresholds.

## **Two-pass operational form:**

* Pass 1: smoothing/regularity adjustments and neutral/dark stabilization. (**Mathematically TBS**)

  * Pass 2: feasibility projection Π*K*(PPP) (closure step, always applied). (**Mathematically TBS**)

  5. **Feasibility projection** Π*K*(PPP)

Π*K*(PPP) is a deterministic closure operator that enforces hard constraints encoded by K(PPP) at the pixel level (**Mathematically TBS**).

A typical projection policy is:

* Clip each channel to its cap.

  * If TAC exceeds TACmax, apply the resolved capped-simplex TAC policy over active CMYKOGV channels.

  6. ## **Drift sampler**

**Input:** PPP and seed. **Output:** drift samples *ϵ*i. Drift (**Mathematically TBS**) is modeled as:

* Per-channel ink-strength multipliers (channelwise scaling).

  * Monotone TRC perturbations (tone-curve deformation), applied channelwise and clipped back to \[0*,* 1\].

  7. ## **Forward surrogate** *F*θ

**Input:** CMYKOGV patch *y*, PPP, and drift *ϵ*. **Output:** perceptual proxy *L*ˆ ∈ Lab (patchwise). (**Mathematically TBS**)

**Requirement:** *F*θ must be versioned and its training regime documented; the inference run manifest must record the exact checkpoint hash.

8. ## **Robustness evaluator and intent-weighted scoring**

For each refined candidate *y*˜(k):

1. Sample *N* drift instances *ϵ*1*, . . . , ϵ*N (default *N* \= 32).

2. For each *ϵ*i, compute Lab proxy via *F*θ and compute per-pixel ∆*E*00 against the reference appearance.

3. Aggregate per-pixel errors with intent weights (*w*B*, w*F *, w*G) to produce Err(*y*˜(k)*, ϵ* ).

   u	u	u	u	i

4. Aggregate across drift by Risk(*y*˜(k)) \= Percentile ({Err(*y*˜(k)*, ϵ* )}N )

   u	q	u	i	i\=1

Above stated strategy needs further refinement for most efficient strategy therefore marking as (**Mathematically TBS**)

9. ## **Selection and multi-shot compute gating**

Selection is performed per patch:

* Discard candidates that fail feasibility after refinement.

  * Among remaining candidates, discard those whose risk exceeds the configured threshold.

    * Select the minimum-risk candidate; tie-break by nominal (non-drift) error.

## **Multi-shot escalation policy (mandatory):** (**Mathematically TBS**)

* If no candidate survives, increase candidate coverage first (increase *K* up to *K*cap) and rerun propose-refine-evaluate.

  * Only if coverage escalation fails should the system consider increasing *N* , and this must be logged as a rare fallback.

**Intent-conditioned compute gating:** patches with high brand or gradient weight may be allowed higher escalation budgets than flat regions, consistent with the intent policy. (**Mathematically TBS**)

10. ## **Aggregation and global safety pass**

Selected patches are blended into the full image using overlapping, edge-aware blending (windowed weighting with seam downweighting based on edges). After blending, the system must apply a full-image feasibility check and must apply pixelwise Π*K*(PPP) as a global safety pass if any violations remain. (**Mathematically TBS**)

7. # **Decision report schema**

The decision report must include, at minimum:

* PPP base family and override mask; effective feasibility and drift parameters.

* *λ*, *K*base, *K*cap, realized *K* per patch (including escalations), *N* , *q*.

* Deterministic seeds for proposer sampling and drift sampling.

* Feasibility pass/fail counts; per-constraint failure reasons (by constraint class).

* Risk summaries (percentile risk per patch), risk hotspot map, and constraint tightness indicators.

* Rollback flags (patch-aware to per-pixel) and the reason for rollback if triggered.

* Global safety pass applied (yes/no) and post-pass feasibility status.

* If bridge enabled: per-object scores under *F*P , actions taken, and review\_required boolean.

8. # **Acceptance criteria and failure modes**

   1. ## **Acceptance criteria**

A run is accepted if:

* Final output is feasible everywhere after the global safety pass.

  * Robustness evaluation executed with declared *N* and *q*, and selection followed the robustness- first rule.

    * Decision report is complete and includes all seeds and thresholds required for replay.

  2. ## **Failure modes and required handling**

     * **Empty feasible set per patch:** invoke multi-shot escalation (increase *K*) before any *N*

increase; log escalations and final outcome.

* **Patch seams or instability after blending:** apply edge-aware blending; if value/stability criteria fail, trigger rollback to per-pixel mapping while preserving semantics.

  * **Residual infeasibility after blending:** global safety pass via pixelwise projection is mandatory; log that it was applied.

9. # **Reproducibility artifacts**

The release must necessarily ship or record:

* PPP preset library and override schema documentation.

  * Drift sampler implementation and parameter ranges.

    * Model checkpoints for *G*ϕ and *F*θ with hashes and configs.

    * Dataset manifests and checksums for any training/evaluation sets used in the release.

    * Evaluation logs and per-run decision reports sufficient to replay patch-level and image-level outputs.

10. # **Mathematically TBS (To Be Specified) Components and Required Deliverables**

If you have given this specification a close-read, you must have noticed that I have marked several componenents (or whole blocks) as **Mathematically TBS** which implies that each of these are therefore **not yet mathematically specified** which is a major roadblock in the development of this pipeline.

As such, find below a list of the specified TBS items:

1. ## **List of Mathematically TBS items**

1. **Intent map construction (pixel-level)**: the exact definition of *local statistics* and *edge-consistency cues*, and the deterministic rule-set that maps *I*RGB to intent maps *B*(*x, y*)*, F* (*x, y*)*, G*(*x, y*).

2. **Intent-to-patch aggregation**: the exact computation of per-patch weights *w*u \= (*w*B*, w*F *, w*G) from pixel maps, including windowing, overlap handling, and any normaliza-

u	u	u  
tion.

3. **Structure token computation**: the deterministic procedure that assigns each patch *u* a token *τ*u ∈ {edge, flat, textured}, including all thresholds/features.

4. **Proposer model** *G*ϕ **(inference mapping)**: the mathematically defined conditional input

space, conditioning mechanism for PPP and *λ*, latent sampling procedure, and the precise

definition of the output candidate set {*y*(k)}K  .  
u	k\=1

5. **Proposer model** *G*ϕ **(training specification)**: the training objective, synthetic-first target construction, and the ICC/profile-derived regularization term (math form and how it is applied).

6. **Forward surrogate** *F*θ **(inference mapping)**: the precise function definition from (*y,* PPP*, ϵ*) to the Lab-like proxy output, including how drift *ϵ* is injected/applied.

7. **Forward surrogate** *F*θ **(training specification)**: the mathematical training objective, synthetic chart generation regime, and ICC regularization term (math form and how it is applied).

8. **Feasibility envelope** K(PPP): the complete mathematical definition of the feasible set induced by PPP, including TAC constraints, per-channel caps, and any overprint sanity rules as explicit inequalities/logic.

9. **Feasibility projection** Π*K*(PPP): the exact projection operator (pixelwise/patchwise), includ- ing the chosen policy for resolving TAC violations (e.g., proportional scaling versus another configured policy), and proof-relevant properties required for correctness.

10. **Refiner** *R*: the exact mathematical form of the refiner, including its update rule (or optimiza- tion formulation), and how it composes (i) ink-limit logic, (ii) patch smoothness/separation regularity, (iii) neutral/dark stabilization, and (iv) mandatory feasibility closure by Π*K*(PPP).

11. **Refiner constraint stack (formalization)**: the full mathematical definitions of:

    1) Ink-limit logic (including overprint sanity constraints),

    2) Patch smoothness / separation regularity term(s),

    3) Neutral/dark stability term(s) and thresholds.

12. **Refiner Pass 1 (formal update)**: the exact mathematical update(s) implementing smooth- ing/regularity adjustments and neutral/dark stabilization.

13. **Refiner Pass 2 (formal closure)**: the exact mathematical role of applying Π*K*(PPP) as an always-on closure step.

14. **Drift sampler** D(PPP): the exact distribution family and sampling procedure for drift *ϵ*, including:

    1) Per-channel ink-strength multipliers (distributional form and parameterization),

    2) Monotone TRC perturbations (construction, constraints for monotonicity, parameteriza- tion),

    3) Clipping/validity handling and seeding behavior.

15. **Per-drift error** Err(*y*u*, ϵ*): the exact intent-weighted perceptual error computation, including:

    1) How ∆*E*00 is computed/aggregated spatially within a patch,

    2) How (*w*B*, w*F *, w*G) reweight the error (explicit formula),  
       u	u	u

    3) Any additional stability penalties (if included).

16. **Risk aggregation** Risk(*y*u): the precise definition of Percentileq(·) as implemented (finite- sample quantile definition, tie handling), and any rejection thresholds used in candidate filtering.

17. **Patch engine end-to-end semantics**: the full mathematical specification of the propose– refine–evaluate–select loop, including feasibility filtering, robustness filtering, tie-break rules, and caching semantics.

18. **Multi-shot escalation policy**: the exact policy definition (when *K* is increased, step size

∆*K*, cap behavior, and when/if *N* is increased), including its logged events and stopping conditions.

19. **Intent-conditioned compute gating**: the exact rule that maps intent weights *w*u to compute allocation (e.g., allowed escalation budget by patch class), including thresholds and caps.

20. **Edge-aware overlap blending**: the explicit blending function used to compose selected patches into *I*CMYKOGV, including window function(s), edge-derived seam downweighting defini- tion, and overlap/stride parameters.

21. **Global feasibility safety pass**: the precise full-image feasibility check and the exact pixelwise application of Π*K*(PPP) (including when it triggers and what is logged).

22. **Manifested determinism requirements (where marked)**: wherever “parameters must be recorded in the artifact manifest” is marked **Mathematically TBS**, the exact parameter list and the formal reproducibility contract (what must be sufficient to replay bit-identically) must be specified.

    2. ## **Mandatory deliverable for each TBS item (before implementation)**

For each item above that is currently unspecified, we must (i) design a high-accuracy mathe- matically specified patch in the form of exactly computable inputs and outputs (for ML parts, inputs and outputs for both training and inference), (ii) provide pseudocode that is sufficient to implement the patch exactly as specified, (iii) provide a formal mathematical guarantee of correctness in the form of a barebones readable proof-like argument, and (iv) experimentally verify and document said experimentation in the GitHub Repo that will be created by Shreeya by 22nd of December, EOD before adding it to the code.

# **Resolved Mathematical Specification**

This section resolves the mathematically TBS items above by consolidating them with
`vae_fs_pargb2cmyk.md`, which is the controlling specification for the VAE proposer, forward
surrogate, target-generation loop, FiLM conditioning, drift-aware scoring, and quality gate.

The canonical internal separation space is CMYKOGV. Every candidate, refiner operation,
surrogate input, drift sample, feasibility projection, and risk computation operates on

```text
z in [0,1]^(H x W x 7), channels ordered as C,M,Y,K,O,G,V.
```

If a downstream consumer requires CMYK, the conversion from the seven-channel internal result
to CMYK is a separate export operator and is not allowed to silently redefine the internal math.

1. ## **Intent Maps, Patch Aggregation, And Structure Tokens**

Convert `I_RGB` deterministically to Lab using the locked RGB -> linear RGB -> XYZ -> Lab D50
pipeline. If alpha is available, retain `alpha(p) in [0,1]`; fully transparent pixels receive zero
statistical weight and partial-alpha edge pixels are retained with weight `alpha(p)^gamma_alpha`.

For each pixel `p`, compute:

```text
L(p), a(p), b(p)
C_lab(p) = sqrt(a(p)^2 + b(p)^2)
E(p) = Sobel magnitude on L
V_r(p) = local Lab variance in a fixed radius r=3 window
rho(p) = (lambda_1(p)-lambda_2(p))/(lambda_1(p)+lambda_2(p)+eps)
```

where `lambda_1 >= lambda_2` are the eigenvalues of the Lab/L structure tensor. Hue smoothness
`H_s(p)` is the fixed-radius circular hue variance where chroma is above the hue-valid threshold.

Raw intent scores are:

```text
S_F(p) = 1[V_r(p) <= theta_flat_var and E(p) <= theta_flat_edge]

S_G(p) = 1[
  theta_grad_min <= E_smooth(p) <= theta_grad_max
  and rho(p) >= theta_grad_coh
  and H_s(p) <= theta_hue_smooth
  and V_r(p) > theta_flat_var
]

S_B(p) = max(
  user_brand_mask(p),
  1[C_lab(p) >= theta_brand_chroma
    and E(p) >= theta_brand_edge
    and rho(p) >= theta_brand_coh]
)
```

The user brand mask defaults to zero if absent. Collision resolution is deterministic:

```text
B(p) = S_B(p)
G(p) = S_G(p) * (1 - B(p))
F(p) = max(S_F(p) * (1 - B(p)) * (1 - G(p)), 1 - B(p) - G(p))
```

For a patch `u` with pixels `P_u`, raised-cosine patch window `omega(p)`, and alpha weight
`A(p)=alpha(p)^gamma_alpha`, patch intent weights are:

```text
r_c(u) = sum_{p in P_u} omega(p) A(p) M_c(p) / max(eps, sum_{p in P_u} omega(p) A(p))
w_c(u) = r_c(u) / max(eps, r_B(u)+r_F(u)+r_G(u))
```

If the denominator is below `alpha_min_patch`, the patch is skipped in training and marked
transparent in inference diagnostics.

The structure token is assigned from deterministic patch features:

```text
edge_density = mean_A(1[E(p) > theta_edge])
var_lab = mean_A(V_r(p))
texture_energy = mean_A(abs(LoG(L)(p)))
coherence = mean_A(rho(p))

tau_u =
  edge      if edge_density >= theta_edge_density and coherence >= theta_edge_coh
  flat      if var_lab <= theta_flat_var_patch and edge_density <= theta_flat_density
  textured  otherwise
```

Guarantee: intent maps, patch weights, and structure tokens are deterministic functions of input
bytes, color conversion, kernels, thresholds, padding, alpha policy, patch window, and stride.
All such parameters and map hashes must be recorded in the manifest.

2. ## **PPP Feasibility Envelope And Projection**

An effective PPP is produced by canonicalizing the selected base family and applying numeric
overrides with an override mask. The per-pixel feasible set is:

```text
K(PPP) = {
  z:
    0 <= z_j <= cap_j                         for j in C,M,Y,K,O,G,V
    sum_j z_j <= TAC_max
    z_O + z_G + z_V <= OGV_max
    z_C + z_O <= pair_CO_max                  if enabled
    z_M + z_G <= pair_MG_max                  if enabled
    z_Y + z_V <= pair_YV_max                  if enabled
    z_O + z_G + z_V <= neutral_OGV_max        if neutral_or_dark(p)
}
```

Neutral/dark activation is:

```text
neutral_or_dark(p) =
  sqrt(a_ref(p)^2 + b_ref(p)^2) <= neutral_chroma_threshold
  or L_ref(p) <= dark_L_threshold
```

The projection `Pi_K` is deterministic and pixelwise:

```text
Pi_K(z):
  z_j <- clip(z_j, 0, cap_j)

  for group in [(C,O,pair_CO_max), (M,G,pair_MG_max), (Y,V,pair_YV_max)]:
    if group is enabled and z_i + z_j > group_cap:
      s <- group_cap / (z_i + z_j + eps)
      z_i <- s*z_i
      z_j <- s*z_j

  if O+G+V > OGV_max:
    s <- OGV_max / (O+G+V+eps)
    O,G,V <- s*O, s*G, s*V

  if neutral_or_dark and O+G+V > neutral_OGV_max:
    s <- neutral_OGV_max / (O+G+V+eps)
    O,G,V <- s*O, s*G, s*V

  if sum_j z_j > TAC_max:
    solve x = argmin ||x-z||_2^2
      subject to 0 <= x_j <= z_j and sum_j x_j = TAC_max
    using deterministic capped-simplex water filling
    z <- x

  return z
```

Correctness: all operations only lower nonnegative channel values or clip to declared caps. The
group cap steps cannot create channel-cap violations, and the final capped-simplex step cannot
increase any channel or group sum. Therefore `Pi_K(z) in K(PPP)`. If `z` already satisfies all
constraints, no branch changes it, so feasible pixels are preserved exactly up to the declared
floating tolerance.

Manifest requirements: effective PPP JSON, override mask, caps, TAC, group caps, neutral
thresholds, projection version, epsilon, dtype, tolerance, pre/post violation counts, and maximum
residual violation.

3. ## **Drift Sampler**

A drift sample is:

```text
epsilon_i = {m_j, T_j}_{j in C,M,Y,K,O,G,V}
```

with per-channel ink multiplier

```text
log m_j ~ TruncatedNormal(0, sigma_j, [-a_j, a_j])
m_j = exp(log m_j)
```

and a monotone tone reproduction curve. With `J=7` interior knots:

```text
x_k = k/(J+1), k=0..J+1
y_0 = 0, y_{J+1} = 1
raw_y_k = x_k + delta_k
delta_k ~ TruncatedNormal(0, sigma_trc_j, [-b_j, b_j])
y <- deterministic isotonic projection of raw_y with fixed endpoints
T_j(t) = linear interpolation through (x_k, y_k)
```

Drift application is:

```text
D_epsilon(z)_j = clip(m_j * T_j(z_j), 0, 1)
```

Seeds are derived by:

```text
seed(scope,u,k,i) = uint64_hash(root_seed, input_hash, ppp_hash, scope, patch_coord_u, candidate_k, drift_i)
```

The same drift bank is used for every candidate of the same patch so candidate comparisons are
paired and fair. The manifest records RNG algorithm, root seed, seed derivation formula, `N`,
`q`, channel sigmas, truncation bounds, knot count, sampled drift bank or drift-bank hash, and
isotonic projection version.

4. ## **Proposer And Synthetic Target Training**

The VAE proposer is the FiLM-conditioned architecture defined in `vae_fs_pargb2cmyk.md`.
Inference is:

```text
G_phi: (x_u, alpha_u, tau_u, w_u, intent_raster_u, PPP, lambda_k, z_k)
       -> y_k in [0,1]^(16 x 16 x 7)
z_k ~ N(0,I)
```

The default lambda schedule appends candidates without resampling earlier candidates:

```text
Lambda_3 = [0.1, 0.5, 0.9]
Lambda_4 = [0.1, 0.5, 0.9, 0.0]
Lambda_5 = [0.1, 0.5, 0.9, 0.0, 1.0]
```

Candidate `k` has its own seed and is reproducible independently. Escalating from `K=3` to
`K=5` only evaluates candidates 4 and 5 in addition to the cached first three.

Synthetic target generation initializes from the ICC CMYK baseline:

```text
y0 = [C_icc, M_icc, Y_icc, K_icc, 0, 0, 0]
```

Before a real seven-channel print characterization or gated surrogate exists, OGV appearance is
not claimed to be ICC-grounded. OGV is initialized to zero and may be introduced only through
the documented constrained solver, with explicit OGV penalties and teacher-mode flags.

The nominal target solve is:

```text
min_y
  alpha_app    * mean_alpha DeltaE00(Render(y), Lab_ref)
+ alpha_anchor * mean_alpha DeltaE00(Render(y), Render(y0))
+ alpha_ogv    * mean_alpha (O+G+V)
+ alpha_neutral* NeutralPenalty(y, Lab_ref)
subject to y in K(PPP)
```

The drift refinement solve is:

```text
min_y
  beta_risk  * Risk_0.90(y)
+ beta_trust * || W_channel * (y - y_stage1) ||_2^2
+ beta_ogv   * mean_alpha(O+G+V)
subject to y in K(PPP)
```

Both stages use fixed-step projected gradient descent:

```text
for t in 1..T:
  y <- y - eta * grad J(y)
  y <- Pi_K(y)
```

Thus every generated target is feasible after every update. Training losses are the Smooth L1
ink reconstruction, appearance auxiliary loss, KL warmup, and lambda monotonicity hinge
already specified in the VAE document.

5. ## **Forward Surrogate**

The surrogate is the Micro-CNN defined in `vae_fs_pargb2cmyk.md`. Under the current
ICC/proxy teacher it is a local appearance-delta risk model, not a global press renderer and not
a feasibility model:

```text
F_theta: (32x32 CMYKOGV context, PPP, tau, intent, lambda, epsilon)
         -> DeltaLab_hat for the scored center 16x16 region
```

The scored Lab prediction is reconstructed around the stored reference:

```text
y_drift = D_epsilon(y)
DeltaLab_hat = F_theta(y_drift, cond(PPP,tau,intent,lambda,epsilon))
Lab_hat = Lab_ref_center + DeltaLab_hat
```

The required v2 shard schema stores `lab_ref_center`, `teacher_lab_nominal`,
`teacher_lab_drifted`, `cmykogv_context`, `candidate_type`, `lambda_value`, drift parameters,
`ppp_hash`, and source metadata. `surrogate-shards-v1` artifacts and checkpoints trained from
them are semantically obsolete: they may be inspected for debugging, but must not unlock VAE
training, hybrid target generation, or paper claims.

Training uses Huber loss against local teacher deltas for nominal and drifted examples, plus the
fixed heldout quality gate:

```text
pass iff
  mean_DeltaE00 <= threshold_mean
  q90_DeltaE00 <= threshold_q90
  Spearman >= threshold_spearman
  top1_agreement >= threshold_top1
  mean_regret <= threshold_mean_regret
  q90_regret <= threshold_q90_regret
```

If the gate fails, hybrid target generation is disabled and any inference using the surrogate must
be marked fallback/experimental. Feasibility remains guaranteed only by `Pi_K`.

6. ## **Refiner**

The refiner is projection-closed:

```text
R(y, x, PPP) = Pi_K(Pass1(y, x, PPP))
```

Pass 1 is a fixed deterministic proximal update:

```text
J_ref(y) =
  0.5 ||y - y_in||_2^2
+ alpha_tv  * sum_{p,q neighbors} g_x(p,q) * Huber(y_p - y_q)
+ alpha_sep * sum_p AntagonistPenalty(y_p)
+ alpha_neu * NeutralDarkPenalty(y_p, Lab_ref_p)

g_x(p,q) = exp(-kappa * ||Lab_x(p)-Lab_x(q)||_2)
```

Neutral/dark stabilization activates under the same neutral/dark predicate used by PPP. The
neutral target operator sets `CMY` toward their mean, suppresses `OGV`, and keeps `K` subject
to PPP caps. Pass 1 may improve smoothness and stability, but Pass 2 is mandatory; only
`Pi_K` provides the hard feasibility guarantee.

7. ## **Error, Risk, Selection, And Escalation**

For a drift sample `epsilon_i`:

```text
d_p(epsilon_i) = DeltaE00(Lab_hat_p(y,epsilon_i), Lab_ref_p)
intent_gain(p) = 1 + beta_B B(p) + beta_G G(p) + beta_F F(p)
W_p = omega(p) * alpha(p)^gamma_alpha * intent_gain(p)
```

Patch error is:

```text
Err(y,epsilon_i) =
  sum_p W_p d_p(epsilon_i) / max(eps, sum_p W_p)
+ rho_tail * WeightedOrderStatistic_0.95({d_p}, {W_p})
+ eta_reg * TV_edge_aware(y)
```

Hard feasibility is not a soft penalty here; infeasible candidates must be projected or rejected.
Risk uses a non-interpolated finite order statistic:

```text
Risk_q(y) = sort([Err(y,epsilon_i)]_{i=1..N})[ceil(q*N)-1]
```

For `q=0.90` and `N=32`, this is zero-based index 28.

Selection uses a total deterministic ordering:

```text
1. feasible after Pi_K
2. below risk threshold if the threshold is hard, else threshold breach is logged
3. minimum Risk_q
4. minimum nominal identity-drift error
5. lower mean TAC
6. lambda-aware tie-break: lower OGV for lambda <= 0.5, higher chroma preservation for lambda > 0.5
7. lower candidate index
```

Compute gating is a pure function of PPP and patch intent:

```text
priority_u = 2*w_B + 1.5*w_G + 0.5*w_F

K_cap_u =
  5 if w_B >= 0.25 or w_G >= 0.35
  4 if priority_u >= 1.0
  3 otherwise

N_u = 32 by default
N_fallback_u = 64 only for high brand/gradient patches or PPP override
```

Escalation increases candidate coverage before drift count:

```text
K <- K_base
while K <= K_cap_u:
  evaluate candidates 1..K, reusing cached candidates and paired drift bank
  if selectable candidate exists: return selected candidate
  K <- K + 1

if no candidate and N_fallback_u > N_u:
  evaluate once at K_cap_u with N_fallback_u

if still no candidate:
  choose the feasible minimum-risk candidate if PPP threshold is soft
  otherwise return a fatal per-patch failure with reason
```

Every selected patch therefore follows a deterministic total ordering and every failure has a
logged reason.

8. ## **Edge-Aware Blending And Global Safety**

Default patch size is `s=16` and default stride is `8`. The base window is:

```text
h(i) = 0.1 + 0.9 * sin(pi*(i+0.5)/16)^2
W(i,j) = h(i)h(j)
```

Let `border(i,j)` be the normalized proximity to the patch border and `E_norm(p)` the clipped
normalized Sobel magnitude of the input image. Patch contribution weight is:

```text
A_u(p) = W(local_p) * (1 - rho_seam * border(local_p) * E_norm(p))
A_u(p) = max(A_u(p), alpha_blend_min)
```

The raw blended image is:

```text
I_raw(p) = sum_{u contains p} A_u(p) y_u(p-u) / sum_{u contains p} A_u(p)
```

Then:

```text
if any pixel violates K(PPP) by more than tau_feas:
  I_final(p) = Pi_K(I_raw(p)) for all p
else:
  I_final = I_raw
```

The global safety pass records changed pixel count, maximum channel delta, maximum TAC
violation before projection, and post-pass feasibility status. If projection delta exceeds the PPP
risk-invalidation threshold, the decision report must flag that final appearance risk should be
re-evaluated.

9. ## **Manifested Determinism Contract**

A run manifest must be sufficient to replay patch choices and final separations. It must include:

```text
input image hash
effective PPP canonical JSON and hash
override mask
dataset and shard hashes for training/evaluation
software versions and git SHA
model architecture and checkpoint hashes
ICC profile hashes
RNG algorithm and root seed
seed derivation formula
per-patch proposer seeds or seed hashes
per-patch drift seeds or drift-bank hashes
patch grid, stride, padding, and alpha policy
intent kernels, thresholds, collision priority, and map hashes
structure kernels and thresholds
projection parameters and tolerance
refiner objective parameters
DeltaE00 implementation identifier
risk quantile policy
candidate lambda schedule
selection ordering
float dtype and deterministic device policy
all fallback/rejection diagnostics
```

With identical input bytes, canonical PPP, model files, ICC profiles, code version, dtype/device
policy, and manifest parameters, every stochastic value and branch decision is reproducible. For
strict bit identity on GPU, deterministic kernels are mandatory; otherwise the contract is numeric
replay within the declared tolerance.

# **Algorithms (reference pseudocode)**

**Algorithm 1** RobustSep v1.1 end-to-end inference (PPP-first, intent-aware)  
1: **Inputs:** *I*RGB, PPP, *λ*, deterministic seed policy  
2: Load *G*ϕ, *F*θ, PPP presets; set seeds  
3: Compute intent maps (*B, F, G*) and per-patch weights *w*u

4: Extract overlapping 16 × 16 patches {*x*u}; compute structure tokens {*τ*u}

5: **for** each patch *u* **do**  
6:	Run PatchEngine(*x*u*, τ*u*, w*u*,* PPP*, λ*) to get selected patch *y*⋆ and diagnostics; cache

candidates

7: **end for**

8: Blend {*y*⋆} into *I*CMYKOGV using edge-aware overlap blending  
u  
9: If any feasibility violations in *I*

CMYKOGV

, apply global pixelwise projection Π*K*(PPP)

10: **if** optional P-profile bridge enabled **then**

11:	Run ObjectBridge(candidate cache, object boundaries, validated P-profile) to update object-wise CMYKOGV  
12:	Recompose objects to image; apply global pixelwise projection Π*K*(PPP) if needed

13: **end if**

14: Emit final *I*CMYKOGV, optional exported CMYK if configured, and decision report

**Algorithm 2** PatchEngine (projection-closed refinement, intent-weighted robustness, multi-shot escalation)  
1: **Inputs:** *x*u*, τ*u*, w*u*,* PPP*, λ*, seeds; defaults *K*base*, K*cap*, N, q*  
2: Set *K* ← *K*base  
3: **repeat**  
4:	Propose candidates {*y*(k)}K	← *G*ϕ(*x*u*, τ*u*,* PPP*, λ,* seed)

5:	**for** each candidate

*y*(k)

k\=1

## **do**

6:	Refine: apply smoothing/regularity and neutral/dark stabilization  
7:	Apply feasibility closure: *y*˜(k) ← Π*K*(PPP)(*y*(k))  
8:	Sample drifts *ϵ*1*, . . . , ϵ*N ∼ D(PPP)  
9:	Compute intent-weighted errors Err(*y*˜(k)*, ϵ*i) using *F*θ and weights *w*u

10:	Risk Risk(*y*˜(k)) ← Percentileq({Err(*y*˜(k)*, ϵ*i)}N )

11:	**end for**

12:	Filter infeasible or over-threshold candidates; select minimum-risk feasible candidate if any  
13:	**if** no candidate survives **then**

14:	Increase candidate coverage first: *K* ← min(*K*cap*, K* \+ ∆*K*); log escalation

15:	**end if**

16: **until** candidate selected OR *K* \= *K*cap and escalation exhausted

17: If still no candidate, optionally increase *N* as rare fallback; log this event

18: Return selected patch and diagnostics (seeds, *K*, *N* , *q*, failures, risk percentile, tight constraints)
