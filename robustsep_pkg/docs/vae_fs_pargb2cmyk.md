# Per-patch proposer (Conditional-VAE) + Surrogate

# (Micro-CNN) Dataflow and Architecture

# Inference -

**Input for each patch (denoted as a tuple in prior documentation) -**

- An RGB patch in sRGB format, size 16 by 16 pixels, with values between 0 and 1.
- A PPP package containing base family, numeric overrides, and an override mask. PPP
    defines the feasibility envelope, the drift distribution, and defaults such as the number of
    drift samples and the risk quantile.
- A structure token for the patch: edge, flat, or textured, computed deterministically during
    preprocessing.
- Intent maps for brand, flat, and gradient regions, used to compute per-patch intent
    weights. These weights are deterministic and typically proportional to how many pixels
    of each intent type occur inside the patch.
- A low-resolution intent raster that captures where brand, flat, and gradient pixels are
    located inside the patch (for example, a coarse 4 by 4 grid with three channels).
- A fixed lambda triplet: 0.1 (safe), 0.5 (balanced), 0.9 (saturated).
- A deterministic seed policy that provides one seed per patch and candidate for proposer
    sampling, and one seed per patch for drift sampling.
- Candidate and robustness budgets: base candidate count is 3, maximum candidate count
    is 5, drift sample count is 32, and the risk quantile is 0.90.
**Step 0** - _Compute target appearance for the patch_ - Convert each RGB patch → sRGB → Linear
RGB using the standard sRGB transfer curve, then Convert linear RGB → XYZ using a fixed
conversion matrix. Convert XYZ to LAB using D50 illuminant and the two-degree standard
observer. This finally produces the RGB patch represented in LAB under assumed
spectrographic conditions.
**Step 1** - This step details out building conditioning vectors and for this purpose, we use FiLM
control signals for usage as latent tensors on both the VAE and the Forward Surrogate. FiLM
means that at selected convolution blocks, feature maps are modulated by a learned scale and
shift derived from the unified conditioning vector. Each sub-step details conversion to tensor for
each additional variable
_1.1 - PPP embedding -_ Convert the base family token into a learned embedding. Normalize
numeric overrides into a standard range using fixed per-field min and max values. Use the


override mask as a binary vector indicating which fields were explicitly set. Concatenate base
family embedding, normalized overrides, and override mask, then pass through a small MLP to
produce the PPP embedding.
_1.2 - Structure embedding -_ Convert the structure token into a learned embedding and pass it
through a small MLP to produce the structure embedding.
_1.3 - Intent embedding -_ Start with the intent weight vector for the patch. Add the flattened
low-resolution intent raster. Concatenate these and pass through a small MLP to produce the
intent embedding.
_1.4 - Unified conditioning vector and FiLM generators -_ For each candidate, concatenate the PPP
embedding, structure embedding, intent embedding, and the candidate’s lambda value to form
one unified conditioning vector. Feed this unified vector into FiLM generators that produce
per-block scale and shift parameters for each -

1. The proposer’s encoder blocks.
2. The proposer’s decoder blocks.
3. The surrogate’s convolution blocks.
**Step 2** - _Feed intent into the proposer at generation time_ - Two intent injection paths are used.
_2.1 - Input-channel broadcast for stability -_ Broadcast the intent weights into constant per-pixel
channels and concatenate them to the RGB patch at the proposer’s first layer. For using the
low-resolution intent raster, upsample it to patch resolution and concatenate those channels too.
This makes intent spatially visible at the input.
_2.2 - FiLM injection as the primary conditioning mechanism_ - Use FiLM modulation in multiple
proposer blocks so that PPP, structure, intent, and lambda influence generation decisions
throughout the network.
**Step 3** - _Conditional VAE proposer -_ The proposer is a conditional variational autoencoder with
an encoder, a reparameterization step, and a decoder. The latent variable is sampled from a
standard Normal distribution using the deterministic proposer seed. The encoder is only used
during training; inference uses prior sampling. The decoder produces a CMYKOGV candidate
patch conditioned on the RGB patch, PPP, structure token, intent, lambda, and the sampled latent
(Conditioning achieved through FiLM Injection at proposer convolution blocks)
_3(a) - Inference loop per patch_ - For each of the three lambda values (safe, balanced, saturated):
Build the unified conditioning vector for that lambda value. Sample one latent vector using the
deterministic seed. Run the decoder with FiLM modulation to generate a CMYKOGV candidate
patch. Clamp all CMYKOGV channel values to the range 0 to 1.


**Step 4** - _Two-pass refiner with projection-closed feasibility_ - For each CMYKOGV candidate:
Pass 1: Apply deterministic regularity smoothing and neutral/dark stabilization.
Pass 2: Apply a feasibility projection to guarantee the candidate lies inside the PPP feasibility
envelope. _A typical feasibility projection is defined as follows:_ Clip each channel to its PPP
channel cap. If total ink coverage exceeds the PPP TAC cap, reduce values by a fixed scaling
policy so the TAC cap is met. The refined output is the feasible CMYKOGV candidate used for
robustness testing.
**Step 5** - _Drift sampling from PPP_ - Using the deterministic drift seed for the patch, sample 32
drift instances (32 is an arbitrary number, please confirm through experiments) from the PPP
drift distribution. Each drift instance contains: Per-channel ink-strength multipliers for cyan,
magenta, yellow, black, orange, green, and violet. Add 7 monotone tone curve perturbations for
each channel (7 is an arbitrary number, please confirm through experiments). Drift is applied
channelwise and then clipped to the range 0 to 1.
**Step 6 -** _Forward surrogate (a small patch-aware CNN, FiLM-conditioned) -_ The surrogate is a
small convolutional neural network (exact structure specified in Appendix 2) that predicts a
Lab-like perceptual proxy from CMYKOGV under PPP and drift. The surrogate input uses a
context window: Provide a 32 by 32 CMYKOGV context region. Score only the 16 by 16 center
region. For each feasible candidate and each drift instance: Apply drift to the candidate inside the
context window. Recompute the unified conditioning vector for that candidate’s lambda value.
Run the small CNN surrogate with FiLM modulation to produce the Lab-like proxy for the
center region.
**Step 7** **_-_** _Error and robustness risk (intent-weighted percentile) -_ For each drift instance:
Compute per-pixel DeltaE00 between the surrogate’s Lab-like output and the reference Lab
patch from Step 0. Also, log per-channel ink usage statistics, TAC margin, neutral/dark stability
flags, and a simple spatial regularity score. Aggregate these per-pixel differences into a single
patch error using the patch intent weights so brand, flat, and gradient regions contribute
according to the weights. Across the 32 drift samples, compute the robustness risk as the 0.
quantile (in practice, consider testing with different values until satisfied) of the error
distribution. Also, compute a nominal error using identity drift for tie-breaking.
**Step 8** - _Robustness-first gating, selection, and escalation_ - Discard any candidate that fails
feasibility after refinement. Discard any candidate whose robustness risk exceeds the configured
threshold (defined by PPP or policy). Select the feasible candidate with minimum risk; break ties
using nominal error. If no candidate survives: Increase the candidate count first, up to the
maximum of 5, and rerun proposal, refinement, drift evaluation, and selection. Only if coverage
\escalation fails, increase the drift sample count as a rare fallback, and log that it happened.


Compute allocation can be conditioned on intent weights, for example by giving more escalation
budget to patches with high brand or high gradient weight.
**Step 9** - _Outputs and diagnostics (per patch)_ - Output the selected CMYKOGV patch. Cache all
diagnostics needed for exact replay and audit which necessarily includes the following:

1. PPP base family and override mask, and the effective feasibility and drift parameters
    used.
2. The structure token and intent weights (and the intent raster if used). The selected lambda
    value.
3. The candidate and drift budgets used, including any escalation.
4. All seeds for generating stochasticity
5. Risk and nominal error values.
6. Failure reasons for rejected candidates.
7. Constraint tightness indicators.
8. Flags indicating any fallbacks.

# Training -

### Conditional - VAE Proposer (7-channel) - For the supervision source, Generate synthetic

CMYKOGV targets for training from CMYK by a deterministic two-stage constrained solve.
The CMYK baseline comes from ICC separation of the RGB patch using Relative Colorimetric
intent with Black Point Compensation. Initialize the 7-channel target as the CMYK baseline for
C, M, Y, K and zeros for O, G, V.
**Stage 1** - _Nominal solve objective_ - Use projected gradient descent with feasibility projection
after every step. The objective combines:

1. A primary appearance term that matches the patch’s Lab reference A* (computed from
    RGB via the locked sRGB to Lab pipeline)
2. An appearance anchor term that keeps the solution close to the ICC CMYK baseline
    appearance
3. An OGV usage penalty that discourages using O, G, V unless it helps
4. A neutral or dark preservation penalty that activates only when A* indicates the patch is
    neutral and dark under PPP-stored Lab thresholds.
**Stage 2 -** _Drift refinement objective_ - Starting from the Stage 1 solution, run a fixed number of
projected gradient steps that primarily reduce 0.90-quantile drift risk under drift samples, while
applying a channel-weighted trust-region penalty that discourages moving far from the Stage 1
solution, with stronger penalties on O, G, V than on C, M, Y, K. Feasibility projection is applied
after every step.


**Stage 3** - _Hybrid scoring rule for target generation_ - Before the surrogate passes the quality gate,
all appearance evaluations inside the solver use ICC only. After the gate, Stage 1 appearance
evaluations become a PPP-specific weighted blend of ICC and surrogate (with alpha_icc stored
per base family), and Stage 2 drift-risk refinement uses the surrogate primarily while retaining
ICC as an anchor as specified by the blend.
**Stage 4** - _Proposer architecture and conditioning_ - Train a conditional VAE with FiLM
modulation in encoder blocks E1 to E4 and decoder blocks D1 to D5. Feed PPP embedding,
structure embedding, intent embedding, and lambda into FiLM generators. Intent also enters at
the input via intent broadcast channels (and optionally an upsampled intent raster).
**Stage 5** - _Proposer losses_ - Use a standard VAE objective with three parts: ink-space
reconstruction loss, appearance-space auxiliary loss, and KL regularization. Ink reconstruction is
Smooth L1 between predicted CMYKOGV and the synthetic CMYKOGV target, averaged over
pixels and channels. Appearance auxiliary is a smaller loss comparing predicted appearance of
the predicted CMYKOGV to predicted appearance of the target CMYKOGV using the current
teacher engine (ICC early, then the ICC–surrogate blend after the gate). KL uses a deterministic
warm-up schedule where its weight ramps linearly from zero to its maximum over the first fixed
fraction of training steps, then stays constant.
**Stage 6** - _Lambda-structured semantics_ - Enforce that OGV usage increases with lambda using
paired latent samples. For each training patch, draw one latent vector z, decode three outputs at
the three lambda values, compute patch-mean O+G+V for each, and add hinge penalties when
the ordering is violated so OGV is non-decreasing from safe to saturated.
The next natural step is to decide on the remaining numeric hyperparameters that are still
implicit here (for example: iteration counts and step sizes for stage 1 and stage 2 projected
gradient descent, the relative weights of the objective terms, and the exact PPP fields that store

### the neutral and dark thresholds and the quality gate thresholds).

### Forward Surrogate - Training data unit is structured such that each example is a 32 by 32

CMYKOGV context window with a designated scored center 16 by 16 region, + PPP, structure
token, intent weights, intent raster, and a drift instance.
Until measured seven-channel press characterization exists, the surrogate is trained as a local
appearance-delta model rather than a global renderer. The model output is
\(\Delta Lab\), and the rendered prediction used for scoring is \(Lab_\mathrm{ref}+\Delta Lab\).
This is necessary because the current ICC/proxy teacher is anchor-calibrated: it predicts local
appearance changes around a known reference Lab, not absolute press Lab from CMYKOGV alone.
**Stage 1** - _Teacher labels from ICC/proxy_ - For each sampled example, store separate
`lab_ref_center`, `teacher_lab_nominal`, and `teacher_lab_drifted` arrays. Nominal uses identity
drift; drifted applies exactly one PPP-defined ink-strength/TRC perturbation before rendering.
The v2 shard schema is `surrogate-shards-v2`; earlier v1 surrogate shards are structurally
readable but semantically obsolete for training or hybrid target generation.
**Stage 2** - _Conditioning and FiLM_ - Feed PPP embedding, structure embedding, intent
embedding, lambda, and a drift embedding into FiLM heads that modulate surrogate convolution


blocks 1 through 7, using the same “Conv then GroupNorm then FiLM then activation” rule used
in the proposer.
**Stage 3** - _Loss_ - Train the surrogate to match local teacher deltas:
\(\Delta Lab_\mathrm{nominal}=teacher\_lab\_nominal-lab\_ref\_center\) and
\(\Delta Lab_\mathrm{drifted}=teacher\_lab\_drifted-lab\_ref\_center\). Use Smooth L1
(Huber) loss, averaged over pixels and Lab channels, with symmetric nominal and drifted
supervision.
**Stage 4** - _Determinism_ - All sampling is deterministic via the seed policy, sampling of patches
and contexts, sampling of the single drift instance per example, and any shuffling is seeded and
logged.
**Stage 5** - _Quality gate to unlock hybrid target generation_ - Evaluate on a fixed heldout set using
a tougher deterministic candidate probe protocol: per patch, generate 5 candidates at lambda
values 0.0, 0.3, 0.6, 0.9, 1.0 using deterministic seeds. Score each candidate under a fixed drift
bank of 32 drift samples that is precomputed once per PPP base family using a frozen seed.
The surrogate training distribution must include the same lambda candidate family, OGV
perturbations, projected candidates, neons, neutrals, darks, and smooth ramps used by the gate.
Compute mean DeltaE00, 0.90-quantile DeltaE00, Spearman rank correlation, top-1 agreement,
and regret \(Risk_\mathrm{teacher}(argmin_\mathrm{surrogate})-\min Risk_\mathrm{teacher}\).
The gate passes only if all accuracy, ranking, and regret metrics beat PPP-stored thresholds.


## Appendix 1 - Glossary

```
● FiLM Modulation - Feature-wise linear modulation, meaning a method to control a
neural network using extra inputs. A conditioning vector (a single list of numbers that
summarizes PPP, structure, intent, and lambda) is passed through a small network to
produce per-channel scale and shift values (two numbers for each feature channel), and
these values modify intermediate feature maps (the internal multi-channel arrays
computed inside a CNN) after normalization (a step like GroupNorm that rescales
activations to a standard range) and before the nonlinearity (an activation function such
as SiLU that makes the network expressive).
● MLP - Multilayer perceptron, meaning a small fully connected neural network made of
stacked linear layers with nonlinear activations. It takes one vector (a list of numbers) as
input and outputs another vector, and we use it to turn raw conditioning inputs (PPP
fields, structure token, intent weights) into embeddings (compact learned vectors) and to
generate FiLM scale and shift parameters.
● PPP - Per-Patch Profile, meaning a compact description of print conditions and
constraints for a patch. It is a two-tier JSON object containing a base_family token (a
categorical label like “film_generic_conservative”) plus numeric overrides (numbers such
as caps or limits) and an override mask (a set of on/off flags telling which overrides were
explicitly provided). PPP defines the feasibility envelope (the set of ink patches that are
allowed because they respect caps like TAC and per-channel limits) and the drift
distribution (a probability rule that generates realistic process perturbations for robustness
testing).
● RGB → sRGB → Linear RGB - RGB are three numbers per pixel representing red,
green, and blue. sRGB is the standard encoding where those numbers are
gamma-compressed (stored in a non-linear form to match display behavior). Linear RGB
means the same three channels after we undo that compression by applying the standard
sRGB inverse transfer curve per channel, so the values are proportional to linear light
intensity (physical brightness).
● Linear RGB → XYZ → LAB (U-D50 & 2-D-SO) - XYZ is a standard color space
defined by the CIE where colors are represented by three coordinates that are
device-independent. We compute XYZ from linear RGB using a fixed 3 by 3 matrix (a
constant linear transform). LAB, also called CIELAB, is a perceptual color space where
distances correlate better with human-visible differences. We convert XYZ to LAB using
D50 (a standard reference daylight white point used in printing) and the two-degree
standard observer (a standard model of human color matching for a small field of view).
The resulting LAB patch is our reference appearance for scoring.
● Reparameterization - A trick used in training a variational autoencoder so sampling
stays differentiable (meaning gradients can pass through it for learning). Instead of
sampling the latent directly, the model predicts a mean and variance (parameters of a
Normal distribution), we sample a noise vector, and then we compute the latent as mean
```

plus noise times standard deviation, which keeps the computation smooth for
gradient-based optimization.
● **Drift Sampling** - A procedure to simulate real-world variability during robustness
testing. We draw random perturbations from a PPP-defined distribution (a probability
model specified by PPP), such as ink-strength multipliers (per-channel scaling factors
that model ink being slightly stronger or weaker) and tone-curve perturbations (small
changes to how input coverage maps to printed tone), then apply these to the candidate
ink patch before scoring.
● **Micro-CNN** - A small convolutional neural network, meaning a lightweight CNN with
relatively few layers and channels. “Convolutional” means it processes images using
local filters that slide over the grid. It is “patch-aware” because it takes a larger context
window around the patch as input, but the model is evaluated only on the center region,
forcing it to use neighboring context to predict the center correctly.
● **Conditional VAE** - A variational autoencoder that is conditioned, meaning its output
depends not only on a random latent vector but also on observed inputs and control
signals. Here the observed input is the RGB patch, and the control signals are PPP, the
structure token, intent information, and lambda. Conditioning is implemented with FiLM
(feature-wise scale and shift inside CNN blocks) and optionally with extra input channels
(such as intent broadcast channels added to the RGB input).
● **CMYKOGV** - A seven-ink channel representation used in expanded-gamut printing. The
channels are cyan, magenta, yellow, black, orange, green, and violet, and each channel
value is a normalized coverage between 0 and 1 indicating how much of that ink is laid
down at a pixel.
● **Conditioning Vector** - A single vector (a list of numbers) created by concatenating
embeddings (learned vectors) for PPP, structure, and intent, plus the lambda value (a
scalar control that moves the system from safer to more saturated solutions). This vector
is fed into MLPs that produce FiLM parameters, so it controls both the proposer and the
surrogate.
● **Monotone Tone Curve Perturbation** - A change to a tone reproduction curve (a
function that maps an input ink coverage to an effective printed tone) that is monotone,
meaning it never reverses order (if input A is larger than input B, the output remains
larger). This models realistic effects like dot gain or curve shifts while avoiding
non-physical behavior like inversions.
● **Latents & Latent Space** - The latent is the hidden random vector sampled for the VAE
decoder that enables multiple diverse candidates for the same RGB patch and
conditioning. The latent space is the set of all possible latent vectors along with their
sampling distribution (in inference, typically a standard Normal distribution), and
changing the latent sample changes the proposed CMYKOGV candidate while keeping
the conditioning fixed.


● **TAC Cap** - Total Area Coverage cap, meaning a hard printability constraint that limits
the total ink laid down at a pixel. It is computed as the sum of all ink channel coverages
at that pixel (over four channels for CMYK or seven channels for CMYKOGV), and it is
enforced by feasibility projection so the final selected patch is physically admissible for
printing.
● **ICC Profile** - An ICC profile is a standardized file that describes how a specific device
or process (a printer, an ink set, a substrate, or a display) maps between color numbers
and perceived color, so we can simulate “what the print will look like” from CMYKOGV
or compute a “best-effort” CMYK separation from RGB in a repeatable way.
**● Smooth L1 (Huber) loss -** Smooth L1, also called Huber loss, is an error measure used
during training that behaves like squared error when the prediction is close to the target
(to encof (x, y) = y 2 , R is the triangular region with vertices (0, 1), (1, 2), (4, 1)urage precision) and behaves like absolute error when the prediction is far from
the target (to avoid one bad sample dominating the update), where “target” means the
value we want the model to match.
**● Deterministic Seed -** A deterministic seed is a fixed integer used to initialize a
pseudo-random number generator (a program that produces repeatable “random-looking”
numbers) so that sampling steps like drawing a VAE latent vector or drawing drift
perturbations can be exactly reproduced later given the same seed.
**● DeltaE00 -** DeltaE00 (also written as Delta E 2000) is a standard way to measure how
different two colors look to a human, computed from their Lab values (Lab is a color
space designed so numeric differences roughly match perceived differences), where a
lower DeltaE00 means the colors look more similar.
**● Spearman Rank Correlation -** Spearman rank correlation is a number between minus
one and one that measures how similarly two methods order the same set of items, where
“order” means ranking candidates from best to worst, so in our gate it checks whether the
surrogate ranks candidates in the same order as the ICC teacher even if the exact error
values differ.
**● Convolution Blocks -** Convolution blocks are repeated building units inside a CNN (a
convolutional neural network, meaning a model that processes images using small sliding
filters) that typically include a convolution layer (a filter operation), a normalization layer
(a scaling step that stabilizes training), and a nonlinearity (an activation function that lets
the network model complex patterns), and these are the places where we apply FiLM
conditioning.
**● Ink-space reconstruction loss -** Ink-space reconstruction loss is the training error
computed directly on the predicted ink channels, meaning we compare the model’s
predicted CMYKOGV numbers to the target CMYKOGV numbers per pixel and channel
and penalize differences, so the model learns to output the right ink values rather than
only matching appearance indirectly.
**● Appearance-space auxiliary loss -** Appearance-space auxiliary loss is an additional
training error computed after converting a predicted CMYKOGV patch into a predicted


```
appearance (for us, a Lab-like patch predicted by the teacher engine such as ICC and later
a blended ICC–surrogate), then comparing that predicted appearance to the target
appearance, so the model is pushed to match what the print would look like, not just the
raw inks.
● KL regularization - KL regularization is the part of VAE training that pushes the latent
code distribution produced by the encoder (the encoder maps inputs to a probability
distribution over latents) to stay close to a simple reference distribution, usually a
standard Normal distribution (a bell curve with mean zero and variance one), so that at
inference we can sample latents from that reference and still get valid outputs.
● Relative Colorimetric intent with Black Point Compensation - elative Colorimetric
intent is an ICC conversion setting that maps colors so that in-gamut colors (colors the
printer can reproduce) are matched as closely as possible while out-of-gamut colors are
clipped to the nearest reproducible boundary, and Black Point Compensation is an
additional setting that adjusts the mapping so the darkest black of the source space and
the darkest black of the destination space align, which helps preserve shadow detail and
avoids crushed blacks when converting between RGB and a print space.
```
## Appendix 2 - Micro-CNN (as forward surrogate for CMYK → LAB (U-D50 &

## 2-D-SO))

**Surrogate CNN -**
Block 0: 3 by 3 conv, 32 channels, stride 1, dilation 1, group norm, SiLU.
Block 1: 3 by 3 conv, 32 channels, stride 1, dilation 1, group norm, FiLM, SiLU.
Block 2: 3 by 3 conv, 64 channels, stride 1, dilation 2, group norm, FiLM, SiLU.
Block 3: 3 by 3 conv, 64 channels, stride 1, dilation 2, group norm, FiLM, SiLU.
Block 4: 3 by 3 conv, 96 channels, stride 1, dilation 4, group norm, FiLM, SiLU.
Block 5: 3 by 3 conv, 96 channels, stride 1, dilation 4, group norm, FiLM, SiLU.
Block 6: 3 by 3 conv, 64 channels, stride 1, dilation 1, group norm, FiLM, SiLU.
Block 7: 3 by 3 conv, 32 channels, stride 1, dilation 1, group norm, FiLM, SiLU.
Head: 1 by 1 conv to 3 channels (local \(\Delta Lab\)), linear output.
**FiLM placement** : apply FiLM after group norm and before SiLU in Blocks 1 through 7.
**FiLM conditioning inputs** : PPP embedding plus structure embedding plus intent embedding
plus lambda, and also the per-drift parameters for that evaluation (because the spec treats drift as
an explicit input to the surrogate).

## Appendix 3 - FiLM Modulation Placement and Conditional-VAE Structure


**Generic FiLM placement rule** - Inside a FiLM-enabled block, run Conv, then GroupNorm, then
apply FiLM (scale and shift), then apply the activation (SiLU). Do not FiLM the very first input
convolution and do not FiLM the final output head. For every FiLM-enabled block, you need
one FiLM head that maps the conditioning vector to a pair of vectors of length equal to the
channel count of that block: one vector for per-channel scale and one vector for per-channel shift.
The scale should be applied as “1 + Scale” to keep initialization stable. Each block has its own
FiLM head (do not share heads across blocks).
**Conditional VAE Proposer -** Inputs to proposer -

- RGB patch, 16 by 16.
- Intent broadcast channels concatenated at input (the constant per-pixel intent weights, and
    the upsampled low-resolution intent raster)
- A conditioning vector is built from PPP embedding + structure embedding + intent
    embedding + the candidate’s lambda value. This conditioning vector feeds all FiLM
    generators for encoder and decoder.
**Encoder (produces latent distribution parameters during training, not used during
inference)**
E0: Conv 3 by 3, 32 channels, stride 1, GroupNorm, SiLU. No FiLM in E0.
E1: Conv 3 by 3, 32 channels, stride 2, GroupNorm, FiLM, SiLU.
E2: Conv 3 by 3, 64 channels, stride 1, GroupNorm, FiLM, SiLU.
E3: Conv 3 by 3, 64 channels, stride 2, GroupNorm, FiLM, SiLU.
E4: Conv 3 by 3, 96 channels, stride 1, GroupNorm, FiLM, SiLU.
Then we global average pool over the 4 by 4 feature map to a vector and then, two linear heads
from that pooled vector: one head outputs latent mean, one head outputs latent log variance. No
FiLM on these two heads.
**Decoder (used in inference. generates CMYKOGV candidate patch)**
D0: Take the sampled latent vector z and pass through a linear layer to 96 by 4 by 4, then
reshape. No FiLM on this linear layer.
D1: Conv 3 by 3, 96 channels, stride 1, GroupNorm, FiLM, SiLU.
D2: Upsample 2x (to 8 by 8), then Conv 3 by 3, 64 channels, GroupNorm, FiLM, SiLU.
D3: Conv 3 by 3, 64 channels, stride 1, GroupNorm, FiLM, SiLU.
D4: Upsample 2x (to 16 by 16), then Conv 3 by 3, 32 channels, GroupNorm, FiLM, SiLU.
D5: Conv 3 by 3, 32 channels, stride 1, GroupNorm, FiLM, SiLU.
Head: Conv 1 by 1 to 7 channels (CMYKOGV). No FiLM on the head. Apply sigmoid, or
clamp to 0 to 1.
**Summary of which proposer blocks are FiLM-modulated -**
Encoder: E1, E2, E3, E4 are FiLM-modulated. E0 is not.

## Decoder: D1, D2, D3, D4, D5 are FiLM-modulated. D0 and the output head are not.


## Appendix 4 - MLP Structure

**4.1 - Embedding lookups -**
Base family token embedding. Use a learned embedding table that maps each PPP base family
category to a fixed-length vector (an embedding is a trainable lookup vector), set base family
embedding dimension = 32. Structure token embedding - Use a learned embedding table that
maps the structure token (edge, flat, textured) to a vector, set structure token embedding
dimension = 8.
**4.2 - Small MLP blocks for PPP, structure, intent embeddings**
Use the same 2-layer MLP pattern everywhere (an MLP is a stack of fully connected linear
layers, where a linear layer is a weight matrix plus bias): Linear → SiLU → Linear, where SiLU
is a smooth activation function that adds nonlinearity.

1. PPP MLP - Input is concatenation of base family embedding, normalized numeric
    overrides, and the override mask; output ppp embedding dimension = 128.
2. Structure MLP - Input is structure token embedding; output structure embedding
    dimension = 32
3. Intent MLP - Input is concatenation of intent weights plus the flattened low-resolution
    intent raster (example given is 4 by 4 grid with 3 channels, so 48 raster numbers, plus 3
    intent weights), output intent embedding dimension = 128.
**4.3 - Unified conditioning vector dimension**
For proposer conditioning, define conditioning as the concatenation of PPP embedding,
structure embedding, intent embedding and lambda, so conditional dimensions in total add up to
128 + 32 + 128 + 1 = 289.
**4.4 FiLM-head MLPs (per FiLM-enabled block)**
Each FiLM-enabled convolution block gets its own FiLM head (a small MLP that outputs
per-channel scare and shift vectors) and heads are not shared across blocks.
Define one FiLM head per block i with channel count C_i as: Linear(cond_dim → 128) → SiLU
→ Linear(128 → 2*C_i), then split into scale_i and shift_i of length C_i; apply scale as 1 +
scale_i (meaning “identity plus learned delta” to keep initialization stable).
**4.5 Surrogate-only conditioning includes drift parameters**
Drift sampling uses 7 ink-strength multipliers for cyan, magenta, yellow, black, orange, green,
violet, and additionally 7 monotone tone curve perturbation parameters per channel (monotone
means the tone curve never inverts ordering). Because drift is treated as an explicit input to the
surrogate, the surrogate FiLM heads take a conditional vector on the surrogate which is a
concatenation of conditional vector (general) and drift parameters With the defaults above, the
dimensions come to be a total of 289 + 56 = 345.

