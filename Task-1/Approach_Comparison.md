# Approach Comparison: Hybrid vs. Cloud-Only (two prompt versions)

Three result files are compared here, all adjudicating the identical
3,096-item match pool and 23,132-item parse pool, all fully resolved (0
unprocessed/failed rows), so every difference is attributable to the
model/prompt change, not missing data:

- `Item_Mapping_Result_Hybrid.xlsx` — local `gemma4:e2b` + cloud
  `gemma4:31b-cloud` escalation, original prompt.
- `Item_Mapping_Result_CloudOnly_BestResult.xlsx` (**recommended**) —
  `gemma4:31b-cloud` only, prompt refined after auditing the hybrid run.
- `Item_Mapping_Result_CloudOnly_PromptV2_EXPERIMENTAL.xlsx` (**not
  recommended**) — same as BestResult, plus four further prompt
  refinements. Testing found real improvements *and* a new regression
  pattern (see "PromptV2 experiment" section below) — kept in this folder
  for transparency about what was tried, not because it's the better file.

The first two are compared below; the PromptV2 experiment has its own
section further down.

## Overall agreement

Across all 5,709 pharmacy items (both sheets combined):

| | Count | % |
|---|---|---|
| Identical Final_Status | 5,192 | 90.9% |
| Different Final_Status | 517 | 9.1% |

Rows that never reach an LLM (`AUTO_ACCEPTED`, pure fuzzy-score matches
≥90; `NO_MATCH`, scores <70) are **always identical** between the two runs
— those tiers are decided before either model is involved. All 517
disagreements are concentrated in the 3,096-item ambiguous pool that
actually reaches an LLM (83.3% agreement within that pool):

```
cloud_status   AUTO_ACCEPTED  LLM_CONFIRMED  LLM_REJECTED  NO_MATCH
hybrid_status
AUTO_ACCEPTED           1667              0             0         0
LLM_CONFIRMED              0           2116           331         0
LLM_REJECTED               0            186           463         0
NO_MATCH                   0              0             0       946
```

## Which one is right? Manual audit of disagreements

### Bucket 1: hybrid CONFIRMED → cloud REJECTED (331 items)

Manually checked an 18-item random sample against actual product identity:
**cloud was correct in 18/18.** Every case was a genuine hybrid error —
overwhelmingly brand-name substitution (confirming a different brand as
the same product) or a hidden strength/dosage-form mismatch behind a
shared brand:

| Source | Hybrid wrongly confirmed | Problem |
|---|---|---|
| `ALLEVO 5 MG 20 TAB` | `DESA 5 MG 20 TAB` | Completely different brand, same strength/pack |
| `CATAPRESAN 150 MG 30 TAB` | `ALKAPRESS 10 MG 30 TAB` | Different brand **and** different strength (150mg vs 10mg) |
| `HAEMOJET 36 CAP` | `HAEMOJET 100MG IM 6 AMP` | Same brand, different dosage form (capsules vs. injection) |
| `POVIDIN ANTISEPTIC 10% 100ML` | `BETADINE 10% ANTISEPTIC SOL 200ML` | Generic-style name matched to an unrelated brand |

### Bucket 2: hybrid REJECTED → cloud CONFIRMED (186 items)

Manual sample (18 items): **cloud correct in ~14-15/18.** Mostly pack-size
or pure-formatting gaps the original prompt over-penalized:

| Source | Cloud correctly confirmed | Note |
|---|---|---|
| `BRAND SEAS SYRUP 120 ML` | `BRAND SEAS SYRUP 120ML` | Literally identical, only a space — hybrid rejected this |
| `EPICOTIL 20 MG 5 SUPP` | `EPICOTIL 20MG 5 SUPP` | Same, spacing only |
| `VIAGRA 50 MG 8 TAB` | `VIAGRA 50MG 4 TAB` | Same brand+strength, pack count differs |

A mechanical audit across the **full** 186-item bucket (flagging candidates
with a missing size, a large numeric gap, or a differing size/variant
qualifier word like NEWBORN/MINI/KIDS) flagged 73/186 (39%) as worth a
second look. That's an upper-bound candidate list, not a confirmed error
count — spot-checking shows real false-alarm noise (e.g. `PENCITARD
1.2MIU VIAL` → `PENCITARD 1200000U 1 VIAL` got flagged for a "huge number
gap" purely because 1.2 MIU *is* 1,200,000 units, a correct match). The
more reliable subset is the 12 "variant word mismatch" cases, e.g.:

- `MOLFIX NO 2 (10) DIAPERS` → confirmed against `MOLFIX 3D SMALL MINI 10
  DIAPERS` — "NO 2" and "SMALL/MINI" are different diaper sizes, not
  interchangeable. Likely a genuine new cloud-only error.

### Verdict

Weighting both buckets by size, cloud-only correctly fixed an estimated
90%+ of the 517 disagreements, at the cost of a small new risk (roughly
12-20 items out of 186) where the relaxed pack-size tolerance may have gone
slightly too far. **Cloud-only is the better-performing approach** — it
resolves a systematic, high-volume brand-substitution error class at the
price of a much smaller, narrower new error class.

## Why cloud inference for an open-weights model

`gemma4:31b` is an open-weights model — it isn't a proprietary closed
model, and it's not tied to the cloud by design. It was accessed through
Ollama's cloud inference API purely because the available hardware (a
laptop GPU with 6GB VRAM) can't run a 31B-parameter model locally; running
`gemma4:e2b` (5.1B) locally was already at the edge of what that hardware
could hold. Given a machine with enough VRAM, `gemma4:31b` could run
entirely on-device, with the same prompt and same accuracy, and no
dependency on network connectivity or Ollama's cloud usage caps (the
session-limit issues documented during this project's runs). The choice
here was a hardware-availability workaround, not an architectural
preference for cloud/closed inference.

## Why Gemini isn't part of this submission

A third LLM provider option — Google Gemini — was also implemented and
tested for the adjudication step. It could not be run to completion:
**Gemini's free tier caps `gemini-2.5-flash` at 20 requests/day per API
key.** This workload needs roughly 1,750 requests total (26,228 items ÷ 15
per batch), which would require on the order of **90 different free-tier
API keys**, or a paid Gemini account with materially higher quotas, to
complete in a reasonable timeframe. Neither was available, so this path
was set aside — not because the approach doesn't work, but because of an
external resource constraint unrelated to matching quality. (Ollama's
`gemma4:31b-cloud`, used in the recommended result, has its own usage caps
too, but they reset frequently enough to complete the full workload within
a single working session across a couple of retry passes.)

## PromptV2 experiment: tested, not adopted

`Item_Mapping_Result_CloudOnly_BestResult.xlsx`'s original prompt had four
known gaps identified from the audit above (originally written up as
speculative "future work" — since tested for real, see below):

1. **Unit-aware number comparison** — `1.2 MIU` vs `1,200,000 U` are the
   same value but look like a huge mismatch to naive comparison.
2. **Don't relax pack-size tolerance when the candidate is missing size
   info entirely**, as opposed to stating a *different* size.
3. **Named variant/size qualifiers should block a match** (`NEWBORN` vs
   `MINI`, etc.) even if pack size otherwise lines up.
4. **Generic/category-sounding brand tokens deserve lower confidence**
   (`GREEN TEA` matched to `GREEN TEA` on a category name, not a brand).

All four were added to the prompt as explicit rules, validated against the
exact known-wrong cases (7/7 fixed, zero regressions on that small check),
then run against the full 3,096-item match pool —
`Item_Mapping_Result_CloudOnly_PromptV2_EXPERIMENTAL.xlsx`. **Result: 0
`LLM_FAILED`, but a mixed quality outcome on closer audit — not adopted as
the recommended file.**

Of 3,092 comparable decisions, 208 were cosmetic (CONFIRM vs
CORRECTED_MATCH label flips on the identical chosen match — no effect on
`Final_Status`) and 108 were substantive changes. Manually auditing a
non-exhaustive read of ~45 of those 108:

**Confirmed real wins** — the targeted fixes working as intended:
- `MOLFIX NO 2 (10) DIAPERS` now correctly rejects `...NEWBORN 60+2...`
  (exactly the case rule 3 was written for)
- `RH RHOPHYLAC 300MG...` — v1/BestResult had matched a `300MCG` candidate
  (a 1000x unit error); PromptV2 correctly matches the `300MG` candidate
- `SULPERAZON 1.5 MG` — BestResult wrongly confirmed a `1.5GM` candidate
  (1000x mismatch); PromptV2 correctly rejects

**A new regression pattern found**, concentrated in rule 3 (variant
qualifiers):
- `ANISOL 20CAP` — BestResult correctly confirmed `ANISOL 14CAP` (same
  brand, pack-size-only difference); **PromptV2 wrongly rejects it**
- `MOVICOL ADULT 20 SACHETS` — BestResult correctly confirmed bare
  `MOVICOL 20 SACHET`; **PromptV2 wrongly rejects it**
- `ALECO CARE CREAM 60GRAM` — same pattern, wrongly rejected over a
  pack-size-only gap

The likely cause: rule 3 says "reject if input and candidate have
*different* qualifier words" but the model appears to also reject when
*only one side states a qualifier at all* (e.g. source says "ADULT",
candidate says nothing) — broader than intended. Found 3+ clear instances
in the partial sample, so this is a systematic effect, not a one-off.

**One more inconsistency**: `PALMERS COCONUT 150GM` got confirmed against
a `150 MG` candidate — the exact class of 1000x-unit-mismatch error rule 2
was built to catch, and which it *did* catch for RHOPHYLAC and SULPERAZON
above. The unit-awareness rule isn't 100% reliably applied.

**Verdict: not confirmed better than BestResult.** Real wins in specific
targeted patterns, offset by a new, systematic regression in the
size-qualifier rule. Recommended next step before promoting this file:
tighten rule 3 to only block when both sides state *conflicting*
qualifiers, not when one side is simply silent (that case should fall
under the normal pack-tolerance rule instead) — then re-audit the specific
regression cases (`ANISOL`, `MOVICOL`, `ALECO CARE`) before re-comparing
against BestResult.

## Further future work

1. Apply the rule-3 tightening above and re-validate before considering
   PromptV2 (or a v3) as a replacement for BestResult.
2. If a paid Gemini account or a batch of keys becomes available, the same
   refined-prompt approach should be tried there too, as a fourth data
   point on the same identical ambiguous pool.
