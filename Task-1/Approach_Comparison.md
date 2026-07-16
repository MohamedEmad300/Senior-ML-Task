# Approach Comparison: Hybrid vs. Cloud-Only

Compares `Item_Mapping_Result_Hybrid.xlsx` (local `gemma4:e2b` + cloud
`gemma4:31b-cloud` escalation, original prompt) against
`Item_Mapping_Result_CloudOnly_RECOMMENDED.xlsx` (`gemma4:31b-cloud` only,
refined prompt). Both adjudicate the identical 3,096-item match pool and
23,132-item parse pool, both fully resolved (0 unprocessed/failed rows in
either), so every difference below is attributable to the model/prompt
change, not missing data.

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

## Future work / tuning notes

1. **Unit-aware number comparison.** The prompt's strength-matching rule
   has no explicit unit-normalization step — `1.2 MIU` vs `1,200,000 U` are
   the same value but look like a huge mismatch to naive comparison.
2. **Don't relax pack-size tolerance when the candidate is missing size
   info entirely**, as opposed to stating a *different* size. Those are
   different situations (`BRAND SEAS SYRUP 120 ML` vs `120ML` is a safe
   relaxation; `Capry Top soap 80gm` vs a candidate with no weight stated
   at all is a different, riskier situation) and could reasonably get
   different confidence levels.
3. **Named variant/size qualifiers should block a match even if pack size
   otherwise lines up.** Words like `NEWBORN`, `JUNIOR`, `SIZE N`,
   `MINI`/`MAXI`, `KIDS`/`ADULT` usually denote genuinely different SKUs of
   the same brand family — treat a mismatch on these like a strength
   mismatch (always blocking), not like tolerable pack-size noise.
4. **Generic/category-sounding brand tokens deserve lower confidence.**
   `GREEN TEA 60 TAB` matched to `GREEN TEA 27 TAB` on a category name
   rather than a proprietary brand — cap confidence at `medium` when the
   "brand" token is a common/generic word.
5. If a paid Gemini account or a batch of keys becomes available, the same
   refined-prompt approach used for cloud-only should be tried there too,
   as a third data point on the same identical ambiguous pool.
