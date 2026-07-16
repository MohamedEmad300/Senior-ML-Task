# Column Reference

Explains every column in every sheet. Both result workbooks
(`Item_Mapping_Result_Hybrid.xlsx` and
`Item_Mapping_Result_CloudOnly_RECOMMENDED.xlsx`) share the identical
sheet/column structure — only the values inside differ (see
`Approach_Comparison.md`).

## Item Master - Cleaned

All 60,936 original rows, plus the 5 fields extracted from `ITEM_LOOKUP_NAME`.

| Column | Meaning |
|---|---|
| `ITEM_LOOKUP_NAME` | The original raw item name (whitespace/tab-stripped) — the row's identifier |
| `Trade Name` | Brand/product name with dosage form, pack size, strength, flavour, and promo noise (`///OFFER`, `20%OFF`, etc.) stripped out |
| `Dosage Form` | Detected pharmaceutical form (Tablet, Capsule, Syrup, Cream, ...), or blank if none was recognized |
| `Pack Size` | The numeric count/volume extracted (e.g. `24` for "24 Tablets", `120` for "120ML") |
| `Unit of Measure` | The unit paired with the dosage form (e.g. "Tablet", "Bottle", "ML", "GM") |
| `Flavour` | Flavour keyword detected (Orange, Mint, ...), or blank |
| `Parse_Confidence` | 0.0–1.0 score from the *regex* parser only (+0.4 form found, +0.3 pack size found, +0.3 usable trade name) — reflects the regex pass, not whether the LLM later filled it in |
| `Needs_LLM_Review` | `True` if `Parse_Confidence < 0.7`, meaning this row was routed to the LLM for field extraction |
| `Parse_Source` | Where the final field values actually came from: `Regex` (confidence was high enough) or `LLM` (regex wasn't confident, LLM filled it in) |

## Pharmacy 1 - Matched

4,709 items reshaped from the raw block export, plus match results.

| Column | Meaning |
|---|---|
| `Item_Name` | Raw item name from the pharmacy export |
| `Name_Normalized` | Uppercased/punctuation-stripped/whitespace-collapsed version used internally for matching — not meant to be human-facing |
| `Quantity` | Quantity from the item's data block |
| `Unit` | Packaging unit (e.g. box) |
| `Manufacturer` | Manufacturer/category field |
| `Item_Code` | The pharmacy's internal item code |
| `Location_Path` | Raw shelf/category path, e.g. `Pharmacy > Creams & Ointments > a > a1` — kept as raw metadata, not a clean taxonomy |
| `Category_Hint` | Just the 2nd segment of `Location_Path` as a soft category guess — blank if the path was too short |
| `Expiry_Date` | Expiry date |
| `Batch` | Batch/lot code |
| `Purchase_Date` | Purchase date |
| `Sale_Price` | Sale price |
| `Cost_Price` | Cost price |
| `Supplier` | Supplier name |
| `Supplier_Code` | A *different*, smaller code — distinct from `Item_Code`, structurally looks like a supplier/batch reference |
| `Best_Match_Name` | The Item Master `ITEM_LOOKUP_NAME` this item was matched to |
| `Best_Match_Score` | The similarity score (0–100) between this item and its match — this is the *retrieval* score, identical between both result files since it comes from the shared retrieval stage, not the LLM |
| `Tier` | `HIGH_AUTO_ACCEPT` (score ≥90), `MEDIUM_LLM_REVIEW` (70–89), or `LOW_NO_MATCH` (<70) — assigned before any LLM involvement |
| `Final_Status` | The actual outcome after LLM adjudication: `AUTO_ACCEPTED` (tier was HIGH, no LLM needed), `LLM_CONFIRMED` (LLM confirmed/corrected the match), `LLM_REJECTED` (LLM said no real match), or `NO_MATCH` (tier was LOW, never sent to the LLM by design) |

## Pharmacy 2 - Matched

1,000 items, already tabular in the source, plus match results.

| Column | Meaning |
|---|---|
| `Internal Reference` | Pharmacy 2's own numeric item ID |
| `Product English Name` | Raw English product name |
| `Product Arabic Name` | Raw Arabic product name |
| `Product Type` | Source field, e.g. "Service", "Storable" |
| `Barcode` | Only 31.3% of rows have one |
| `Generic Name` | Active-ingredient name — only 10.2% filled |
| `Manufacture` | Manufacturer (100% filled) |
| `Origin Type` | e.g. "Local", "Imported", "Other" |
| `Unit Of Measure` | Pharmacy 2's own packaging descriptor (e.g. "Box [3 Strip * 10 Tablet]") — **not** the same concept as Item Master's derived `Unit of Measure`, this is the source's own field |
| `Maximum Discount` | Source field, discount % allowed |
| `Tracking By` | Source field, e.g. "lot"/"By Lots" |
| `Name_Normalized` | Matching key, built from `Product English Name` (falls back to `Product Arabic Name` if English is blank) |
| `Best_Match_Name` / `Best_Match_Score` / `Tier` / `Final_Status` | Same meaning as the Pharmacy 1 columns above |

Note: the source's `Product Category` column was dropped entirely — it was
the literal string `"all"` on every one of the 1,000 rows, zero information.

## Summary

Just two columns, `Metric` and `Value` — a flat report, not row-per-item
data. Covers Item Master parse-source counts, each pharmacy sheet's
tier/status breakdown, and LLM call/failure counts.

## Approach Notes

Single column of free text — the write-up of alternatives considered, why
this approach was chosen, and known limitations, as required by the task
brief. See `Approach_Comparison.md` in this folder for the deeper
side-by-side analysis between the two result files.
