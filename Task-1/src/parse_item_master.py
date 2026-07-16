"""Step A: extract Trade Name / Dosage Form / Pack Size / Unit of Measure / Flavour
from Item Master's ITEM_LOOKUP_NAME via regex + dictionary lookup.

Real ITEM_LOOKUP_NAME values are messy: many are brand-only with no dosage-form
token at all (e.g. "SYNTOCINON", "BIO CAL"), some carry promo noise
("///OFFER", "20%OFF", "FREE GIFT"), some are non-pharma merchandise (perfume,
shower gel). Rows where we can't confidently extract fields are flagged
Needs_LLM_Review rather than guessed at.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ---------------------------------------------------------------------------
# Dosage form dictionary: canonical name -> (regex alternation, unit of
# measure default, is_measured). "is_measured" forms take a volume/weight
# pack size (ML/GM/...); count forms take a bare integer pack size.
# Ordered so longer/more specific alternatives are tried first within each
# regex (handled by alternation ordering, longest-first).
# ---------------------------------------------------------------------------
DOSAGE_FORMS = [
    # canonical, regex-alternation-of-tokens, unit_of_measure, is_measured
    ("Tablet", r"TABLETS?|TABS?", "Tablet", False),
    ("Capsule", r"CAPSULES?|CAPS?", "Capsule", False),
    ("Effervescent Tablet", r"EFFERVESCENT(?:\s+TAB(?:LET)?S?)?|EFF\s*TAB", "Tablet", False),
    ("Suppository", r"SUPPOSITOR(?:Y|IES)|SUPPS?", "Suppository", False),
    ("Ampoule", r"AMPOULES?|AMPS?", "Ampoule", False),
    ("Vial", r"VIALS?", "Vial", False),
    ("Injection", r"INJECTIONS?|INJ", "Injection", False),
    ("Patch", r"PATCH(?:ES)?", "Patch", False),
    ("Sachet", r"SACHETS?", "Sachet", False),
    ("Lozenge", r"LOZENGES?|LOZ", "Lozenge", False),
    ("Inhaler", r"INHALERS?", "Inhaler", False),
    ("Pen", r"PENS?", "Pen", False),
    ("Suspension", r"SUSPENSIONS?|SUSP", "Bottle", True),
    ("Syrup", r"SYRUPS?|SYR|SYP", "Bottle", True),
    ("Solution", r"SOLUTIONS?|SOL", "Bottle", True),
    ("Emulsion", r"EMULSIONS?", "Bottle", True),
    ("Drops", r"DROPS?", "Bottle", True),
    ("Cream", r"CREAMS?", "Tube", True),
    ("Ointment", r"OINTMENTS?|OINT", "Tube", True),
    ("Gel", r"GELS?", "Tube", True),
    ("Lotion", r"LOTIONS?", "Bottle", True),
    ("Foam", r"FOAM(?:L)?S?", "Can", True),
    ("Shampoo", r"SHAMPOO?S?", "Bottle", True),
    ("Mouthwash", r"MOUTHWASH(?:ES)?", "Bottle", True),
    ("Powder", r"POWDERS?|PWD", "Sachet", True),
    ("Spray", r"SPRAYS?", "Bottle", True),
]

# regex tokens for measured (volume/weight) units, used for pack size on
# is_measured dosage forms.
MEASURED_UNIT_RE = r"(ML|MLS|GM|GRAMS?|G|KG|L|LITT?ERS?)\b"

# strength/concentration units to strip out of the trade name (these are not
# pack size -- e.g. "400MG", "2.5MG", "100 IU", "5/850 MG").
STRENGTH_UNIT_RE = r"(?:MG|MCG|IU|MEQ|MMOL|%)"
STRENGTH_RE = re.compile(
    rf"\b\d+(?:[.,]\d+)?(?:\s*/\s*\d+(?:[.,]\d+)?)*\s*{STRENGTH_UNIT_RE}"
    rf"(?:\s*/\s*\d+(?:[.,]\d+)?\s*(?:ML|MG|MCG|G|GM))?\b",
    re.IGNORECASE,
)

FLAVOURS = [
    "ORANGE", "STRAWBERRY", "MINT", "MENTHOL", "LEMON", "VANILLA",
    "CHOCOLATE", "GRAPE", "APPLE", "BANANA", "HONEY", "CITRUS", "TROPICAL",
    "BUBBLEGUM", "RASPBERRY", "PEACH", "CHERRY", "WATERMELON", "MIXED FRUIT",
    "PINEAPPLE", "MANGO", "CARAMEL", "COFFEE", "ANISE", "LICORICE",
    "SPEARMINT", "PEPPERMINT", "FRUIT", "BERRY",
]
FLAVOUR_RE = re.compile(r"\b(" + "|".join(FLAVOURS) + r")\b", re.IGNORECASE)

# Promo / noise tokens that pollute Trade Name but carry no product meaning.
NOISE_PATTERNS = [
    re.compile(r"///\s*[A-Z0-9]+", re.IGNORECASE),      # ///OFFER, ///ST
    re.compile(r"\d+\s*%\s*OFF\b", re.IGNORECASE),        # 20%OFF
    re.compile(r"\bFREE\s+GIFT\b", re.IGNORECASE),
    re.compile(r"\bOFFER\b|\bOFR\b", re.IGNORECASE),
    re.compile(r"\bNEW\b\s*$", re.IGNORECASE),             # trailing "NEW"
    re.compile(r"\b\d+\s*\+\s*\d+\b"),                    # 2+1 promo
]

# Build one combined regex per dosage form entry for detection + extraction.
_FORM_PATTERNS = []
for canonical, alt, unit, is_measured in DOSAGE_FORMS:
    # branch 1: NUMBER (optional space) FORM  e.g. "30 TAB", "15TAB"
    p_num_before = re.compile(rf"(?P<num>\d+(?:\.\d+)?)\s*(?:{alt})\b", re.IGNORECASE)
    # branch 2: FORM (optional space) NUMBER  e.g. "TAB 20"
    p_num_after = re.compile(rf"\b(?:{alt})\s*(?P<num>\d+(?:\.\d+)?)?", re.IGNORECASE)
    # bare form token match (for stripping out of trade name / detection)
    p_bare = re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)
    _FORM_PATTERNS.append((canonical, unit, is_measured, p_num_before, p_num_after, p_bare))

_MEASURED_PACK_RE = re.compile(rf"(?P<num>\d+(?:\.\d+)?)\s*{MEASURED_UNIT_RE}", re.IGNORECASE)


def detect_dosage_form(name: str):
    """Return (canonical_form, unit_of_measure, is_measured, match) for the
    LEFTMOST dosage-form token found in the string (not the first dictionary
    entry that happens to match anywhere), or None if none found."""
    candidates = []
    for canonical, unit, is_measured, p_num_before, p_num_after, p_bare in _FORM_PATTERNS:
        m = p_num_before.search(name)
        if not m:
            m = p_bare.search(name)
        if m:
            candidates.append((m.start(), canonical, unit, is_measured, m))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    _, canonical, unit, is_measured, m = candidates[0]
    return canonical, unit, is_measured, m


def extract_pack_size(name: str, form_match, is_measured: bool, form_span):
    """Find the pack-size number associated with the detected dosage form."""
    if is_measured:
        m = _MEASURED_PACK_RE.search(name)
        if m:
            return m.group("num"), m.span()
        return None, None
    # count form: prefer a number immediately adjacent to the form token
    # (before or after), matched via form_match if it already captured one.
    if form_match and "num" in form_match.groupdict() and form_match.group("num"):
        return form_match.group("num"), form_match.span()
    # try number-after pattern anchored at the form token span
    tail = name[form_span[1]:form_span[1] + 6]
    m = re.match(r"\s*(\d+(?:\.\d+)?)", tail)
    if m:
        start = form_span[1] + m.start()
        end = form_span[1] + m.end()
        return m.group(1), (form_span[0], end)
    return None, None


def strip_span(name: str, span) -> str:
    if span is None:
        return name
    return (name[:span[0]] + " " + name[span[1]:]).strip()


def parse_name(raw_name: str) -> dict:
    name = str(raw_name).strip()
    working = name

    # 1. noise removal (promo text) -- purely cosmetic for trade name
    for pat in NOISE_PATTERNS:
        working = pat.sub(" ", working)

    # 2. dosage form
    form_info = detect_dosage_form(working)
    dosage_form = None
    unit_of_measure = None
    pack_size = None
    is_measured = False

    if form_info:
        dosage_form, unit_of_measure, is_measured, form_match = form_info
        pack_size, pack_span = extract_pack_size(working, form_match, is_measured, form_match.span())
        # remove pack-size number span first (if distinct from form span), then form token
        if pack_span and pack_span != form_match.span():
            working = strip_span(working, pack_span)
            # recompute form span after mutation by re-searching bare token
            canonical, unit, meas, _, _, p_bare = next(
                fp for fp in _FORM_PATTERNS if fp[0] == dosage_form
            )
            m = p_bare.search(working)
            if m:
                working = strip_span(working, m.span())
        else:
            working = strip_span(working, form_match.span())

    # 3. strip strength/concentration tokens
    working = STRENGTH_RE.sub(" ", working)

    # 4. flavour
    flavour = None
    fm = FLAVOUR_RE.search(working)
    if fm:
        flavour = fm.group(1).title()
        working = strip_span(working, fm.span())

    # 5. collapse whitespace / stray punctuation for trade name
    trade_name = re.sub(r"[\-/,]+\s*$|^\s*[\-/,]+", " ", working)
    trade_name = re.sub(r"\s+", " ", trade_name).strip(" -/,")
    trade_name = re.sub(r"\s+", " ", trade_name)

    # 6. confidence score
    confidence = 0.0
    if dosage_form:
        confidence += 0.4
    if pack_size:
        confidence += 0.3
    if trade_name and len(trade_name) >= 2:
        confidence += 0.3

    needs_review = confidence < config.PARSE_CONFIDENCE_THRESHOLD

    return {
        "ITEM_LOOKUP_NAME": name,
        "Trade Name": trade_name if trade_name else None,
        "Dosage Form": dosage_form,
        "Pack Size": pack_size,
        "Unit of Measure": unit_of_measure,
        "Flavour": flavour,
        "Parse_Confidence": round(confidence, 2),
        "Needs_LLM_Review": needs_review,
    }


def parse_item_master(df: pd.DataFrame) -> pd.DataFrame:
    records = [parse_name(n) for n in df["ITEM_LOOKUP_NAME"]]
    parsed = pd.DataFrame.from_records(records)
    return parsed


def main():
    print(f"Reading {config.INPUT_PATH} ...")
    df = pd.read_excel(config.INPUT_PATH, sheet_name=config.SHEET_ITEM_MASTER)
    print(f"Loaded {len(df)} rows.")

    parsed = parse_item_master(df)

    n_review = parsed["Needs_LLM_Review"].sum()
    n_total = len(parsed)
    print(f"Parsed {n_total} rows. Needs_LLM_Review: {n_review} "
          f"({n_review / n_total:.1%}). "
          f"High-confidence (no LLM needed): {n_total - n_review} "
          f"({(n_total - n_review) / n_total:.1%})")

    dist = parsed["Parse_Confidence"].value_counts().sort_index()
    print("Confidence distribution:")
    print(dist.to_string())

    out_path = config.CACHE_DIR / "item_master_parsed.pkl"
    parsed.to_pickle(out_path)
    print(f"Saved parsed table to {out_path}")
    return parsed


if __name__ == "__main__":
    main()
