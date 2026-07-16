"""Central paths, thresholds, and model config for the item mapping pipeline.

This is the recommended approach's implementation (see ../README.md and
../Approach_Comparison.md for why): Steps A-C are identical to the original
proof-of-concept, and Step D's LLM adjudication uses a single Ollama cloud
model (gemma4:31b-cloud) throughout, with a prompt refined based on manual
error analysis of an earlier local-model pass (see Approach_Comparison.md).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "Item Mapping.xlsx"
OUTPUT_PATH = ROOT / "output" / "Item_Mapping_Result.xlsx"

CACHE_DIR = ROOT / "cache"
LOG_DIR = ROOT / "logs"

ITEM_MASTER_EMBEDDINGS_PATH = CACHE_DIR / "item_master_embeddings.npy"
ITEM_MASTER_EMBEDDINGS_META_PATH = CACHE_DIR / "item_master_embeddings_meta.json"
ITEM_MASTER_PARSED_PATH = CACHE_DIR / "item_master_parsed.pkl"
PHARMACY1_MATCHED_PATH = CACHE_DIR / "pharmacy1_matched.pkl"
PHARMACY2_MATCHED_PATH = CACHE_DIR / "pharmacy2_matched.pkl"
LLM_PARSE_CACHE_PATH = CACHE_DIR / "llm_parse_cache.jsonl"
LLM_MATCH_CACHE_PATH = CACHE_DIR / "llm_match_cache.jsonl"

SHEET_ITEM_MASTER = "Item Master File"
SHEET_PHARMACY1 = "Pharmacy 1"
SHEET_PHARMACY2 = "Pharmacy 2"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "embeddinggemma:latest"

# Single model throughout Step D -- no local/cloud tiering. An earlier local
# (gemma4:e2b) + cloud-escalation design was tried first; manual error
# analysis found the local model was the source of most matching errors, and
# switching to the cloud model alone (with a refined prompt) measurably
# improved accuracy. See ../Approach_Comparison.md.
CLOUD_MODEL = "gemma4:31b-cloud"

# Step A: field-extraction confidence threshold. Rows scoring below this
# get Needs_LLM_Review=True instead of a guessed value.
PARSE_CONFIDENCE_THRESHOLD = 0.7

# Step C: retrieval + tiering
RETRIEVAL_TOP_K = 8
TIER_HIGH_MIN = 90
TIER_MEDIUM_MIN = 70

# Step D: LLM adjudication batching
LLM_BATCH_SIZE = 15
LLM_MAX_RETRIES = 1

RANDOM_SEED = 42
