"""Step C: candidate retrieval (TF-IDF char n-gram + embeddinggemma semantic,
unioned) + rapidfuzz rerank + tiering.

Retrieval fits ONCE over all 60,085 unique (deduped) Item Master names -- the
only step touching the full master list. Embedding embeddinggemma is
non-trivial at this scale (~25 min warm), so results are cached to
cache/item_master_embeddings.npy keyed by the exact ordered name list (a
content hash in the sidecar meta.json invalidates the cache if the source
names change).

Measured batch embedding throughput against the local Ollama daemon: cold
start ~19s, then ~20-50 items/sec once the model is loaded; batches of 500
occasionally crash the runner, so batch size is capped at 150 with retry/backoff.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import ollama_utils

EMBED_BATCH_SIZE = 150
EMBED_MAX_RETRIES = 4


def normalize_name(name) -> str:
    if not isinstance(name, str):
        return ""
    n = name.upper()
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def build_dedup_index(item_master_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per unique normalized name. Keeps the first
    original ITEM_LOOKUP_NAME as the representative display name and the
    list of original row indices sharing that normalized name (for the 851
    exact-duplicate rows)."""
    df = item_master_df.copy()
    df["Name_Normalized"] = df["ITEM_LOOKUP_NAME"].apply(normalize_name)
    grouped = df.groupby("Name_Normalized", sort=False).agg(
        Item_Lookup_Name=("ITEM_LOOKUP_NAME", "first"),
        Original_Indices=("ITEM_LOOKUP_NAME", lambda s: list(s.index)),
    ).reset_index()
    # drop empty-name rows (shouldn't exist, but guard anyway)
    grouped = grouped[grouped["Name_Normalized"].str.len() > 0].reset_index(drop=True)
    return grouped


def embed_texts_batched(client, texts, log_prefix="", batch_size=EMBED_BATCH_SIZE):
    """Embed a list of texts via embeddinggemma, batching with retry/backoff.
    Returns an (N, D) numpy array in the same order as `texts`."""
    vectors = [None] * len(texts)
    n = len(texts)
    t_start = time.time()
    i = 0
    cur_batch = batch_size
    while i < n:
        batch = texts[i:i + cur_batch]
        attempt = 0
        while True:
            try:
                resp = client.embed(model=config.EMBEDDING_MODEL, input=batch)
                embs = resp["embeddings"]
                for j, e in enumerate(embs):
                    vectors[i + j] = e
                break
            except Exception as e:
                attempt += 1
                if attempt > EMBED_MAX_RETRIES or cur_batch <= 5:
                    raise RuntimeError(
                        f"Embedding batch failed permanently at offset {i} "
                        f"(batch_size={cur_batch}): {e}"
                    )
                cur_batch = max(5, cur_batch // 2)
                batch = texts[i:i + cur_batch]
                print(f"  {log_prefix}WARNING: embed batch failed ({e}); "
                      f"retrying at offset {i} with batch_size={cur_batch} "
                      f"(attempt {attempt}/{EMBED_MAX_RETRIES})")
                time.sleep(1.5 * attempt)
        i += len(batch)
        elapsed = time.time() - t_start
        rate = i / elapsed if elapsed > 0 else 0
        eta = (n - i) / rate if rate > 0 else float("nan")
        print(f"  {log_prefix}embedded {i}/{n} ({i/n:.1%}) "
              f"rate={rate:.1f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s")
        # batch size can recover after a run of successes
        cur_batch = min(batch_size, cur_batch * 2) if cur_batch < batch_size else cur_batch

    return np.array(vectors, dtype=np.float32)


def get_or_build_item_master_embeddings(client, names: list[str]) -> np.ndarray:
    import hashlib
    content_hash = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()

    if config.ITEM_MASTER_EMBEDDINGS_PATH.exists() and config.ITEM_MASTER_EMBEDDINGS_META_PATH.exists():
        meta = json.loads(config.ITEM_MASTER_EMBEDDINGS_META_PATH.read_text())
        if meta.get("content_hash") == content_hash and meta.get("count") == len(names):
            print(f"Loading cached item master embeddings ({len(names)} vectors) ...")
            return np.load(config.ITEM_MASTER_EMBEDDINGS_PATH)
        print("Item master embeddings cache is stale (name list changed); rebuilding.")

    print(f"Embedding {len(names)} unique item master names via {config.EMBEDDING_MODEL} "
          f"(this can take a while on first run)...")
    vectors = embed_texts_batched(client, names, log_prefix="[item_master] ")
    config.CACHE_DIR.mkdir(exist_ok=True)
    np.save(config.ITEM_MASTER_EMBEDDINGS_PATH, vectors)
    config.ITEM_MASTER_EMBEDDINGS_META_PATH.write_text(json.dumps({
        "content_hash": content_hash,
        "count": len(names),
        "model": config.EMBEDDING_MODEL,
    }))
    return vectors


def get_or_build_query_embeddings(client, names: list[str], cache_path: Path, label: str) -> np.ndarray:
    import hashlib
    content_hash = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    meta_path = cache_path.with_suffix(".meta.json")

    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("content_hash") == content_hash and meta.get("count") == len(names):
            print(f"Loading cached {label} query embeddings ({len(names)} vectors) ...")
            return np.load(cache_path)

    print(f"Embedding {len(names)} {label} names via {config.EMBEDDING_MODEL} ...")
    vectors = embed_texts_batched(client, names, log_prefix=f"[{label}] ")
    config.CACHE_DIR.mkdir(exist_ok=True)
    np.save(cache_path, vectors)
    meta_path.write_text(json.dumps({"content_hash": content_hash, "count": len(names)}))
    return vectors


def build_retrieval_indexes(dedup_df: pd.DataFrame, item_master_embeddings: np.ndarray):
    names = dedup_df["Name_Normalized"].tolist()

    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    tfidf_matrix = tfidf.fit_transform(names)
    tfidf_nn = NearestNeighbors(metric="cosine", n_neighbors=min(config.RETRIEVAL_TOP_K, len(names)))
    tfidf_nn.fit(tfidf_matrix)

    emb_nn = NearestNeighbors(metric="cosine", n_neighbors=min(config.RETRIEVAL_TOP_K, len(names)))
    emb_nn.fit(item_master_embeddings)

    return tfidf, tfidf_nn, emb_nn


def retrieve_and_rerank(query_names, query_embeddings, dedup_df, tfidf, tfidf_nn, emb_nn):
    """For each query name, union TF-IDF and embedding nearest neighbors,
    rerank with rapidfuzz token_sort_ratio, return best match + tier + top-3
    candidates for every query."""
    master_names = dedup_df["Name_Normalized"].tolist()
    master_display = dedup_df["Item_Lookup_Name"].tolist()

    query_tfidf = tfidf.transform(query_names)
    _, tfidf_idx = tfidf_nn.kneighbors(query_tfidf)
    _, emb_idx = emb_nn.kneighbors(query_embeddings)

    results = []
    for i, qname in enumerate(query_names):
        candidate_idx = sorted(set(tfidf_idx[i].tolist()) | set(emb_idx[i].tolist()))
        scored = []
        for ci in candidate_idx:
            score = fuzz.token_sort_ratio(qname, master_names[ci])
            scored.append((score, ci))
        scored.sort(key=lambda x: -x[0])

        if not scored:
            results.append({
                "Best_Match_Name": None, "Best_Match_Score": 0, "Tier": "LOW_NO_MATCH",
                "Top_Candidates": [],
            })
            continue

        best_score, best_ci = scored[0]
        if best_score >= config.TIER_HIGH_MIN:
            tier = "HIGH_AUTO_ACCEPT"
        elif best_score >= config.TIER_MEDIUM_MIN:
            tier = "MEDIUM_LLM_REVIEW"
        else:
            tier = "LOW_NO_MATCH"

        top3 = [
            {"name": master_display[ci], "score": score}
            for score, ci in scored[:3]
        ]

        results.append({
            "Best_Match_Name": master_display[best_ci],
            "Best_Match_Score": best_score,
            "Tier": tier,
            "Top_Candidates": top3,
        })

    return pd.DataFrame(results)


def main():
    import ollama

    print("Loading parsed item master ...")
    item_master = pd.read_pickle(config.CACHE_DIR / "item_master_parsed.pkl")
    dedup_df = build_dedup_index(item_master.rename(columns={"ITEM_LOOKUP_NAME": "ITEM_LOOKUP_NAME"}))
    print(f"Deduped item master: {len(dedup_df)} unique names "
          f"(from {len(item_master)} rows, {len(item_master) - len(dedup_df)} duplicates collapsed).")

    client = ollama.Client(host=config.OLLAMA_HOST)
    ollama_utils.unload_other_models(config.EMBEDDING_MODEL)

    master_embeddings = get_or_build_item_master_embeddings(
        client, dedup_df["Name_Normalized"].tolist()
    )

    print("Building TF-IDF + embedding retrieval indexes ...")
    tfidf, tfidf_nn, emb_nn = build_retrieval_indexes(dedup_df, master_embeddings)

    print("Loading cleaned pharmacy sheets ...")
    p1 = pd.read_pickle(config.CACHE_DIR / "pharmacy1_cleaned.pkl")
    p2 = pd.read_pickle(config.CACHE_DIR / "pharmacy2_cleaned.pkl")

    p1_embeddings = get_or_build_query_embeddings(
        client, p1["Name_Normalized"].tolist(),
        config.CACHE_DIR / "pharmacy1_query_embeddings.npy", "pharmacy1"
    )
    p2_embeddings = get_or_build_query_embeddings(
        client, p2["Name_Normalized"].tolist(),
        config.CACHE_DIR / "pharmacy2_query_embeddings.npy", "pharmacy2"
    )

    print("Retrieving + reranking Pharmacy 1 ...")
    p1_matches = retrieve_and_rerank(
        p1["Name_Normalized"].tolist(), p1_embeddings, dedup_df, tfidf, tfidf_nn, emb_nn
    )
    p1_result = pd.concat([p1.reset_index(drop=True), p1_matches], axis=1)

    print("Retrieving + reranking Pharmacy 2 ...")
    p2_matches = retrieve_and_rerank(
        p2["Name_Normalized"].tolist(), p2_embeddings, dedup_df, tfidf, tfidf_nn, emb_nn
    )
    p2_result = pd.concat([p2.reset_index(drop=True), p2_matches], axis=1)

    for label, res in [("Pharmacy 1", p1_result), ("Pharmacy 2", p2_result)]:
        dist = res["Tier"].value_counts()
        print(f"{label} tier distribution:")
        for tier in ["HIGH_AUTO_ACCEPT", "MEDIUM_LLM_REVIEW", "LOW_NO_MATCH"]:
            n = dist.get(tier, 0)
            print(f"  {tier}: {n} ({n/len(res):.1%})")

    p1_result.to_pickle(config.CACHE_DIR / "pharmacy1_matched.pkl")
    p2_result.to_pickle(config.CACHE_DIR / "pharmacy2_matched.pkl")
    dedup_df.to_pickle(config.CACHE_DIR / "item_master_dedup.pkl")
    print("Saved match results to cache/.")

    ollama_utils.unload_all_models()

    return p1_result, p2_result, dedup_df


if __name__ == "__main__":
    main()
