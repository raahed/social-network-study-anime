import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# -------------------------
#  CONFIG
# -------------------------

MIN_RATINGS = 5
TOP_N = 5
K_NEIGHBORS = 25
MIN_OVERLAP = 6

OUTPUT_FILE = "../prediction/users-recommendations.csv"

# -------------------------
#  DATA LOADING
# -------------------------

anime_df = pd.read_csv("../data/clean/anime-dataset-filtered.csv")
watch_2023 = pd.read_csv("../data/clean/users-score-shrunk-2023.csv")
watch_2025 = pd.read_csv("../data/clean/users-score-shrunk-2025.csv")

rating_matrix = watch_2023.pivot_table(
    index="user_id",
    columns="anime_id",
    values="rating",
    aggfunc="mean"
)

user_ids = rating_matrix.index.tolist()
anime_ids = rating_matrix.columns.tolist()

# -------------------------
#  PEARSON CORRELATION
# -------------------------

def pearson_on_overlap(u, v, mtx, min_overlap=MIN_OVERLAP):
    r_u = mtx.loc[u]
    r_v = mtx.loc[v]

    mask = r_u.notna() & r_v.notna()
    if mask.sum() < min_overlap:
        return np.nan

    x = r_u[mask]
    y = r_v[mask]

    if x.nunique() <= 1 or y.nunique() <= 1:
        return np.nan

    return x.corr(y)

# -------------------------
#  FIND NEIGHBORS
# -------------------------

def get_neighbors_standard(target_user, k=K_NEIGHBORS):
    sims = {}

    for other in user_ids:
        if other == target_user:
            continue

        sim = pearson_on_overlap(target_user, other, rating_matrix)
        if not np.isnan(sim):
            sims[other] = sim

    if not sims:
        return []

    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:k]

# -------------------------
#  PREDICT RATING
# -------------------------

def predict_rating_standard(target_user, anime_id, neighbors, mtx):
    if anime_id not in mtx.columns:
        return np.nan

    target_mean = mtx.loc[target_user].mean()
    numer = 0.0
    denom = 0.0

    for uid, sim in neighbors:
        r = mtx.at[uid, anime_id]
        if pd.isna(r):
            continue

        mean_u = mtx.loc[uid].mean()

        numer += sim * (r - mean_u)
        denom += abs(sim)

    if denom == 0:
        return np.nan

    return target_mean + numer / denom

# -------------------------
#  RECOMMENDATION
# -------------------------

def recommend_user(user_id, top_n=TOP_N, k_neighbors=K_NEIGHBORS):
    neighbors = get_neighbors_standard(user_id, k_neighbors)
    if len(neighbors) == 0:
        return {}

    seen = set(rating_matrix.loc[user_id].dropna().index)
    candidates = [aid for aid in anime_ids if aid not in seen]

    preds = {}
    for aid in candidates:
        pred = predict_rating_standard(user_id, aid, neighbors, rating_matrix)
        if not pd.isna(pred):
            preds[aid] = pred

    if not preds:
        return {}

    # Top-N auswählen
    top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return {aid: score for aid, score in top_items}

# -------------------------
#  PREPARATION (Checkpointing)
# -------------------------

user_counts = watch_2023.groupby("user_id").size()
eligible_users = user_counts[user_counts >= MIN_RATINGS].index.tolist()

# Check if output exists → already processed users
if os.path.exists(OUTPUT_FILE):
    processed = pd.read_csv(OUTPUT_FILE)["user_id"].unique().tolist()
else:
    processed = []
    # write header
    with open(OUTPUT_FILE, "w") as f:
        f.write("user_id,anime_id,rating\n")

# -------------------------
#  RUN & SAVE RECOMMENDATIONS
# -------------------------

for uid in tqdm(eligible_users, desc="Processing users"):
    if uid in processed:  # skip if already completed
        continue

    recs = recommend_user(uid)

    if recs:
        out_df = pd.DataFrame({
            "user_id": [uid] * len(recs),
            "anime_id": list(recs.keys()),
            "rating": list(recs.values())
        })

        # append to CSV
        out_df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

print("FERTIG! Empfehlungen gespeichert in:", OUTPUT_FILE)
