import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

# Für das Content-based-Modell
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel
from scipy import sparse

# ---------------------------------------------------------
# 1. KONFIGURATION
# ---------------------------------------------------------

# Anzahl der Empfehlungen pro User
TOP_N = 10  # konsistent

# Alle User verwenden (kein Limit mehr)
MAX_USERS_FOR_TEST = None   # None = alle Nutzer

# Basis-Pfad zu deinem Datenordner
DATA_DIR = "/Users/lubovschlundt/Desktop/TH Nürnberg/2. Semester/Soziale Netzwerkanalyse/data"


# ---------------------------------------------------------
# 2. DATEN LADEN
# ---------------------------------------------------------

anime_df = pd.read_csv(f"{DATA_DIR}/anime-dataset-filtered.csv")
watch_2023 = pd.read_csv(f"{DATA_DIR}/users-score-shrunk-2023.csv")
watch_2025 = pd.read_csv(f"{DATA_DIR}/users-score-shrunk-2025.csv")

print(f"Anime-Datensatz: {anime_df.shape[0]} Einträge")
print(f"Ratings 2023:    {watch_2023.shape[0]} Einträge")
print(f"Ratings 2025:    {watch_2025.shape[0]} Einträge")

# Sicherstellen, dass anime_id als Index verfügbar ist
anime_df = anime_df.set_index("anime_id", drop=False)


# ---------------------------------------------------------
# 3. NUTZER-AUSWAHL FÜR DIE EVALUATION
# ---------------------------------------------------------

# Datensatz ist bereits geschrumpft/gefiltert -> wir nehmen alle User
user_counts = watch_2023.groupby("user_id").size()
eligible_users = user_counts.index.tolist()  # wirklich unterschiedliche user_ids

# Optionales Limit (hier: keins, weil None)
if MAX_USERS_FOR_TEST is not None:
    eligible_users = eligible_users[:MAX_USERS_FOR_TEST]

print(f"Anzahl berücksichtigter Nutzer: {len(eligible_users)}")

# Pre-Cache: Welche Anime hat ein User im Jahr 2025 geschaut?
watched_2025: Dict[int, set] = {
    uid: set(watch_2025[watch_2025["user_id"] == uid]["anime_id"])
    for uid in eligible_users
}


# ---------------------------------------------------------
# 4. CONTENT-FEATURES AUS GENRES / THEMES / DEMOGRAPHICS / TYPE / STUDIOS
# ---------------------------------------------------------

# Fehlende Spalten abfangen, falls irgendwas im CSV nicht da ist
for col in ["Genres", "Themes", "Demographics", "Type", "Studios"]:
    if col in anime_df.columns:
        anime_df[col] = anime_df[col].fillna("")
    else:
        anime_df[col] = ""

def tokenize_multi_value(val: str) -> List[str]:
    """
    Zerlegt kommagetrennte Felder (z.B. 'Action, Adventure')
    in Tokens und ersetzt Leerzeichen durch '_'.
    Beispiel: 'Slice of Life' -> 'Slice_of_Life'
    """
    if not isinstance(val, str):
        val = str(val)
    tokens = []
    for part in val.split(","):
        t = part.strip()
        if not t:
            continue
        t = t.replace(" ", "_")
        tokens.append(t)
    return tokens

def build_content_string(row: pd.Series) -> str:
    """
    Baut aus mehreren Spalten einen kombinierten Content-String für einen Anime.

    Verwendete Felder:
    - Genres
    - Themes
    - Demographics
    - Type     (TV, Movie, OVA, ...)
    - Studios  (z.B. Madhouse, Bones, Kyoto_Animation)

    Alle Tokens werden einfach zusammengefügt, OHNE zusätzliche Gewichte
    oder Prefix-Gruppierung.
    """
    tokens = []
    tokens.extend(tokenize_multi_value(row["Genres"]))
    tokens.extend(tokenize_multi_value(row["Themes"]))
    tokens.extend(tokenize_multi_value(row["Demographics"]))
    tokens.extend(tokenize_multi_value(row["Type"]))
    tokens.extend(tokenize_multi_value(row["Studios"]))
    return " ".join(tokens)

# Content-Text pro Anime erzeugen
content_corpus = anime_df.apply(build_content_string, axis=1)

print("Beispiel-Content eines Animes:")
print(content_corpus.iloc[0])


# ---------------------------------------------------------
# 5. TF-IDF-FEATURE-MATRIX ERZEUGEN (ohne manuelle Gewichtung)
# ---------------------------------------------------------

# Standard-TF-IDF mit L2-Norm (Sklearn-Default)
vectorizer = TfidfVectorizer()  # norm='l2' ist Default
tfidf_matrix = vectorizer.fit_transform(content_corpus.values)  # (n_anime, n_features)

anime_ids_cb: List[int] = content_corpus.index.to_list()
anime_id_to_idx: Dict[int, int] = {aid: idx for idx, aid in enumerate(anime_ids_cb)}

print(f"TF-IDF-Matrix: {tfidf_matrix.shape[0]} Anime x {tfidf_matrix.shape[1]} Features")


# ---------------------------------------------------------
# 6. USER-PROFILE AUF BASIS VON RATING + CONTENT
# ---------------------------------------------------------

# Nur Ratings, für die wir Content-Features haben
watch_2023_cb = watch_2023[watch_2023["anime_id"].isin(anime_id_to_idx.keys())].copy()

# Durchschnittsrating pro User
user_means_cb = watch_2023_cb.groupby("user_id")["rating"].mean()
watch_2023_cb = watch_2023_cb.join(user_means_cb, on="user_id", rsuffix="_mean")

# Mean-centering: Gewicht = r_{u,i} - mean_u
watch_2023_cb["weight"] = watch_2023_cb["rating"] - watch_2023_cb["rating_mean"]

def build_user_profile_tfidf(user_id: int):
    """
    Baut ein Content-basiertes User-Profil als TF-IDF-Vektor.

    - Nutzt ALLE Ratings 2023 des Users (Basis des Systems)
    - Gewichte = mean-centred Ratings (r - mean_u)
    - Profil wird mit L2-Norm normalisiert
    """
    rows = watch_2023_cb[watch_2023_cb["user_id"] == user_id]
    if rows.empty:
        return None

    rows = rows[rows["anime_id"].isin(anime_id_to_idx.keys())]
    if rows.empty:
        return None

    indices = [anime_id_to_idx[aid] for aid in rows["anime_id"]]
    weights = rows["weight"].values

    if np.allclose(weights, 0):
        return None

    user_tfidf = tfidf_matrix[indices]  # (n_items, n_features)

    w_diag = sparse.diags(weights)
    weighted = w_diag @ user_tfidf
    profile = weighted.sum(axis=0)      # (1, n_features)

    profile = sparse.csr_matrix(profile)
    profile = normalize(profile, norm="l2")  # Standardnorm

    return profile

# Welche Anime hat ein User 2023 gesehen?
seen_2023_cb: Dict[int, set] = {
    uid: set(watch_2023_cb[watch_2023_cb["user_id"] == uid]["anime_id"])
    for uid in watch_2023_cb["user_id"].unique()
}


# ---------------------------------------------------------
# 7. CONTENT-BASED EMPFEHLUNG FÜR EINEN USER
# ---------------------------------------------------------

def recommend_user_content_based(user_id: int, top_n: int = TOP_N) -> List[Tuple[int, float]]:
    """
    Erzeugt Top-N Content-basierte Empfehlungen für einen User.

    Rückgabeformat: Liste von (anime_id, score).

    WICHTIG:
    - Ratings 2023 sind Basis für das User-Profil
    - Empfohlen werden NUR Anime, die 2023 noch nicht gesehen wurden
      (typisches Empfehlungs-Szenario).
    """
    profile = build_user_profile_tfidf(user_id)
    if profile is None:
        return []

    # Cosine-Similarity: User-Profil vs. alle Anime (bei L2-normalisierten Vektoren)
    sims = linear_kernel(profile, tfidf_matrix).ravel()  # (n_anime,)

    sim_series = pd.Series(sims, index=anime_ids_cb)

    # Anime entfernen, die der User 2023 bereits gesehen/bewertet hat
    seen = seen_2023_cb.get(user_id, set())
    sim_series = sim_series.drop(labels=list(seen), errors="ignore")

    # Nur positive Similarities
    sim_series = sim_series[sim_series > 0]

    if sim_series.empty:
        return []

    top_items = sim_series.sort_values(ascending=False).head(top_n)

    # Rückgabe: Liste von (anime_id, score)
    return list(zip(top_items.index.tolist(), top_items.values.tolist()))


# ---------------------------------------------------------
# 8. EVALUATION: HIT RATE + ERGEBNIS-FORMAT "user_id, anime_id"
# ---------------------------------------------------------

rows_out = []   # eine Zeile pro Vorhersage: user_id, anime_id, score, hit
hit_count_cb = 0
total_recommendations_cb = 0

# Nur Nutzer, die in watch_2023_cb vorkommen (sonst kein Profil möglich)
eligible_users_cb = [u for u in eligible_users if u in watch_2023_cb["user_id"].unique()]

print(f"Nutzer für Content-based Evaluation: {len(eligible_users_cb)}")

for uid in tqdm(eligible_users_cb, desc="Evaluating users (content-based)"):
    recs = recommend_user_content_based(uid)

    if not recs:
        continue

    seen_2025 = watched_2025.get(uid, set())

    for aid, score in recs:
        is_hit = 1 if aid in seen_2025 else 0
        hit_count_cb += is_hit
        total_recommendations_cb += 1

        rows_out.append({
            "user_id": uid,
            "anime_id": aid,
            "score": score,
            "hit": is_hit
        })

# Ergebnisse in CSV im gleichen Ordner wie die Input-Dateien speichern
output_path = f"{DATA_DIR}/content_based_results_flat.csv"
results_df = pd.DataFrame(rows_out)
results_df.to_csv(output_path, index=False)
print(f"Detailergebnisse (eine Zeile pro Vorhersage) wurden gespeichert unter:\n{output_path}")

# Hit Rate berechnen
if total_recommendations_cb == 0:
    print("Content-based Hit Rate: 0% (keine Empfehlungen generiert)")
else:
    hit_rate_cb = (hit_count_cb / total_recommendations_cb) * 100
    print(f"Content-based Hit Rate (ohne Feature-Gewichte): {hit_rate_cb:.2f}%")