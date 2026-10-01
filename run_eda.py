import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_parquet("data/enriched/movies_enriched.parquet")
print(f"Shape: {df.shape}")

# 1. Distribution des notes et popularité
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["vote_average"].dropna(), bins=30, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Distribution des notes moyennes (vote_average)")
axes[0].set_xlabel("Note moyenne (0-10)")
axes[0].axvline(df["vote_average"].mean(), color="red", linestyle="--", label=f"Moyenne: {df['vote_average'].mean():.2f}")
axes[0].legend()

sns.histplot(df["popularity"].dropna(), bins=30, kde=True, ax=axes[1], color="coral")
axes[1].set_title("Distribution de la popularité")
axes[1].set_xlabel("Popularité TMDB")
axes[1].set_yscale("log")
axes[1].axvline(df["popularity"].median(), color="red", linestyle="--", label=f"Médiane: {df['popularity'].median():.1f}")
axes[1].legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "1_distribution_notes_popularite.png", dpi=300)
plt.close()
print("1 done")

# 2. Films par genre
genres = pd.read_parquet("data/enriched/genres.parquet")
genre_counts = genres["genre_name"].value_counts().sort_values(ascending=True)
plt.figure(figsize=(10, 8))
plt.barh(range(len(genre_counts)), genre_counts.values, color="steelblue")
plt.yticks(range(len(genre_counts)), genre_counts.index)
plt.xlabel("Nombre de films")
plt.title("Nombre de films par genre")
for i, (idx, val) in enumerate(genre_counts.items()):
    plt.text(val + 5, i, str(val), va="center")
plt.tight_layout()
plt.savefig(FIG_DIR / "2_films_par_genre.png", dpi=300)
plt.close()
print("2 done")

# 3. Sorties par année et décennie
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
year_counts = df["release_year"].dropna().astype(int).value_counts().sort_index()
axes[0].plot(year_counts.index, year_counts.values, color="steelblue", marker="o", markersize=3)
axes[0].set_title("Nombre de films par année")
axes[0].set_xlabel("Année")
axes[0].set_ylabel("Nombre de films")
axes[0].grid(True, alpha=0.3)

decade_counts = df["decade"].dropna().astype(int).value_counts().sort_index()
axes[1].bar(decade_counts.index.astype(str), decade_counts.values, color="coral")
axes[1].set_title("Nombre de films par décennie")
axes[1].set_xlabel("Décennie")
axes[1].set_ylabel("Nombre de films")
for i, v in enumerate(decade_counts.values):
    axes[1].text(i, v + 5, str(v), ha="center")
plt.tight_layout()
plt.savefig(FIG_DIR / "3_sorties_par_annee_decennie.png", dpi=300)
plt.close()
print("3 done")

# 4. Durée
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["runtime"].dropna(), bins=30, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Distribution de la durée des films")
axes[0].set_xlabel("Durée (minutes)")
axes[0].axvline(df["runtime"].median(), color="red", linestyle="--", label=f"Médiane: {df['runtime'].median():.0f} min")
axes[0].legend()

order = ["<60", "60-90", "90-120", "120-150", "150-180", ">180"]
sns.boxplot(data=df, x="runtime_bucket", y="vote_average", order=order, ax=axes[1], palette="Set2")
axes[1].set_title("Note moyenne par tranche de durée")
axes[1].set_xlabel("Tranche de durée")
axes[1].set_ylabel("Note moyenne")
plt.tight_layout()
plt.savefig(FIG_DIR / "4_duree_films.png", dpi=300)
plt.close()
print("4 done")

# 5. Budget vs Revenus
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
mask = df["budget"].notna() & df["revenue"].notna() & (df["budget"] > 0) & (df["revenue"] > 0)
scatter = axes[0].scatter(df.loc[mask, "budget"], df.loc[mask, "revenue"], c=df.loc[mask, "vote_average"], cmap="RdYlGn", alpha=0.6, s=20)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].plot([1e4, 3e8], [1e4, 3e8], "k--", alpha=0.5, label="Break-even")
axes[0].set_title("Budget vs Revenus (échelle log)")
axes[0].set_xlabel("Budget (USD)")
axes[0].set_ylabel("Revenus (USD)")
axes[0].legend()
plt.colorbar(scatter, ax=axes[0], label="Note moyenne")

sns.boxplot(data=df.dropna(subset=["budget_tier", "revenue"]), x="budget_tier", y="revenue", order=["Very Low", "Low", "Medium", "High", "Very High"], ax=axes[1], palette="Blues")
axes[1].set_yscale("log")
axes[1].set_title("Revenus par tranche de budget")
axes[1].set_xlabel("Tranche de budget")
axes[1].set_ylabel("Revenus (USD, log)")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(FIG_DIR / "5_budget_vs_revenus.png", dpi=300)
plt.close()
print("5 done")

# 6. Votes vs Popularité
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
scatter = axes[0].scatter(df["vote_count"], df["popularity"], c=df["vote_average"], cmap="viridis", alpha=0.6, s=20)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_title("Nombre de votes vs Popularité")
axes[0].set_xlabel("Nombre de votes (log)")
axes[0].set_ylabel("Popularité (log)")
plt.colorbar(scatter, ax=axes[0], label="Note moyenne")

axes[1].scatter(df["vote_count"], df["vote_average"], alpha=0.5, s=15, color="steelblue")
axes[1].set_xscale("log")
axes[1].set_title("Note moyenne vs Nombre de votes")
axes[1].set_xlabel("Nombre de votes (log)")
axes[1].set_ylabel("Note moyenne")
axes[1].axhline(df["vote_average"].mean(), color="red", linestyle="--", label=f"Moyenne globale: {df['vote_average'].mean():.2f}")
axes[1].legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "6_votes_vs_popularite.png", dpi=300)
plt.close()
print("6 done")

# 7. Boxplots et Corrélations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
genres = pd.read_parquet("data/enriched/genres.parquet")
df_genres = df.merge(genres, left_on="id", right_on="movie_id")
top_genres = df_genres["genre_name"].value_counts().head(8).index
sns.boxplot(data=df_genres[df_genres["genre_name"].isin(top_genres)], x="genre_name", y="vote_average", ax=axes[0,0], palette="Set3")
axes[0,0].set_title("Note moyenne par genre (Top 8)")
axes[0,0].tick_params(axis="x", rotation=45)
axes[0,0].set_xlabel("")
axes[0,0].set_ylabel("Note moyenne")

sns.boxplot(data=df.dropna(subset=["budget_tier", "roi"]), x="budget_tier", y="roi", order=["Very Low", "Low", "Medium", "High", "Very High"], ax=axes[0,1], palette="Reds")
axes[0,1].set_yscale("symlog")
axes[0,1].set_title("ROI par tranche de budget")
axes[0,1].set_xlabel("Tranche de budget")
axes[0,1].set_ylabel("ROI (log symétrique)")
axes[0,1].tick_params(axis="x", rotation=45)

num_cols = ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count", "profit", "roi", "weighted_rating", "release_year"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1,0], square=True, cbar_kws={"shrink": 0.8})
axes[1,0].set_title("Matrice de corrélation (variables numériques)")

sample = df[num_cols].dropna().sample(min(500, len(df)), random_state=42)
sns.scatterplot(data=sample, x="budget", y="revenue", hue="vote_average", palette="RdYlGn", alpha=0.7, ax=axes[1,1])
axes[1,1].set_xscale("log")
axes[1,1].set_yscale("log")
axes[1,1].set_title("Budget vs Revenue (échantillon, coloré par note)")
axes[1,1].set_xlabel("Budget (log)")
axes[1,1].set_ylabel("Revenus (log)")

plt.tight_layout()
plt.savefig(FIG_DIR / "7_boxplots_correlations.png", dpi=300)
plt.close()
print("7 done")

print("All figures saved to", FIG_DIR)