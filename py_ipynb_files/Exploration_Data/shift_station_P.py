#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
shift_station_P.py — Justification du choix de ne garder que les données post-shift
=====================================================================================
Ce script montre pourquoi les données de débit de la Station P antérieures
au 2025-04-03 ont été écartées de la modélisation :

  1. Rupture brutale de niveau : le débit moyen passe de ~46 m³/h à ~235 m³/h
  2. Distributions totalement différentes (histogrammes + boxplots)
  3. Corrélation pluie→débit quasi nulle avant le shift, significative après
  4. Réponse aux événements de pluie : avant le shift, la pluie n'entraîne
     pas de hausse cohérente du débit

Conclusion : les deux périodes ne représentent pas le même système physique.
Entraîner un modèle sur les deux périodes mélangées introduirait du bruit
et dégraderait les performances.
"""

import warnings; warnings.filterwarnings("ignore")
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

# ══════════════════════════════════════════════════════════════
#  CHEMINS & CONSTANTES
# ══════════════════════════════════════════════════════════════
BASE  = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\bronze\bronze")
OUT   = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\Exploration_Data")
SHIFT_DATE = "2025-04-03"

COL_PRE  = "#e07b39"   # orange — avant shift
COL_POST = "#2e86ab"   # bleu   — après shift

# ══════════════════════════════════════════════════════════════
#  1. CHARGEMENT DÉBIT
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("1. CHARGEMENT DES DONNÉES")
print("=" * 65)

chunks = []
for chunk in pd.read_csv(BASE / "sensor_P_20231110_20260106.csv", chunksize=500_000):
    sub = chunk[chunk["sensor"] == "entry_debit_f1"]
    if len(sub): chunks.append(sub)
df_raw = pd.concat(chunks)
df_raw["ts"] = pd.to_datetime(df_raw["ts"])
debit = (df_raw.groupby("ts")["value"].mean()
         .resample("h").mean().ffill().clip(lower=0))

# ── Météo ────────────────────────────────────────────────────
weather_files = sorted((BASE / "weather_P").glob("*_hourly2.csv"))
if not weather_files:
    weather_files = sorted((BASE / "weather_P").glob("*_hourly.csv"))

dfs_w = []
for f in weather_files:
    d = pd.read_csv(f)
    d["ts"] = (pd.to_datetime(d["date"]).dt.tz_localize("UTC")
               .dt.tz_convert("Europe/Paris").dt.tz_localize(None))
    d = d.set_index("ts")
    dfs_w.append(d[["rain"]])
df_weather = pd.concat(dfs_w).groupby(level=0)[["rain"]].mean()
df_weather = df_weather[~df_weather.index.duplicated(keep="first")]

df = pd.DataFrame({"debit": debit}).join(df_weather, how="inner")
df["rain"] = df["rain"].fillna(0)

df_pre  = df[df.index <  SHIFT_DATE].dropna()
df_post = df[df.index >= SHIFT_DATE].dropna()

print(f"  Période PRE-shift  : {df_pre.index.min().date()} → {df_pre.index.max().date()}  ({len(df_pre):,} h)")
print(f"  Période POST-shift : {df_post.index.min().date()} → {df_post.index.max().date()}  ({len(df_post):,} h)")

# ══════════════════════════════════════════════════════════════
#  2. STATISTIQUES DESCRIPTIVES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("2. STATISTIQUES DESCRIPTIVES")
print("=" * 65)

stats_pre  = df_pre["debit"].describe()
stats_post = df_post["debit"].describe()

print(f"\n  {'Statistique':<12}  {'PRE-shift':>12}  {'POST-shift':>12}  {'Ratio':>8}")
print(f"  {'─'*50}")
for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
    v_pre  = stats_pre[stat]
    v_post = stats_post[stat]
    ratio  = v_post / v_pre if v_pre != 0 else float("nan")
    print(f"  {stat:<12}  {v_pre:>12.1f}  {v_post:>12.1f}  {ratio:>7.1f}×")

# ══════════════════════════════════════════════════════════════
#  3. CORRÉLATION PLUIE → DÉBIT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("3. CORRÉLATION PLUIE → DÉBIT  (lags 0 à 72h)")
print("=" * 65)

lags = list(range(0, 73, 3))  # 0, 3, 6, ..., 72h
corr_pre, corr_post = [], []

for lag in lags:
    rain_shifted = df["rain"].shift(lag)

    # PRE
    idx = df_pre.index
    r_pre, _ = pearsonr(
        df.loc[idx, "debit"].values,
        rain_shifted.loc[idx].fillna(0).values
    )
    corr_pre.append(r_pre)

    # POST
    idx = df_post.index
    r_post, _ = pearsonr(
        df.loc[idx, "debit"].values,
        rain_shifted.loc[idx].fillna(0).values
    )
    corr_post.append(r_post)

corr_pre  = np.array(corr_pre)
corr_post = np.array(corr_post)

print(f"\n  Corrélation max PRE-shift  : {corr_pre.max():.3f}  (lag={lags[corr_pre.argmax()]}h)")
print(f"  Corrélation max POST-shift : {corr_post.max():.3f}  (lag={lags[corr_post.argmax()]}h)")
print(f"\n  Tableau (Pearson r) :")
print(f"  {'Lag':>5}  {'PRE':>7}  {'POST':>7}")
print(f"  {'─'*22}")
for i, lag in enumerate(lags):
    print(f"  {lag:>4}h  {corr_pre[i]:>7.3f}  {corr_post[i]:>7.3f}")

# ══════════════════════════════════════════════════════════════
#  4. ÉVÉNEMENTS DE PLUIE — RÉPONSE DU DÉBIT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("4. RÉPONSE AUX ÉVÉNEMENTS DE PLUIE (corrélation glissante)")
print("=" * 65)

# Rolling 30 jours de corrélation pluie (sum 24h) vs débit
df["rain_sum_24h"] = df["rain"].rolling(24).sum()
roll_corr = (df["debit"]
             .rolling(30 * 24)
             .corr(df["rain_sum_24h"]))

mean_corr_pre  = roll_corr[df_pre.index].mean()
mean_corr_post = roll_corr[df_post.index].mean()
print(f"  Corrélation glissante 30j moyenne PRE  : {mean_corr_pre:.3f}")
print(f"  Corrélation glissante 30j moyenne POST : {mean_corr_post:.3f}")

# ══════════════════════════════════════════════════════════════
#  5. GRAPHIQUES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("5. GÉNÉRATION DES GRAPHIQUES")
print("=" * 65)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Station P — Justification de l'exclusion des données pré-shift",
             fontsize=14, fontweight="bold", y=0.98)

gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32,
              top=0.94, bottom=0.07, left=0.07, right=0.97)

patch_pre  = mpatches.Patch(color=COL_PRE,  label=f"Pré-shift  ({df_pre.index.min().date()} → {df_pre.index.max().date()})")
patch_post = mpatches.Patch(color=COL_POST, label=f"Post-shift ({df_post.index.min().date()} → {df_post.index.max().date()})")

# ── (A) Série temporelle complète ──────────────────────────
ax_ts = fig.add_subplot(gs[0, :])
ax_ts.fill_between(df_pre.index,  df_pre["debit"],  alpha=0.5, color=COL_PRE,  label="_nolegend_")
ax_ts.fill_between(df_post.index, df_post["debit"], alpha=0.5, color=COL_POST, label="_nolegend_")
ax_ts.axvline(pd.Timestamp(SHIFT_DATE), color="red", lw=2, ls="--", label="Shift 2025-04-03")

# Moyennes
ax_ts.axhline(df_pre["debit"].mean(),  color=COL_PRE,  lw=1.5, ls=":",
              label=f"Moyenne pré   = {df_pre['debit'].mean():.0f} m³/h")
ax_ts.axhline(df_post["debit"].mean(), color=COL_POST, lw=1.5, ls=":",
              label=f"Moyenne post = {df_post['debit'].mean():.0f} m³/h")

ax_ts.set_title("(A) Série temporelle complète du débit — rupture visible au 2025-04-03",
                fontsize=11)
ax_ts.set_ylabel("Débit (m³/h)")
ax_ts.legend(fontsize=9, loc="upper left")
ax_ts.set_xlim(df.index.min(), df.index.max())

# ── (B) Histogrammes ────────────────────────────────────────
ax_hist = fig.add_subplot(gs[1, 0])
bins = np.linspace(0, max(df_pre["debit"].max(), df_post["debit"].max()), 60)
ax_hist.hist(df_pre["debit"],  bins=bins, color=COL_PRE,  alpha=0.7, density=True, label="Pré-shift")
ax_hist.hist(df_post["debit"], bins=bins, color=COL_POST, alpha=0.7, density=True, label="Post-shift")
ax_hist.set_title("(B) Distribution du débit\n(les deux populations sont distinctes)", fontsize=11)
ax_hist.set_xlabel("Débit (m³/h)")
ax_hist.set_ylabel("Densité")
ax_hist.legend(fontsize=9)

# Annotations moyennes
ax_hist.axvline(df_pre["debit"].mean(),  color=COL_PRE,  lw=2, ls="--")
ax_hist.axvline(df_post["debit"].mean(), color=COL_POST, lw=2, ls="--")
ax_hist.text(df_pre["debit"].mean() + 5,  ax_hist.get_ylim()[1]*0.85,
             f"μ={df_pre['debit'].mean():.0f}", color=COL_PRE, fontsize=9)
ax_hist.text(df_post["debit"].mean() + 5, ax_hist.get_ylim()[1]*0.85,
             f"μ={df_post['debit'].mean():.0f}", color=COL_POST, fontsize=9)

# ── (C) Boxplots ────────────────────────────────────────────
ax_box = fig.add_subplot(gs[1, 1])
bp = ax_box.boxplot(
    [df_pre["debit"].values, df_post["debit"].values],
    labels=["Pré-shift\n(2024-05 → 2025-03)", "Post-shift\n(2025-04 → 2026-01)"],
    patch_artist=True,
    medianprops=dict(color="black", lw=2),
    flierprops=dict(marker=".", markersize=2, alpha=0.3)
)
bp["boxes"][0].set_facecolor(COL_PRE + "99")
bp["boxes"][1].set_facecolor(COL_POST + "99")
ax_box.set_title("(C) Boxplot du débit\n(médianes : 37 m³/h vs 238 m³/h)", fontsize=11)
ax_box.set_ylabel("Débit (m³/h)")

# ── (D) Corrélation pluie→débit par lag ─────────────────────
ax_corr = fig.add_subplot(gs[2, 0])
ax_corr.plot(lags, corr_pre,  color=COL_PRE,  lw=2, marker="o", ms=4, label=f"Pré-shift  (max={corr_pre.max():.3f})")
ax_corr.plot(lags, corr_post, color=COL_POST, lw=2, marker="o", ms=4, label=f"Post-shift (max={corr_post.max():.3f})")
ax_corr.axhline(0, color="gray", lw=0.8, ls="--")
ax_corr.set_title("(D) Corrélation de Pearson  pluie(t−lag) ~ débit(t)\n(pré-shift : quasi nulle ; post-shift : significative)", fontsize=11)
ax_corr.set_xlabel("Lag (heures)")
ax_corr.set_ylabel("Corrélation de Pearson r")
ax_corr.legend(fontsize=9)
ax_corr.set_xlim(0, 72)

# ── (E) Corrélation glissante 30j ───────────────────────────
ax_roll = fig.add_subplot(gs[2, 1])
ax_roll.plot(roll_corr.index, roll_corr.values,
             color="steelblue", lw=1.2, alpha=0.8)
ax_roll.axvline(pd.Timestamp(SHIFT_DATE), color="red", lw=2, ls="--", label="Shift")

# Moyennes par période
ax_roll.axhline(mean_corr_pre,  color=COL_PRE,  lw=1.8, ls="--",
                label=f"Moyenne pré   = {mean_corr_pre:.3f}")
ax_roll.axhline(mean_corr_post, color=COL_POST, lw=1.8, ls="--",
                label=f"Moyenne post = {mean_corr_post:.3f}")

ax_roll.axhline(0, color="gray", lw=0.7, ls=":")
ax_roll.fill_between(df_pre.index,  -0.2, 1, alpha=0.06, color=COL_PRE)
ax_roll.fill_between(df_post.index, -0.2, 1, alpha=0.06, color=COL_POST)
ax_roll.set_title("(E) Corrélation glissante 30 jours\npluie (sum 24h) ~ débit", fontsize=11)
ax_roll.set_ylabel("Pearson r (fenêtre 30j)")
ax_roll.set_xlim(df.index.min(), df.index.max())
ax_roll.legend(fontsize=9, loc="upper left")

# Légende globale
fig.legend(handles=[patch_pre, patch_post], loc="lower center",
           ncol=2, fontsize=10, frameon=True,
           bbox_to_anchor=(0.5, 0.01))

out_path = OUT / "shift_station_P_justification.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  -> {out_path.name}")

# ══════════════════════════════════════════════════════════════
#  6. SYNTHÈSE CHIFFRÉE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("6. SYNTHÈSE — POURQUOI EXCLURE LES DONNÉES PRÉ-SHIFT")
print("=" * 65)

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  ARGUMENT 1 — RUPTURE DE NIVEAU                         │
  │  Débit moyen : {df_pre['debit'].mean():>5.0f} m³/h (pré) → {df_post['debit'].mean():>5.0f} m³/h (post)  │
  │  Médiane     : {df_pre['debit'].median():>5.0f} m³/h (pré) → {df_post['debit'].median():>5.0f} m³/h (post)  │
  │  Le débit est ~5× plus élevé après le shift.            │
  │  ⇒ Les deux périodes ne représentent PAS le même        │
  │     système physique (changement de réseau / capteur).  │
  ├─────────────────────────────────────────────────────────┤
  │  ARGUMENT 2 — CORRÉLATION PLUIE→DÉBIT                  │
  │  Pearson max PRE  : {corr_pre.max():.3f}  (lag={lags[corr_pre.argmax()]}h)           │
  │  Pearson max POST : {corr_post.max():.3f}  (lag={lags[corr_post.argmax()]}h)           │
  │  Avant le shift, la pluie n'explique pas le débit.      │
  │  ⇒ Entraîner un modèle sur la période pré-shift         │
  │     n'apprendrait rien d'utile pour la période post.    │
  ├─────────────────────────────────────────────────────────┤
  │  ARGUMENT 3 — STATIONNARITÉ                             │
  │  La corrélation glissante 30j est instable et souvent   │
  │  négative avant le shift. Le signal est non-stationnaire│
  │  et inexplicable par les features météo disponibles.    │
  ├─────────────────────────────────────────────────────────┤
  │  DÉCISION : garder uniquement POST-SHIFT (>= 2025-04-03)│
  │  → {len(df_post):,} heures de données cohérentes               │
  └─────────────────────────────────────────────────────────┘
""")

print("=" * 65)
print("FIN shift_station_P.py")
print("=" * 65)
