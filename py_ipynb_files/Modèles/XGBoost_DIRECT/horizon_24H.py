#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
horizon_24H.py — Modèle final XGBoost  |  Station P  |  H+24h
===============================================================
Prédiction directe du débit à l'horizon H=24h.

Features retenues (enseignements de quick_ablation_24h.py) :
  - cyclic_custom = #1 (24.3%) : à H=24h on prédit la même heure
    demain → le profil journalier moyen est le meilleur ancre
  - prev_pluie_24h = #2 (7.25%) : pluie oracle à l'instant cible
  - prev_pluie_22/23h dominent les prev_pluie (#3, #4)
  - target_is_weekend = top 20 : savoir si demain est un weekend
    impacte directement le débit prévu
  - G12 (baseflow, lags pluie) ne progresse PAS vs G6 seul
    (MAE CV 47.92 vs 48.67 → G12 non retenu)
  - Naïf H=24h = 57.61 m³/h (faible car daily periodicity aide
    même la simple persistence) → gain vs naïf = ~28-30%

Résultats ablation :
  H12-G12 sans adaptation : MAE CV=56.11  (baseline)
  H24-G6  (G6 adapté H=24h) : MAE CV=47.92  (-8.18) ← retenu
  H24-G12 (G6+lags+baseflow) : MAE CV=48.67  (-7.43) ← non retenu

Pipeline :
  1. Chargement
  2. Feature engineering (G6 adapté H=24h)
  3. GridSearch temporel
  4. Robustesse sliding window CV (6 configs)
  5. Modèle final + évaluation complète
  6. Graphiques (Bland-Altman débit réel en X)
"""

import warnings; warnings.filterwarnings("ignore")
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

try:
    from scipy.optimize import curve_fit; HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
try:
    import holidays as hol_lib; HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

# ══════════════════════════════════════════════════════════════
BASE             = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\bronze\bronze")
OUT              = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\Modèles\XGBoost_DIRECT")
POST_SHIFT_START = "2025-04-03"
HORIZON          = 24
W_TEST_FINAL     = 8 * 7 * 24

# ══════════════════════════════════════════════════════════════
#  1. CHARGEMENT
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("1. CHARGEMENT")
print("=" * 65)

chunks = []
for chunk in pd.read_csv(BASE / "sensor_P_20231110_20260106.csv", chunksize=500_000):
    sub = chunk[chunk["sensor"] == "entry_debit_f1"]
    if len(sub): chunks.append(sub)
df_raw = pd.concat(chunks)
df_raw["ts"] = pd.to_datetime(df_raw["ts"])
debit = (df_raw.groupby("ts")["value"].mean()
         .resample("h").mean().ffill().clip(lower=0))
debit = debit[debit.index >= POST_SHIFT_START]

weather_files = sorted((BASE / "weather_P").glob("*_hourly2.csv"))
if not weather_files:
    weather_files = sorted((BASE / "weather_P").glob("*_hourly.csv"))
dfs_w = []
for f in weather_files:
    d = pd.read_csv(f)
    d["ts"] = (pd.to_datetime(d["date"]).dt.tz_localize("UTC")
               .dt.tz_convert("Europe/Paris").dt.tz_localize(None))
    d = d.set_index("ts")
    dfs_w.append(d[["rain", "temperature_2m"]])
df_weather = pd.concat(dfs_w).groupby(level=0)[["rain", "temperature_2m"]].mean()
df_weather = df_weather[~df_weather.index.duplicated(keep="first")]

df = pd.DataFrame({"debit": debit}).join(df_weather, how="inner")
df["rain"] = df["rain"].fillna(0)
df["temperature_2m"] = df["temperature_2m"].ffill().fillna(15)
print(f"  Débit  : {debit.index.min().date()} -> {debit.index.max().date()}")
print(f"  Météo  : {len(weather_files)} stations | Fusionné : {len(df)} heures")

# ══════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("2. FEATURE ENGINEERING")
print("=" * 65)

h_arr = df.index.hour.astype(float)
SEUIL_CRUE = float(df["debit"].quantile(0.90))

# ── G6 adapté H=24h ───────────────────────────────────────────
df["slope_1h"]     = df["debit"] - df["debit"].shift(1)
df["slope_3h"]     = df["debit"] - df["debit"].shift(3)
df["slope_6h"]     = df["debit"] - df["debit"].shift(6)
df["slope_12h"]    = df["debit"] - df["debit"].shift(12)
df["slope_24h"]    = df["debit"] - df["debit"].shift(24)   # ← H=24h
df["debit_lag1"]   = df["debit"].shift(1)
df["debit_lag3"]   = df["debit"].shift(3)
df["debit_lag6"]   = df["debit"].shift(6)
df["debit_lag12"]  = df["debit"].shift(12)
df["debit_lag24"]  = df["debit"].shift(24)                 # ← H=24h : même heure hier
df["debit_lag48"]  = df["debit"].shift(48)                 # ← même heure avant-hier
df["debit_lag7d"]  = df["debit"].shift(7 * 24)
df["rain_sum_3h"]  = df["rain"].rolling(3).sum()
df["rain_sum_6h"]  = df["rain"].rolling(6).sum()
df["rain_sum_12h"] = df["rain"].rolling(12).sum()
df["rain_sum_24h"] = df["rain"].rolling(24).sum()          # ← H=24h
df["rain_sum_3d"]  = df["rain"].rolling(72).sum()
df["rain_sum_7d"]  = df["rain"].rolling(168).sum()
df["rain_max_3h"]  = df["rain"].rolling(3).max()
df["rain_max_6h"]  = df["rain"].rolling(6).max()
df["rain_max_12h"] = df["rain"].rolling(12).max()
df["rain_max_24h"] = df["rain"].rolling(24).max()          # ← H=24h
df["ressuyage_exp"]   = df["rain"].ewm(halflife=48, adjust=False).mean()
rain_std = df["rain"].std()
df["rain_exp_norm"]   = np.expm1(df["rain"] / rain_std) if rain_std > 0 else 0.
df["is_raining_hard"] = (df["rain"] > 1.0).astype(int)
df["is_weekend"]      = (df.index.dayofweek >= 5).astype(int)
df["is_monday_morning"] = ((df.index.dayofweek == 0) & (df.index.hour < 10)).astype(int)
df["is_holiday"] = 0
fr_hol = None
if HAS_HOLIDAYS:
    fr_hol = hol_lib.France(years=range(2023, 2027))
    df["is_holiday"] = df.index.normalize().isin(fr_hol).astype(int)
df["en_crue"]        = (df["debit"] > SEUIL_CRUE).astype(int)
df["intensite_crue"] = np.maximum(df["debit"] - SEUIL_CRUE, 0.)
df["hour_sin"]       = np.sin(2 * np.pi * h_arr / 24)
df["hour_cos"]       = np.cos(2 * np.pi * h_arr / 24)

CYCLIC_OK = False
if HAS_SCIPY:
    profile_h = df.groupby(df.index.hour)["debit"].mean()
    def cyclic_func(h, beta, alpha, phi, alpha2):
        return beta + alpha * np.cos(h * 2*np.pi/24 + phi)**2 + alpha2 * np.sin(h * 2*np.pi/24 + phi)
    try:
        popt, _ = curve_fit(cyclic_func, profile_h.index.values.astype(float),
                            profile_h.values, p0=[profile_h.mean(), 50., -2.7, 80.], maxfev=20000)
        df["cyclic_custom"] = cyclic_func(h_arr, *popt)
        r2_fit = r2_score(profile_h.values, cyclic_func(profile_h.index.values.astype(float), *popt))
        print(f"  Cyclic fit  R2={r2_fit:.3f}  beta={popt[0]:.1f}  alpha={popt[1]:.1f}")
        CYCLIC_OK = True
    except Exception as e:
        print(f"  Cyclic fit échoué : {e}")
if not CYCLIC_OK:
    df["cyclic_custom"] = df["hour_sin"] * 100 + 200

# Features du JOUR CIBLE (demain à la même heure) ← nouveau H=24h
# Note : target_hour = current_hour (identique, 24h plus tard)
# → target_hour_sin/cos = hour_sin/cos (redondant, non ajouté)
# On ajoute seulement le JOUR cible : weekend ? lundi matin ?
target_ts = df.index + pd.Timedelta(hours=HORIZON)
df["target_is_weekend"]        = (target_ts.dayofweek >= 5).astype(int)
df["target_is_monday_morning"] = (
    (target_ts.dayofweek == 0) & (target_ts.hour < 10)
).astype(int)
df["target_is_holiday"] = 0
if HAS_HOLIDAYS and fr_hol is not None:
    df["target_is_holiday"] = target_ts.normalize().isin(fr_hol).astype(int)

# Pluie future sur tout l'horizon H=24h (toutes les heures)
for i in range(1, 25):
    df[f"prev_pluie_{i}h"] = df["rain"].shift(-i)
df["prev_pluie_cumul_12h"] = sum(df[f"prev_pluie_{i}h"] for i in range(1, 13))
df["prev_pluie_cumul_24h"] = sum(df[f"prev_pluie_{i}h"] for i in range(1, 25))
print(f"  G6 adapté H=24h  |  seuil crue P90 = {SEUIL_CRUE:.1f} m³/h")
print(f"  (G12 non retenu : ablation H24-G6=47.92 < H24-G12=48.67 en MAE CV)")

df["target"] = df["debit"].shift(-HORIZON)

# ── Features finales (G6 uniquement, G12 non retenu) ──────────
FEATURES = [
    # Débit courant et lags
    "debit","debit_lag1","debit_lag3","debit_lag6","debit_lag12",
    "debit_lag24","debit_lag48","debit_lag7d",
    # Dynamique (pentes)
    "slope_1h","slope_3h","slope_6h","slope_12h","slope_24h",
    # Cumuls pluie passée
    "rain_sum_3h","rain_sum_6h","rain_sum_12h","rain_sum_24h","rain_sum_3d","rain_sum_7d",
    "rain_max_3h","rain_max_6h","rain_max_12h","rain_max_24h",
    # Pluie future oracle (couverture complète H=24h)
    "prev_pluie_1h","prev_pluie_2h","prev_pluie_3h","prev_pluie_4h",
    "prev_pluie_5h","prev_pluie_6h","prev_pluie_7h","prev_pluie_8h",
    "prev_pluie_9h","prev_pluie_10h","prev_pluie_11h","prev_pluie_12h",
    "prev_pluie_13h","prev_pluie_14h","prev_pluie_15h","prev_pluie_16h",
    "prev_pluie_17h","prev_pluie_18h","prev_pluie_19h","prev_pluie_20h",
    "prev_pluie_21h","prev_pluie_22h","prev_pluie_23h","prev_pluie_24h",
    "prev_pluie_cumul_12h","prev_pluie_cumul_24h",
    # Profil journalier (dominant à H=24h)
    "hour_sin","hour_cos","cyclic_custom",
    # Calendrier COURANT
    "is_weekend","is_holiday","is_monday_morning",
    # Calendrier CIBLE (demain) ← spécifique H=24h
    "target_is_weekend","target_is_monday_morning","target_is_holiday",
    # Contexte météo et crue
    "ressuyage_exp","rain_exp_norm","is_raining_hard",
    "en_crue","intensite_crue",
]
print(f"  Total : {len(FEATURES)} features (G6 adapté uniquement)")

df_clean = df.dropna(subset=FEATURES + ["target"]).copy()
n = len(df_clean)
df_cv   = df_clean.iloc[: n - W_TEST_FINAL]
df_test = df_clean.iloc[n - W_TEST_FINAL :]
print(f"\n  CV   : {df_cv.index.min().date()} -> {df_cv.index.max().date()}  ({len(df_cv)} h)")
print(f"  Test : {df_test.index.min().date()} -> {df_test.index.max().date()}  ({len(df_test)} h)")

# ══════════════════════════════════════════════════════════════
#  3. GRIDSEARCH TEMPOREL
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("3. GRIDSEARCH TEMPOREL")
print("   (train = 10 premières semaines du CV, val = 2 suivantes)")
print("=" * 65)

W_GS_TRAIN = 10 * 7 * 24
W_GS_VAL   =  2 * 7 * 24
gs_train = df_cv.iloc[:W_GS_TRAIN].dropna(subset=FEATURES + ["target"])
gs_val   = df_cv.iloc[W_GS_TRAIN : W_GS_TRAIN + W_GS_VAL].dropna(subset=FEATURES + ["target"])

PARAM_GRID = {
    "max_depth":        [4, 5, 6],
    "learning_rate":    [0.03, 0.05, 0.10],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
}
N_EST_MAX  = 800
EARLY_STOP = 30

print(f"\n  Grid : 3 × 3 × 2 × 2 = 36 combinaisons\n")
print(f"  {'max_d':>5} {'lr':>5} {'sub':>5} {'col':>5} | {'MAE_val':>8} {'R2_val':>7} {'n_est':>6}")
print("  " + "─" * 55)

best_mae_gs = np.inf
best_params = None
best_n_est  = None

for md, lr, sub, col in product(
        PARAM_GRID["max_depth"], PARAM_GRID["learning_rate"],
        PARAM_GRID["subsample"], PARAM_GRID["colsample_bytree"]):
    m = xgb.XGBRegressor(
        max_depth=md, learning_rate=lr, n_estimators=N_EST_MAX,
        subsample=sub, colsample_bytree=col,
        early_stopping_rounds=EARLY_STOP,
        eval_metric="mae", random_state=42, n_jobs=-1, verbosity=0)
    m.fit(gs_train[FEATURES], gs_train["target"],
          eval_set=[(gs_val[FEATURES], gs_val["target"])],
          verbose=False)
    n_est = m.best_iteration + 1
    preds = m.predict(gs_val[FEATURES]).clip(min=100)
    mae   = mean_absolute_error(gs_val["target"], preds)
    r2    = r2_score(gs_val["target"], preds)
    tag   = " ← best" if mae < best_mae_gs else ""
    print(f"  {md:>5} {lr:>5.2f} {sub:>5.1f} {col:>5.1f} | {mae:>8.2f} {r2:>7.3f} {n_est:>6}{tag}")
    if mae < best_mae_gs:
        best_mae_gs = mae
        best_params = {"max_depth": md, "learning_rate": lr,
                       "subsample": sub, "colsample_bytree": col}
        best_n_est  = n_est

scale_factor      = len(df_cv) / W_GS_TRAIN
best_n_est_scaled = max(300, int(best_n_est * scale_factor))
print(f"\n  Meilleurs paramètres : {best_params}")
print(f"  n_estimators (GS fenêtre)          : {best_n_est}")
print(f"  n_estimators (scaled ×{scale_factor:.1f}, CV complet) : {best_n_est_scaled}")
print(f"  MAE val GridSearch = {best_mae_gs:.2f} m³/h")
best_n_est = best_n_est_scaled

# ══════════════════════════════════════════════════════════════
#  4. ROBUSTESSE — SLIDING WINDOW CV
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("4. ROBUSTESSE — SLIDING WINDOW CV (6 configs de fenêtres)")
print("=" * 65)

WINDOW_CONFIGS = [
    (8  * 7 * 24,  1 * 7 * 24,  1 * 7 * 24),
    (8  * 7 * 24,  2 * 7 * 24,  2 * 7 * 24),
    (10 * 7 * 24,  1 * 7 * 24,  1 * 7 * 24),
    (10 * 7 * 24,  2 * 7 * 24,  2 * 7 * 24),
    (12 * 7 * 24,  2 * 7 * 24,  2 * 7 * 24),
    (12 * 7 * 24,  3 * 7 * 24,  3 * 7 * 24),
]
XGB_BEST = dict(**best_params, n_estimators=best_n_est,
                random_state=42, n_jobs=-1, verbosity=0)

def run_sliding_cv(df_data, features, w_train, w_val, step):
    results = []
    start = 0
    ndata = len(df_data)
    while start + w_train + w_val <= ndata:
        tr = df_data.iloc[start : start + w_train].dropna(subset=features + ["target"])
        va = df_data.iloc[start + w_train : start + w_train + w_val].dropna(subset=features + ["target"])
        if len(tr) < 200 or len(va) < 20:
            start += step; continue
        m = xgb.XGBRegressor(**XGB_BEST)
        m.fit(tr[features], tr["target"], verbose=False)
        preds  = m.predict(va[features]).clip(min=100)
        mae    = mean_absolute_error(va["target"], preds)
        rmse   = np.sqrt(mean_squared_error(va["target"], preds))
        r2     = r2_score(va["target"], preds)
        mask_p = va["target"] > SEUIL_CRUE
        mae_p  = mean_absolute_error(va["target"][mask_p], preds[mask_p]) if mask_p.sum() > 0 else np.nan
        results.append({"mae": mae, "rmse": rmse, "r2": r2, "mae_pics": mae_p})
        start += step
    return pd.DataFrame(results)

print(f"\n  {'W_train':>8} {'W_val':>6} {'Step':>6} | {'Folds':>5} | "
      f"{'MAE moy':>8} {'MAE std':>8} {'R2 moy':>7} | {'MAEpics':>8}")
print("  " + "─" * 75)

robustness_results = []
for (w_tr, w_va, step) in WINDOW_CONFIGS:
    df_r = run_sliding_cv(df_cv.dropna(subset=FEATURES + ["target"]), FEATURES, w_tr, w_va, step)
    if len(df_r) == 0: continue
    mae_mean = df_r["mae"].mean(); mae_std = df_r["mae"].std()
    r2_mean  = df_r["r2"].mean();  maep    = df_r["mae_pics"].mean()
    tag = " ← ref" if (w_tr == 10*7*24 and w_va == 2*7*24) else ""
    print(f"  {w_tr//168:>5}sem {w_va//168:>4}sem {step//168:>4}sem | {len(df_r):>5} | "
          f"{mae_mean:>8.2f} {mae_std:>8.2f} {r2_mean:>7.3f} | {maep:>8.2f}{tag}")
    robustness_results.append({"label": f"{w_tr//168}s/{w_va//168}s",
                                "mae_mean": mae_mean, "mae_std": mae_std,
                                "r2_mean": r2_mean, "maep_mean": maep,
                                "folds": len(df_r), "fold_maes": df_r["mae"].values})

# ══════════════════════════════════════════════════════════════
#  5. MODELE FINAL — EVALUATION TEST SET
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("5. MODELE FINAL — EVALUATION TEST SET")
print("=" * 65)

df_cv_f = df_cv.dropna(subset=FEATURES + ["target"])
df_te_f = df_test.dropna(subset=FEATURES + ["target"])

model_final = xgb.XGBRegressor(**XGB_BEST)
model_final.fit(df_cv_f[FEATURES], df_cv_f["target"], verbose=False)

preds_tr = model_final.predict(df_cv_f[FEATURES]).clip(min=100)
preds_te = model_final.predict(df_te_f[FEATURES]).clip(min=100)
actuals  = df_te_f["target"].values
errors   = actuals - preds_te
abs_err  = np.abs(errors)

mae_tr   = mean_absolute_error(df_cv_f["target"], preds_tr)
mae_te   = mean_absolute_error(actuals, preds_te)
rmse_te  = np.sqrt(mean_squared_error(actuals, preds_te))
r2_te    = r2_score(actuals, preds_te)
mae_naif = mean_absolute_error(actuals, df_te_f["debit"].clip(lower=100).values)

mask_p   = actuals > SEUIL_CRUE
mae_pics = mean_absolute_error(actuals[mask_p], preds_te[mask_p]) if mask_p.sum() > 0 else np.nan
n_pics   = mask_p.sum()

mask_rain = df_te_f["rain"] > 1.0
mae_pluie = mean_absolute_error(actuals[mask_rain], preds_te[mask_rain]) if mask_rain.sum() > 0 else np.nan

# Bland-Altman
diff_ba = preds_te - actuals
mean_ba = actuals
bias_ba = diff_ba.mean()
std_ba  = diff_ba.std()
loa_lo  = bias_ba - 1.96 * std_ba
loa_hi  = bias_ba + 1.96 * std_ba
pct_loa = ((diff_ba >= loa_lo) & (diff_ba <= loa_hi)).mean() * 100

print(f"\n  ┌─────────────────────────────────────────────────┐")
print(f"  │  MÉTRIQUES GLOBALES                             │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  MAE  train    = {mae_tr:>6.2f} m³/h                    │")
print(f"  │  MAE  test     = {mae_te:>6.2f} m³/h                    │")
print(f"  │  RMSE test     = {rmse_te:>6.2f} m³/h                    │")
print(f"  │  R²   test     = {r2_te:>6.3f}                          │")
print(f"  │  MAE naïf      = {mae_naif:>6.2f} m³/h  (persist. H=24)  │")
print(f"  │  Gain vs naïf  = {(1-mae_te/mae_naif)*100:>6.1f} %                       │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  MÉTRIQUES CRUES  (seuil P90 = {SEUIL_CRUE:.0f} m³/h)       │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  MAEpics       = {mae_pics:>6.2f} m³/h  (n={n_pics})             │")
print(f"  │  MAEpluie      = {mae_pluie:>6.2f} m³/h  (rain > 1mm)       │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  BLAND-ALTMAN                                   │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  Biais         = {bias_ba:>6.2f} m³/h                    │")
print(f"  │  LoA inférieure= {loa_lo:>6.1f} m³/h                    │")
print(f"  │  LoA supérieure= {loa_hi:>+6.1f} m³/h                    │")
print(f"  │  % dans LoA    = {pct_loa:>6.1f} %                       │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  DISTRIBUTION ERREURS                           │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  Médiane |err| = {np.median(abs_err):>6.2f} m³/h                    │")
print(f"  │  P90 |err|     = {np.percentile(abs_err,90):>6.2f} m³/h                    │")
print(f"  │  P95 |err|     = {np.percentile(abs_err,95):>6.2f} m³/h                    │")
print(f"  │  % sous-estim  = {(errors < 0).mean()*100:>6.1f} %                       │")
print(f"  └─────────────────────────────────────────────────┘")

# MAE par intensité de pluie
bins_rain = [(0, 0.1, "Sec   (<0.1mm)"),
             (0.1, 1.0, "Légère (0.1-1mm)"),
             (1.0, 3.0, "Modérée (1-3mm)"),
             (3.0, 999, "Forte  (>3mm)")]
print(f"\n  MAE par intensité pluie :")
for lo, hi, lbl in bins_rain:
    mask = (df_te_f["rain"] >= lo) & (df_te_f["rain"] < hi)
    if mask.sum() > 0:
        print(f"    {lbl:22s}: MAE={mean_absolute_error(actuals[mask], preds_te[mask]):.1f}  n={mask.sum()}")

# MAE par régime de débit
bins_deb = [(0, 200, "Faible"), (200, 350, "Moyen"), (350, 500, "Élevé"), (500, 9999, "Crue")]
print(f"\n  MAE par régime de débit :")
for lo, hi, lbl in bins_deb:
    mask = (actuals >= lo) & (actuals < hi)
    if mask.sum() > 0:
        print(f"    {lbl:10s}: MAE={mean_absolute_error(actuals[mask], preds_te[mask]):.1f}  n={mask.sum()}")

# MAE par mois
print(f"\n  MAE par mois (test) :")
for m_num in sorted(df_te_f.index.month.unique()):
    mask = df_te_f.index.month == m_num
    if mask.sum() > 0:
        mname = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"][m_num-1]
        print(f"    {mname} : MAE={mean_absolute_error(actuals[mask], preds_te[mask]):.1f}  n={mask.sum()}")

# Top 10 pires erreurs
print(f"\n  Top 10 pires erreurs :")
worst_idx = np.argsort(abs_err)[-10:][::-1]
for i in worst_idx:
    ts   = df_te_f.index[i]
    r24  = df_te_f["debit_rise_24h"].iloc[i] if "debit_rise_24h" in df_te_f else float("nan")
    print(f"    {ts.strftime('%d/%m %Hh')}  reel={int(actuals[i])}  pred={int(preds_te[i])}  "
          f"err={int(abs_err[i])}  pluie={df_te_f['rain'].iloc[i]:.1f}mm  "
          f"rise24h={r24:.0f}" if not np.isnan(r24) else
          f"    {ts.strftime('%d/%m %Hh')}  reel={int(actuals[i])}  pred={int(preds_te[i])}  err={int(abs_err[i])}")

# Top 20 features
imp_ser = pd.Series(model_final.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n  Top 20 features :")
for feat, val in imp_ser.head(20).items():
    bar = "#" * int(val * 400)
    tag = ""
    if "target_" in feat: tag = " ← TARGET DAY"
    print(f"    {feat:35s} {val:.4f}  {bar}{tag}")

# ══════════════════════════════════════════════════════════════
#  6. GRAPHIQUES
# ══════════════════════════════════════════════════════════════
print(f"\nGeneration des graphiques...")

fig = plt.figure(figsize=(20, 24))
gs  = GridSpec(4, 2, figure=fig, hspace=0.40, wspace=0.30)

# ── (A) Série temporelle test ─────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
idx_te = df_te_f.index
ax0.plot(idx_te, actuals,  color="#1f77b4", lw=1.2, label="Réel")
ax0.plot(idx_te, preds_te, color="#ff7f0e", lw=1.0, alpha=0.8, label="Prédit H=24h")
ax0.axhline(SEUIL_CRUE, color="red", lw=0.8, ls="--", alpha=0.5, label=f"P90={SEUIL_CRUE:.0f}")
ax0.set_title("Série temporelle — Test set (H=24h)", fontsize=13, fontweight="bold")
ax0.set_ylabel("Débit (m³/h)"); ax0.legend(loc="upper left"); ax0.grid(True, alpha=0.3)

# ── (B) Scatter réel vs prédit ────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
ax1.scatter(actuals, preds_te, alpha=0.3, s=8, color="#1f77b4")
lim = max(actuals.max(), preds_te.max()) * 1.05
ax1.plot([100, lim], [100, lim], "r--", lw=1.2)
ax1.set_xlabel("Débit réel (m³/h)"); ax1.set_ylabel("Débit prédit (m³/h)")
ax1.set_title(f"Réel vs Prédit  (R²={r2_te:.3f}, MAE={mae_te:.1f})", fontweight="bold")
ax1.grid(True, alpha=0.3)

# ── (C) Bland-Altman (X = débit réel) ────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
sc = ax2.scatter(mean_ba, diff_ba, alpha=0.25, s=8,
                 c=actuals, cmap="RdYlBu_r", vmin=100, vmax=700)
plt.colorbar(sc, ax=ax2, label="Débit réel (m³/h)")
ax2.axhline(bias_ba, color="red",    lw=1.5, label=f"Biais={bias_ba:.1f}")
ax2.axhline(loa_lo,  color="orange", lw=1.2, ls="--", label=f"LoA±={loa_lo:.0f}/{loa_hi:.0f}")
ax2.axhline(loa_hi,  color="orange", lw=1.2, ls="--")
ax2.axhline(0, color="gray", lw=0.8, ls=":")
ax2.set_xlabel("Débit réel (m³/h)"); ax2.set_ylabel("Prédit − Réel (m³/h)")
ax2.set_title(f"Bland-Altman  ({pct_loa:.1f}% dans LoA)", fontweight="bold")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# ── (D) Distribution des erreurs absolues ────────────────────
ax3 = fig.add_subplot(gs[2, 0])
ax3.hist(abs_err, bins=60, color="#1f77b4", alpha=0.75, edgecolor="none")
for pct, col, lbl in [(50, "red", f"Médiane={np.median(abs_err):.0f}"),
                       (90, "orange", f"P90={np.percentile(abs_err,90):.0f}"),
                       (95, "purple", f"P95={np.percentile(abs_err,95):.0f}")]:
    ax3.axvline(np.percentile(abs_err, pct), color=col, lw=1.5, ls="--", label=lbl)
ax3.set_xlabel("Erreur absolue (m³/h)"); ax3.set_ylabel("Nb observations")
ax3.set_title("Distribution des erreurs absolues", fontweight="bold")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# ── (E) Importances features (top 20) ────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
top20 = imp_ser.head(20)[::-1]
colors = ["#d62728" if "target_" in f else "#1f77b4" for f in top20.index]
ax4.barh(range(len(top20)), top20.values, color=colors)
ax4.set_yticks(range(len(top20)))
ax4.set_yticklabels(top20.index, fontsize=7)
ax4.set_xlabel("Importance (gain)")
ax4.set_title("Top 20 features", fontweight="bold")
patch_tgt = mpatches.Patch(color="#d62728", label="Target day feature")
patch_std = mpatches.Patch(color="#1f77b4", label="Autre feature")
ax4.legend(handles=[patch_tgt, patch_std], fontsize=7)
ax4.grid(True, alpha=0.3, axis="x")

# ── (F) Robustesse CV ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
if robustness_results:
    lbls = [r["label"] for r in robustness_results]
    means = [r["mae_mean"] for r in robustness_results]
    stds  = [r["mae_std"]  for r in robustness_results]
    ax5.bar(range(len(lbls)), means, yerr=stds, capsize=4,
            color="#2ca02c", alpha=0.75, ecolor="black")
    ax5.set_xticks(range(len(lbls))); ax5.set_xticklabels(lbls, rotation=30, ha="right", fontsize=8)
    ax5.set_ylabel("MAE CV (m³/h)"); ax5.set_title("Robustesse — Sliding Window CV", fontweight="bold")
    ax5.grid(True, alpha=0.3, axis="y")

# ── (G) MAE par heure du jour (target hour = current hour à H=24) ─
ax6 = fig.add_subplot(gs[3, 1])
df_err_h = pd.DataFrame({"hour": df_te_f.index.hour, "abs_err": abs_err})
mae_by_h = df_err_h.groupby("hour")["abs_err"].mean()
ax6.bar(mae_by_h.index, mae_by_h.values, color="#9467bd", alpha=0.75)
ax6.set_xlabel("Heure de la prédiction (= heure cible)"); ax6.set_ylabel("MAE (m³/h)")
ax6.set_title("MAE par heure du jour  (H=24h : heure cible = heure actuelle)", fontweight="bold")
ax6.grid(True, alpha=0.3, axis="y")

out_path = OUT / "horizon_24H_evaluation.png"
fig.savefig(out_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  -> {out_path}")

print("\n" + "=" * 65)
print("FIN horizon_24H.py")
print("=" * 65)
