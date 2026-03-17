#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
horizon_12H.py — Modèle final XGBoost  |  Station P  |  H+12h
===============================================================
Prédiction directe du débit à l'horizon H=12h.

Features adaptées à H=12h (enseignements de quick_ablation_12h.py) :
  - debit_lag12 devient la feature #1 (9.1%) → débit 12h avant
    = état de départ exact pour la prédiction à H=12h
  - cyclic_custom revient au #2 (8.5%) → à H=12h on traverse
    un demi-cycle journalier, la périodicité reprend de l'importance
  - prev_pluie_12h = #3 (7.2%) → pluie à l'instant exact de la prédiction
  - Baseflow (debit_min_12h) perd de l'importance vs H=6h (#1 → #19)
    car debit_lag12 capture déjà cette info plus directement

Résultats ablation :
  H6-G12 sans adaptation H=12h : MAE CV=54.66  (baseline)
  H12-G6  (G6 adapté H=12h)   : MAE CV=49.22  (-5.44) ← gros gain
  H12-G12 (G6+lags+baseflow)  : MAE CV=48.92  (-5.74) ← retenu

Pipeline :
  1. Chargement
  2. Feature engineering (G6 adapté H=12h + G12 élargi)
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
HORIZON          = 12
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

# ── G6 adapté H=12h ──────────────────────────────────────────
df["slope_1h"]          = df["debit"] - df["debit"].shift(1)
df["slope_3h"]          = df["debit"] - df["debit"].shift(3)
df["slope_6h"]          = df["debit"] - df["debit"].shift(6)
df["slope_12h"]         = df["debit"] - df["debit"].shift(12)
df["slope_24h"]         = df["debit"] - df["debit"].shift(24)  # ← H=12h
df["debit_lag1"]        = df["debit"].shift(1)
df["debit_lag3"]        = df["debit"].shift(3)
df["debit_lag6"]        = df["debit"].shift(6)
df["debit_lag12"]       = df["debit"].shift(12)                # ← H=12h : #1 feature !
df["debit_lag24"]       = df["debit"].shift(24)
df["debit_lag48"]       = df["debit"].shift(48)
df["debit_lag7d"]       = df["debit"].shift(7 * 24)
df["rain_sum_3h"]       = df["rain"].rolling(3).sum()
df["rain_sum_6h"]       = df["rain"].rolling(6).sum()
df["rain_sum_12h"]      = df["rain"].rolling(12).sum()         # ← H=12h
df["rain_sum_24h"]      = df["rain"].rolling(24).sum()
df["rain_sum_3d"]       = df["rain"].rolling(72).sum()
df["rain_sum_7d"]       = df["rain"].rolling(168).sum()
df["rain_max_3h"]       = df["rain"].rolling(3).max()
df["rain_max_6h"]       = df["rain"].rolling(6).max()
df["rain_max_12h"]      = df["rain"].rolling(12).max()         # ← H=12h
df["ressuyage_exp"]     = df["rain"].ewm(halflife=48, adjust=False).mean()
rain_std = df["rain"].std()
df["rain_exp_norm"]     = np.expm1(df["rain"] / rain_std) if rain_std > 0 else 0.
df["is_raining_hard"]   = (df["rain"] > 1.0).astype(int)
df["is_weekend"]        = (df.index.dayofweek >= 5).astype(int)
df["is_monday_morning"] = ((df.index.dayofweek == 0) & (df.index.hour < 10)).astype(int)
df["is_holiday"]        = 0
if HAS_HOLIDAYS:
    fr_hol = hol_lib.France(years=range(2023, 2027))
    df["is_holiday"] = df.index.normalize().isin(fr_hol).astype(int)
df["en_crue"]           = (df["debit"] > SEUIL_CRUE).astype(int)
df["intensite_crue"]    = np.maximum(df["debit"] - SEUIL_CRUE, 0.)
df["hour_sin"]          = np.sin(2 * np.pi * h_arr / 24)
df["hour_cos"]          = np.cos(2 * np.pi * h_arr / 24)

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

# Pluie future sur tout l'horizon H=12h
for i in range(1, 13):
    df[f"prev_pluie_{i}h"] = df["rain"].shift(-i)
df["prev_pluie_cumul_6h"]  = sum(df[f"prev_pluie_{i}h"] for i in range(1, 7))
df["prev_pluie_cumul_12h"] = sum(df[f"prev_pluie_{i}h"] for i in range(1, 13))
print(f"  G6 adapté H=12h  |  seuil crue P90 = {SEUIL_CRUE:.1f} m³/h")

# ── G12 élargi H=12h ─────────────────────────────────────────
for lag in [6, 9, 12, 15, 18, 24, 30, 36, 48, 60, 72, 84, 96]:
    df[f"rain_lag_{lag}h"] = df["rain"].shift(lag)

for win in [6, 12, 24, 48, 72]:
    df[f"debit_min_{win}h"]  = df["debit"].rolling(win).min()
    df[f"debit_rise_{win}h"] = df["debit"] - df[f"debit_min_{win}h"]

df["rain_lag_24h_x_rise_12h"] = df["rain_lag_24h"] * df["debit_rise_12h"]
df["rain_lag_48h_x_rise_24h"] = df["rain_lag_48h"] * df["debit_rise_24h"]
# Signal crue retardée adapté H=12h (seuil montée sur 24h, pas 12h)
df["crue_retardee_signal"]    = (
    (df["debit_rise_24h"] > 120) & (df["rain_sum_12h"] < 2.0)
).astype(float)
print(f"  G12 élargi H=12h  (lags jusqu'à 96h, baseflow jusqu'à 72h)")

df["target"] = df["debit"].shift(-HORIZON)

# ── Features finales ──────────────────────────────────────────
G6_FEATS = [
    "debit","debit_lag1","debit_lag3","debit_lag6","debit_lag12",
    "debit_lag24","debit_lag48","debit_lag7d",
    "slope_1h","slope_3h","slope_6h","slope_12h","slope_24h",
    "rain_sum_3h","rain_sum_6h","rain_sum_12h","rain_sum_24h","rain_sum_3d","rain_sum_7d",
    "rain_max_3h","rain_max_6h","rain_max_12h",
    "prev_pluie_1h","prev_pluie_2h","prev_pluie_3h",
    "prev_pluie_4h","prev_pluie_5h","prev_pluie_6h",
    "prev_pluie_7h","prev_pluie_8h","prev_pluie_9h",
    "prev_pluie_10h","prev_pluie_11h","prev_pluie_12h",
    "prev_pluie_cumul_6h","prev_pluie_cumul_12h",
    "hour_sin","hour_cos","cyclic_custom",
    "ressuyage_exp","rain_exp_norm","is_raining_hard",
    "is_weekend","is_holiday","is_monday_morning",
    "en_crue","intensite_crue",
]
G12_FEATS = [
    "rain_lag_6h","rain_lag_9h","rain_lag_12h","rain_lag_15h",
    "rain_lag_18h","rain_lag_24h","rain_lag_30h","rain_lag_36h",
    "rain_lag_48h","rain_lag_60h","rain_lag_72h","rain_lag_84h","rain_lag_96h",
    "debit_min_6h","debit_min_12h","debit_min_24h","debit_min_48h","debit_min_72h",
    "debit_rise_6h","debit_rise_12h","debit_rise_24h","debit_rise_48h","debit_rise_72h",
    "rain_lag_24h_x_rise_12h","rain_lag_48h_x_rise_24h",
    "crue_retardee_signal",
]
FEATURES = G6_FEATS + G12_FEATS
print(f"  Total : {len(FEATURES)} features  ({len(G6_FEATS)} G6 + {len(G12_FEATS)} G12)")

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
mask_rain = df_te_f["rain"].values > 1.0
mae_rain  = mean_absolute_error(actuals[mask_rain], preds_te[mask_rain]) if mask_rain.sum() > 0 else np.nan

diffs  = preds_te - actuals
bias   = diffs.mean(); std_d = diffs.std()
loa_up = bias + 1.96 * std_d; loa_dn = bias - 1.96 * std_d
pct_in = ((diffs >= loa_dn) & (diffs <= loa_up)).mean() * 100
p90_err = np.percentile(abs_err, 90); p95_err = np.percentile(abs_err, 95)
pct_sous = (errors > 0).mean() * 100

print(f"""
  ┌─────────────────────────────────────────────────┐
  │  MÉTRIQUES GLOBALES                             │
  ├─────────────────────────────────────────────────┤
  │  MAE  train    = {mae_tr:>6.2f} m³/h                    │
  │  MAE  test     = {mae_te:>6.2f} m³/h                    │
  │  RMSE test     = {rmse_te:>6.2f} m³/h                    │
  │  R²   test     = {r2_te:>6.3f}                          │
  │  MAE naïf      = {mae_naif:>6.2f} m³/h  (persist. H=12)  │
  │  Gain vs naïf  = {(mae_naif-mae_te)/mae_naif*100:>6.1f} %                       │
  ├─────────────────────────────────────────────────┤
  │  MÉTRIQUES CRUES  (seuil P90 = {SEUIL_CRUE:.0f} m³/h)       │
  ├─────────────────────────────────────────────────┤
  │  MAEpics       = {mae_pics:>6.2f} m³/h  (n={n_pics})             │
  │  MAEpluie      = {mae_rain:>6.2f} m³/h  (rain > 1mm)       │
  ├─────────────────────────────────────────────────┤
  │  BLAND-ALTMAN                                   │
  ├─────────────────────────────────────────────────┤
  │  Biais         = {bias:>+6.2f} m³/h                    │
  │  LoA inférieure= {loa_dn:>+6.1f} m³/h                    │
  │  LoA supérieure= {loa_up:>+6.1f} m³/h                    │
  │  % dans LoA    = {pct_in:>6.1f} %                       │
  ├─────────────────────────────────────────────────┤
  │  DISTRIBUTION ERREURS                           │
  ├─────────────────────────────────────────────────┤
  │  Médiane |err| = {np.median(abs_err):>6.2f} m³/h                    │
  │  P90 |err|     = {p90_err:>6.2f} m³/h                    │
  │  P95 |err|     = {p95_err:>6.2f} m³/h                    │
  │  % sous-estim  = {pct_sous:>6.1f} %                       │
  └─────────────────────────────────────────────────┘
""")

month_names = {1:"Jan",2:"Fev",3:"Mar",4:"Avr",5:"Mai",6:"Jun",
               7:"Jul",8:"Aou",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

print("  MAE par intensité pluie :")
for label, mask_r in [("Sec   (<0.1mm)", df_te_f["rain"]<0.1),
                       ("Légère (0.1-1mm)", (df_te_f["rain"]>=0.1)&(df_te_f["rain"]<1)),
                       ("Modérée (1-3mm)", (df_te_f["rain"]>=1)&(df_te_f["rain"]<3)),
                       ("Forte  (>3mm)", df_te_f["rain"]>=3)]:
    sub = abs_err[mask_r.values]
    if len(sub): print(f"    {label:<20}: MAE={sub.mean():.1f}  n={len(sub)}")

q25, q75 = np.percentile(actuals, [25, 75])
print("\n  MAE par régime de débit :")
for label, lo, hi in [("Faible", 0, q25), ("Moyen", q25, q75),
                       ("Élevé", q75, SEUIL_CRUE), ("Crue", SEUIL_CRUE, 1e9)]:
    m = (actuals >= lo) & (actuals < hi)
    sub = abs_err[m]
    if len(sub): print(f"    {label:<8}: MAE={sub.mean():.1f}  n={len(sub)}")

df_eval = pd.DataFrame({
    "reel": actuals, "pred": preds_te, "err": errors, "abs_err": abs_err,
    "rain": df_te_f["rain"].values,
    "en_crue": (actuals > SEUIL_CRUE).astype(int),
    "cr_signal": df_te_f["crue_retardee_signal"].values,
    "rise_24h": df_te_f["debit_rise_24h"].values,
}, index=df_te_f.index)
df_eval["month"] = df_eval.index.month
df_eval["hour"]  = df_eval.index.hour

print("\n  MAE par mois (test) :")
for m, grp in df_eval.groupby("month"):
    print(f"    {month_names[m]:<4}: MAE={grp['abs_err'].mean():.1f}  n={len(grp)}")

print("\n  Top 10 pires erreurs :")
for ts, row in df_eval.nlargest(10, "abs_err").iterrows():
    print(f"    {ts.strftime('%d/%m %Hh')}  "
          f"reel={row['reel']:.0f}  pred={row['pred']:.0f}  "
          f"err={row['abs_err']:.0f}  "
          f"pluie={row['rain']:.1f}mm  rise24h={row['rise_24h']:.0f}")

fi = (pd.DataFrame({"feature": FEATURES, "importance": model_final.feature_importances_})
      .sort_values("importance", ascending=False).reset_index(drop=True))
print("\n  Top 20 features :")
for _, row in fi.head(20).iterrows():
    bar = "#" * int(row["importance"] * 40 / fi["importance"].max())
    tag = " ← G12" if row["feature"] in G12_FEATS else ""
    print(f"    {row['feature']:<35} {row['importance']:.4f}  {bar}{tag}")

# ══════════════════════════════════════════════════════════════
#  6. GRAPHIQUES
# ══════════════════════════════════════════════════════════════
print("\nGeneration des graphiques...")

fig = plt.figure(figsize=(18, 22))
gs_layout = GridSpec(4, 2, figure=fig, hspace=0.38, wspace=0.3)
fig.suptitle(
    f"XGBoost DIRECT H=12h — Station P\n"
    f"MAE={mae_te:.1f} m³/h | R²={r2_te:.3f} | MAEpics={mae_pics:.1f} m³/h | "
    f"Gain vs naïf={((mae_naif-mae_te)/mae_naif*100):.1f}%",
    fontsize=14, fontweight="bold")

ax1 = fig.add_subplot(gs_layout[0, :])
ax1.plot(df_eval.index, df_eval["reel"], "k-", lw=0.9, alpha=0.8, label="Débit réel")
ax1.plot(df_eval.index, df_eval["pred"], color="#9b59b6", lw=1.3, alpha=0.9, label="Prédit H+12h")
ax1.fill_between(df_eval.index, df_eval["reel"], df_eval["pred"],
                 where=df_eval["err"]>0, alpha=0.25, color="#e74c3c", label="Sous-estim.")
ax1.fill_between(df_eval.index, df_eval["reel"], df_eval["pred"],
                 where=df_eval["err"]<0, alpha=0.20, color="#3498db", label="Sur-estim.")
cr_pts = df_eval[df_eval["cr_signal"]==1]
if len(cr_pts) > 0:
    ax1.scatter(cr_pts.index, cr_pts["reel"], marker="^", s=18, color="orange",
                zorder=5, alpha=0.7, label=f"Signal crue retardée (n={len(cr_pts)})")
ax1.axhline(SEUIL_CRUE, color="purple", ls="--", lw=1, alpha=0.6, label=f"Seuil P90={SEUIL_CRUE:.0f}")
ax1.set_title("Série temporelle — Test set (nov 2025 – jan 2026)", fontsize=11)
ax1.set_ylabel("Débit (m³/h)"); ax1.legend(fontsize=7, ncol=3); ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs_layout[1, 0])
mae_h = df_eval.groupby("hour")["abs_err"].mean()
ax2.bar(mae_h.index, mae_h.values, color="#9b59b6", alpha=0.8)
ax2.axhline(mae_te, color="red", ls="--", lw=1.5, label=f"MAE globale={mae_te:.1f}")
ax2.set_title("MAE par heure de la journée"); ax2.set_xlabel("Heure")
ax2.set_ylabel("MAE (m³/h)"); ax2.set_xticks(range(0,24,3))
ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")

ax3 = fig.add_subplot(gs_layout[1, 1])
if robustness_results:
    bp = ax3.boxplot([r["fold_maes"] for r in robustness_results],
                     labels=[r["label"] for r in robustness_results],
                     patch_artist=True, medianprops=dict(color="red", lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#9b59b6"); patch.set_alpha(0.5)
    ax3.axhline(mae_te, color="red", ls="--", lw=1, label=f"MAE test={mae_te:.1f}")
    ax3.set_title("Robustesse — MAE CV par config de fenêtre\n(Wtrain/Wval en semaines)")
    ax3.set_ylabel("MAE (m³/h)"); ax3.set_xlabel("W_train / W_val")
    ax3.legend(); ax3.grid(True, alpha=0.3, axis="y")

ax4 = fig.add_subplot(gs_layout[2, 0])
regime_labels = ["Faible", "Moyen", "Élevé", "Crue"]
regime_bounds = [(0, q25), (q25, q75), (q75, SEUIL_CRUE), (SEUIL_CRUE, 1e9)]
regime_maes, regime_ns = [], []
for lo, hi in regime_bounds:
    m = (actuals >= lo) & (actuals < hi)
    regime_maes.append(abs_err[m].mean() if m.sum() > 0 else 0)
    regime_ns.append(m.sum())
bars = ax4.bar(regime_labels, regime_maes,
               color=["#2ecc71","#f39c12","#e74c3c","#8e44ad"], alpha=0.8)
for bar, n_r in zip(bars, regime_ns):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"n={n_r}", ha="center", va="bottom", fontsize=8)
ax4.axhline(mae_te, color="black", ls="--", lw=1.5, label=f"MAE={mae_te:.1f}")
ax4.set_title("MAE par régime de débit (test)"); ax4.set_ylabel("MAE (m³/h)")
ax4.legend(); ax4.grid(True, alpha=0.3, axis="y")

ax5 = fig.add_subplot(gs_layout[2, 1])
fi_top = fi.head(20)
colors_fi = ["#27ae60" if f in G12_FEATS else "#9b59b6" for f in fi_top["feature"]]
ax5.barh(fi_top["feature"][::-1], fi_top["importance"][::-1],
         color=colors_fi[::-1], alpha=0.8)
ax5.legend(handles=[mpatches.Patch(color="#27ae60", alpha=0.7, label="G12 (lags+baseflow)"),
                    mpatches.Patch(color="#9b59b6", alpha=0.7, label="G6 adapté H=12h")])
ax5.set_title("Feature Importance (top 20)"); ax5.set_xlabel("Importance")
ax5.grid(True, alpha=0.3, axis="x")

ax6 = fig.add_subplot(gs_layout[3, 0])
top20 = df_eval.nlargest(20, "abs_err").copy()
ax6.barh(range(20), top20["abs_err"].values,
         color=["#8e44ad" if r else "#e74c3c" for r in top20["en_crue"]], alpha=0.8)
ax6.set_yticks(range(20))
ax6.set_yticklabels(top20.index.strftime("%d/%m %Hh"), fontsize=7)
for i, (_, row) in enumerate(top20.iterrows()):
    ax6.text(row["abs_err"]+1, i, f"R={row['reel']:.0f} P={row['pred']:.0f}",
             va="center", fontsize=6.5)
ax6.legend(handles=[mpatches.Patch(color="#8e44ad", alpha=0.7, label=f"En crue (>{SEUIL_CRUE:.0f})"),
                    mpatches.Patch(color="#e74c3c", alpha=0.7, label="Hors crue")])
ax6.set_title("Top 20 pires erreurs"); ax6.set_xlabel("Erreur absolue (m³/h)")
ax6.grid(True, alpha=0.3, axis="x"); ax6.invert_yaxis()

ax7 = fig.add_subplot(gs_layout[3, 1])
regime_color = np.where(actuals > SEUIL_CRUE, "#8e44ad",
               np.where(actuals > q75, "#e74c3c",
               np.where(actuals > q25, "#f39c12", "#2ecc71")))
ax7.scatter(actuals, diffs, c=regime_color, alpha=0.35, s=7)
cr_idx = df_eval["cr_signal"].values == 1
if cr_idx.sum() > 0:
    ax7.scatter(actuals[cr_idx], diffs[cr_idx], c="orange", s=30, zorder=5,
                alpha=0.8, edgecolors="k", lw=0.4, label="Crue retardée")
ax7.axhline(bias,   color="blue", lw=2,  ls="-")
ax7.axhline(loa_up, color="red",  lw=1.5, ls="--")
ax7.axhline(loa_dn, color="red",  lw=1.5, ls="--")
ax7.axhline(0, color="black", lw=0.8, ls=":", alpha=0.5)
ax7.text(0.98, 0.97, f"Biais = {bias:+.1f}", transform=ax7.transAxes,
         ha="right", va="top", fontsize=8, color="blue")
ax7.text(0.98, 0.91, f"+1.96σ = {loa_up:+.1f}", transform=ax7.transAxes,
         ha="right", va="top", fontsize=8, color="red")
ax7.text(0.98, 0.85, f"-1.96σ = {loa_dn:+.1f}", transform=ax7.transAxes,
         ha="right", va="top", fontsize=8, color="red")
ax7.text(0.02, 0.04, f"{pct_in:.1f}% dans LoA", transform=ax7.transAxes,
         fontsize=8, color="gray")
ax7.legend(handles=[mpatches.Patch(color=c, alpha=0.6, label=l) for c, l in
           [("#2ecc71",f"Faible (<{q25:.0f})"),("#f39c12","Moyen"),
            ("#e74c3c","Élevé"),(  "#8e44ad",f"Crue (>{SEUIL_CRUE:.0f})")]], fontsize=7)
ax7.set_xlabel("Débit réel (m³/h)", fontsize=10)
ax7.set_ylabel("Prédit − Réel (m³/h)", fontsize=10)
ax7.set_title(f"Bland-Altman (X = débit réel)\nBiais={bias:+.1f}  LoA=[{loa_dn:.1f}, {loa_up:.1f}]",
              fontsize=10)
ax7.grid(True, alpha=0.3)

plt.savefig(OUT / "horizon_12H_evaluation.png", dpi=150, bbox_inches="tight")
print(f"  -> {OUT / 'horizon_12H_evaluation.png'}")
print("\n" + "=" * 65)
print("FIN horizon_12H.py")
print("=" * 65)
