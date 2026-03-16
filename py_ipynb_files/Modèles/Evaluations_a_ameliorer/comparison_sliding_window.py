#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
comparison_sliding_window.py — Comparaison équitable tous modèles
==================================================================
Évalue XGBoost Direct (H=3/6/12/24h), XGBoost Récursif (référence)
et CNN 1D sur les MÊMES 4 configs de sliding window.

Pour chaque config et chaque fold : MAE, MAE_crues, R²
Sortie : moyenne ± écart-type par config, par modèle, par horizon.

4 configs (identiques au modèle récursif) :
  1. 8sem/2sem   (~11 folds)
  2. 10sem/2sem  (~10 folds)
  3. 12sem/2sem  (~9 folds)
  4. 12sem/3sem  (~6 folds)

CNN : le modèle pré-entraîné (H=3h, weights sauvegardés) est évalué
      sans ré-entraînement sur chaque fenêtre de validation.
      → mesure la robustesse temporelle de généralisation.

Référence récursif (depuis modele_recursif.py, 10sem/2sem) :
  H=3h : 46.6 m³/h | H=6h : 47.4 | H=12h : 47.5 | H=24h : 47.9
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from scipy.optimize import curve_fit; HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
try:
    import holidays as hol_lib; HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

# ══════════════════════════════════════════════════════════════
#  CHEMINS & CONSTANTES
# ══════════════════════════════════════════════════════════════
BASE             = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\bronze\bronze")
OUT              = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\Modèles")
CNN_WEIGHTS      = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\Modèles\CNN1d_DIRECT\cnn1d_H3_weights.pt")
POST_SHIFT_START = "2025-04-03"
W_TEST_FINAL     = 8 * 7 * 24
CNN_LOOKBACK     = 48

WINDOW_CONFIGS = [
    (8  * 7 * 24, 2 * 7 * 24, 2 * 7 * 24, "8sem/2sem"),
    (10 * 7 * 24, 2 * 7 * 24, 2 * 7 * 24, "10sem/2sem"),
    (12 * 7 * 24, 2 * 7 * 24, 2 * 7 * 24, "12sem/2sem"),
    (12 * 7 * 24, 3 * 7 * 24, 3 * 7 * 24, "12sem/3sem"),
]

# Best params déterminés par GridSearch dans chaque script standalone
BEST_PARAMS = {
    3:  {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "n_estimators": 300},
    12: {"max_depth": 6, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.7, "n_estimators": 300},
    24: {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.9, "n_estimators": 300},
    # H=6h : déterminé par GridSearch dans la section 2 ci-dessous
}

# ══════════════════════════════════════════════════════════════
#  1. CHARGEMENT
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("1. CHARGEMENT")
print("=" * 70)

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
print(f"  {len(df)} heures  |  {debit.index.min().date()} → {debit.index.max().date()}")

# ══════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING (toutes les features, tous horizons)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. FEATURE ENGINEERING")
print("=" * 70)

SEUIL_CRUE = float(df["debit"].quantile(0.90))
h_arr = df.index.hour.astype(float)

# ── Cyclic custom ──────────────────────────────────────────────
if HAS_SCIPY:
    profile_h = df.groupby(df.index.hour)["debit"].mean()
    def cyclic_func(h, beta, alpha, phi, alpha2):
        return beta + alpha * np.cos(h * 2*np.pi/24 + phi)**2 + alpha2 * np.sin(h * 2*np.pi/24 + phi)
    try:
        popt, _ = curve_fit(cyclic_func, profile_h.index.values.astype(float),
                            profile_h.values, p0=[profile_h.mean(), 50., -2.7, 80.], maxfev=20000)
        df["cyclic_custom"] = cyclic_func(h_arr, *popt)
        print(f"  Cyclic fit  R2={r2_score(profile_h.values, cyclic_func(profile_h.index.values.astype(float), *popt)):.3f}")
    except Exception:
        df["cyclic_custom"] = np.sin(2 * np.pi * h_arr / 24) * 100 + 200
else:
    df["cyclic_custom"] = np.sin(2 * np.pi * h_arr / 24) * 100 + 200

print(f"  Seuil crue P90 = {SEUIL_CRUE:.1f} m³/h")

# ── Features communes ─────────────────────────────────────────
df["slope_1h"]   = df["debit"] - df["debit"].shift(1)
df["slope_3h"]   = df["debit"] - df["debit"].shift(3)
df["slope_6h"]   = df["debit"] - df["debit"].shift(6)
df["slope_12h"]  = df["debit"] - df["debit"].shift(12)
df["slope_24h"]  = df["debit"] - df["debit"].shift(24)

for lag in [1, 3, 6, 12, 24, 48, 7*24]:
    df[f"debit_lag{lag}h" if lag < 100 else "debit_lag7d"] = df["debit"].shift(lag)
df["debit_lag1"]  = df["debit"].shift(1)
df["debit_lag3"]  = df["debit"].shift(3)
df["debit_lag6"]  = df["debit"].shift(6)
df["debit_lag12"] = df["debit"].shift(12)
df["debit_lag24"] = df["debit"].shift(24)
df["debit_lag48"] = df["debit"].shift(48)
df["debit_lag7d"] = df["debit"].shift(7 * 24)

for w in [3, 6, 12, 24, 72, 168]:
    df[f"rain_sum_{w}h"]  = df["rain"].rolling(w).sum()
df["rain_sum_3d"] = df["rain_sum_72h"]
df["rain_sum_7d"] = df["rain_sum_168h"]

for w in [3, 6, 12, 24]:
    df[f"rain_max_{w}h"] = df["rain"].rolling(w).max()

for lag in [6, 9, 12, 15, 18, 24, 30, 36, 48, 60, 72, 84, 96]:
    df[f"rain_lag_{lag}h"] = df["rain"].shift(lag)

for win in [6, 12, 24, 48, 72]:
    df[f"debit_min_{win}h"]  = df["debit"].rolling(win).min()
    df[f"debit_rise_{win}h"] = df["debit"] - df[f"debit_min_{win}h"]

df["rain_sum_3h"]  = df["rain"].rolling(3).sum()
df["rain_sum_6h"]  = df["rain"].rolling(6).sum()

df["hour_sin"]         = np.sin(2 * np.pi * h_arr / 24)
df["hour_cos"]         = np.cos(2 * np.pi * h_arr / 24)
df["ressuyage_exp"]    = df["rain"].ewm(halflife=48, adjust=False).mean()
rain_std               = df["rain"].std()
df["rain_exp_norm"]    = np.expm1(df["rain"] / rain_std) if rain_std > 0 else 0.
df["is_raining_hard"]  = (df["rain"] > 1.0).astype(float)
df["en_crue"]          = (df["debit"] > SEUIL_CRUE).astype(float)
df["intensite_crue"]   = np.maximum(df["debit"] - SEUIL_CRUE, 0.)
df["is_weekend"]       = (df.index.dayofweek >= 5).astype(float)
df["is_monday_morning"]= ((df.index.dayofweek == 0) & (df.index.hour < 10)).astype(float)
df["is_holiday"]       = 0.
if HAS_HOLIDAYS:
    fr_hol = hol_lib.France(years=range(2023, 2027))
    df["is_holiday"] = df.index.normalize().isin(fr_hol).astype(float)

# Pluie future oracle (toutes heures jusqu'à H=24h)
for i in range(1, 25):
    df[f"prev_pluie_{i}h"] = df["rain"].shift(-i)
df["prev_pluie_cumul_3h"]  = sum(df[f"prev_pluie_{i}h"] for i in range(1, 4))
df["prev_pluie_cumul_6h"]  = sum(df[f"prev_pluie_{i}h"] for i in range(1, 7))
df["prev_pluie_cumul_12h"] = sum(df[f"prev_pluie_{i}h"] for i in range(1, 13))
df["prev_pluie_cumul_24h"] = sum(df[f"prev_pluie_{i}h"] for i in range(1, 25))

# Interactions
df["rain_lag_24h_x_rise_6h"]  = df["rain_lag_24h"] * df["debit_rise_6h"]
df["rain_lag_12h_x_rise_6h"]  = df["rain_lag_12h"] * df["debit_rise_6h"]
df["rain_lag_12h_x_rise_12h"] = df["rain_lag_12h"] * df["debit_rise_12h"]
df["rain_lag_24h_x_rise_12h"] = df["rain_lag_24h"] * df["debit_rise_12h"]
df["rain_lag_48h_x_rise_24h"] = df["rain_lag_48h"] * df["debit_rise_24h"]

# Signaux crue retardée (adaptés par horizon)
df["crue_ret_H3"]  = ((df["debit_rise_6h"]  > 50)  & (df["rain_sum_6h"]  < 1.0)).astype(float)
df["crue_ret_H6"]  = ((df["debit_rise_12h"] > 80)  & (df["rain_sum_6h"]  < 1.0)).astype(float)
df["crue_ret_H12"] = ((df["debit_rise_24h"] > 120) & (df["rain_sum_12h"] < 2.0)).astype(float)

# Features jour cible H=24h
target_ts = df.index + pd.Timedelta(hours=24)
df["target_is_weekend"]        = (target_ts.dayofweek >= 5).astype(int)
df["target_is_monday_morning"] = ((target_ts.dayofweek == 0) & (target_ts.hour < 10)).astype(int)
df["target_is_holiday"]        = 0
if HAS_HOLIDAYS:
    df["target_is_holiday"]    = target_ts.normalize().isin(fr_hol).astype(int)

# ── Feature sets par horizon ───────────────────────────────────
FEATS = {
    3: [
        "debit","debit_lag1","debit_lag3","debit_lag6","debit_lag24","debit_lag48","debit_lag7d",
        "slope_1h","slope_3h",
        "rain_sum_3h","rain_sum_6h","rain_sum_24h","rain_sum_3d","rain_sum_7d","rain_max_3h",
        "prev_pluie_1h","prev_pluie_2h","prev_pluie_3h","prev_pluie_cumul_3h",
        "hour_sin","hour_cos","cyclic_custom",
        "ressuyage_exp","rain_exp_norm","is_raining_hard",
        "is_weekend","is_holiday","is_monday_morning",
        "en_crue","intensite_crue",
        # G12
        "rain_lag_6h","rain_lag_9h","rain_lag_12h","rain_lag_15h",
        "rain_lag_18h","rain_lag_24h","rain_lag_30h","rain_lag_36h","rain_lag_48h",
        "debit_min_6h","debit_min_12h","debit_min_24h",
        "debit_rise_6h","debit_rise_12h","debit_rise_24h",
        "rain_lag_24h_x_rise_6h","rain_lag_12h_x_rise_6h",
        "crue_ret_H3",
    ],
    6: [
        "debit","debit_lag1","debit_lag3","debit_lag6","debit_lag24","debit_lag48","debit_lag7d",
        "slope_1h","slope_3h","slope_6h","slope_12h",
        "rain_sum_3h","rain_sum_6h","rain_sum_24h","rain_sum_3d","rain_sum_7d",
        "rain_max_3h","rain_max_6h",
        "prev_pluie_1h","prev_pluie_2h","prev_pluie_3h",
        "prev_pluie_4h","prev_pluie_5h","prev_pluie_6h","prev_pluie_cumul_6h",
        "hour_sin","hour_cos","cyclic_custom",
        "ressuyage_exp","rain_exp_norm","is_raining_hard",
        "is_weekend","is_holiday","is_monday_morning",
        "en_crue","intensite_crue",
        # G12
        "rain_lag_6h","rain_lag_9h","rain_lag_12h","rain_lag_15h",
        "rain_lag_18h","rain_lag_24h","rain_lag_30h","rain_lag_36h",
        "rain_lag_48h","rain_lag_60h","rain_lag_72h",
        "debit_min_6h","debit_min_12h","debit_min_24h","debit_min_48h",
        "debit_rise_6h","debit_rise_12h","debit_rise_24h","debit_rise_48h",
        "rain_lag_24h_x_rise_6h","rain_lag_12h_x_rise_12h",
        "crue_ret_H6",
    ],
    12: [
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
        # G12
        "rain_lag_6h","rain_lag_9h","rain_lag_12h","rain_lag_15h",
        "rain_lag_18h","rain_lag_24h","rain_lag_30h","rain_lag_36h",
        "rain_lag_48h","rain_lag_60h","rain_lag_72h","rain_lag_84h","rain_lag_96h",
        "debit_min_6h","debit_min_12h","debit_min_24h","debit_min_48h","debit_min_72h",
        "debit_rise_6h","debit_rise_12h","debit_rise_24h","debit_rise_48h","debit_rise_72h",
        "rain_lag_24h_x_rise_12h","rain_lag_48h_x_rise_24h",
        "crue_ret_H12",
    ],
    24: [
        "debit","debit_lag1","debit_lag3","debit_lag6","debit_lag12",
        "debit_lag24","debit_lag48","debit_lag7d",
        "slope_1h","slope_3h","slope_6h","slope_12h","slope_24h",
        "rain_sum_3h","rain_sum_6h","rain_sum_12h","rain_sum_24h","rain_sum_3d","rain_sum_7d",
        "rain_max_3h","rain_max_6h","rain_max_12h","rain_max_24h",
        "prev_pluie_1h","prev_pluie_2h","prev_pluie_3h","prev_pluie_4h",
        "prev_pluie_5h","prev_pluie_6h","prev_pluie_7h","prev_pluie_8h",
        "prev_pluie_9h","prev_pluie_10h","prev_pluie_11h","prev_pluie_12h",
        "prev_pluie_13h","prev_pluie_14h","prev_pluie_15h","prev_pluie_16h",
        "prev_pluie_17h","prev_pluie_18h","prev_pluie_19h","prev_pluie_20h",
        "prev_pluie_21h","prev_pluie_22h","prev_pluie_23h","prev_pluie_24h",
        "prev_pluie_cumul_12h","prev_pluie_cumul_24h",
        "hour_sin","hour_cos","cyclic_custom",
        "is_weekend","is_holiday","is_monday_morning",
        "target_is_weekend","target_is_monday_morning","target_is_holiday",
        "ressuyage_exp","rain_exp_norm","is_raining_hard",
        "en_crue","intensite_crue",
    ],
}

# Targets
for h in [3, 6, 12, 24]:
    df[f"target_{h}h"] = df["debit"].shift(-h)

all_feats_needed = list(set(f for fl in FEATS.values() for f in fl))
all_targets = [f"target_{h}h" for h in [3, 6, 12, 24]]
df_clean = df.dropna(subset=all_feats_needed + all_targets).copy()

n = len(df_clean)
df_cv   = df_clean.iloc[: n - W_TEST_FINAL]
df_test = df_clean.iloc[n - W_TEST_FINAL :]

print(f"  CV   : {df_cv.index.min().date()} → {df_cv.index.max().date()}  ({len(df_cv)} h)")
print(f"  Test : {df_test.index.min().date()} → {df_test.index.max().date()}  ({len(df_test)} h)")

# ══════════════════════════════════════════════════════════════
#  3. GRIDSEARCH H=6h (paramètres non connus a priori)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. GRIDSEARCH H=6h (10sem train / 2sem val)")
print("=" * 70)

W_GS_TRAIN = 10 * 7 * 24
W_GS_VAL   =  2 * 7 * 24
gs_tr6 = df_cv.iloc[:W_GS_TRAIN].dropna(subset=FEATS[6] + ["target_6h"])
gs_va6 = df_cv.iloc[W_GS_TRAIN: W_GS_TRAIN + W_GS_VAL].dropna(subset=FEATS[6] + ["target_6h"])

PARAM_GRID = {
    "max_depth":        [4, 5, 6],
    "learning_rate":    [0.03, 0.05, 0.10],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
}

best_mae_gs6 = np.inf
best_p6 = None
best_n6 = 300
for md, lr, sub, col in product(
        PARAM_GRID["max_depth"], PARAM_GRID["learning_rate"],
        PARAM_GRID["subsample"], PARAM_GRID["colsample_bytree"]):
    m = xgb.XGBRegressor(
        max_depth=md, learning_rate=lr, n_estimators=800,
        subsample=sub, colsample_bytree=col,
        early_stopping_rounds=30, eval_metric="mae",
        random_state=42, n_jobs=-1, verbosity=0)
    m.fit(gs_tr6[FEATS[6]], gs_tr6["target_6h"],
          eval_set=[(gs_va6[FEATS[6]], gs_va6["target_6h"])], verbose=False)
    n_est = m.best_iteration + 1
    mae   = mean_absolute_error(gs_va6["target_6h"],
                                 m.predict(gs_va6[FEATS[6]]).clip(min=100))
    if mae < best_mae_gs6:
        best_mae_gs6 = mae
        best_p6      = {"max_depth": md, "learning_rate": lr,
                        "subsample": sub, "colsample_bytree": col}
        best_n6 = max(300, int(n_est * len(df_cv) / W_GS_TRAIN))

BEST_PARAMS[6] = {**best_p6, "n_estimators": best_n6}
print(f"  H=6h best : {best_p6}  n_est={best_n6}  MAE_val={best_mae_gs6:.2f}")

# ══════════════════════════════════════════════════════════════
#  4. SLIDING WINDOW CV — XGBoost DIRECT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. SLIDING WINDOW CV — XGBoost DIRECT (4 horizons × 4 configs)")
print("=" * 70)

def xgb_fold(df_tr, df_va, features, target_col, params):
    """Entraîne XGBoost sur un fold et retourne les métriques."""
    tr = df_tr.dropna(subset=features + [target_col])
    va = df_va.dropna(subset=features + [target_col])
    if len(tr) < 100 or len(va) < 10:
        return None
    m = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0, **params)
    m.fit(tr[features], tr[target_col], verbose=False)
    preds = m.predict(va[features]).clip(min=100)
    trues = va[target_col].values
    naive = va["debit"].clip(lower=100).values
    mae   = mean_absolute_error(trues, preds)
    rmse  = np.sqrt(mean_squared_error(trues, preds))
    r2    = r2_score(trues, preds)
    mae_n = mean_absolute_error(trues, naive)
    mask  = trues > SEUIL_CRUE
    mae_c = mean_absolute_error(trues[mask], preds[mask]) if mask.sum() > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2, "mae_naive": mae_n, "mae_crue": mae_c}

# Résultats par horizon et config
xgb_results = {h: {} for h in [3, 6, 12, 24]}

for h in [3, 6, 12, 24]:
    target_col = f"target_{h}h"
    features   = FEATS[h]
    params     = {k: v for k, v in BEST_PARAMS[h].items()}
    print(f"\n  ── H={h}h  ({len(features)} features) ─────────────────────────────")
    print(f"  {'Config':<12}  {'Folds':>5}  {'MAE moy':>8} {'MAE std':>8}  "
          f"{'R² moy':>7} {'R² std':>7}  {'MAEc moy':>9} {'MAEc std':>9}")
    print("  " + "─" * 74)

    for (w_tr, w_va, step, label) in WINDOW_CONFIGS:
        fold_results = []
        start = 0
        while start + w_tr + w_va <= len(df_cv):
            tr_part = df_cv.iloc[start: start + w_tr]
            va_part = df_cv.iloc[start + w_tr: start + w_tr + w_va]
            res = xgb_fold(tr_part, va_part, features, target_col, params)
            if res:
                fold_results.append(res)
            start += step

        if not fold_results:
            print(f"  {label:<12}  aucun fold valide")
            continue

        df_r = pd.DataFrame(fold_results)
        mae_m  = df_r["mae"].mean();      mae_s  = df_r["mae"].std()
        r2_m   = df_r["r2"].mean();       r2_s   = df_r["r2"].std()
        maec_m = df_r["mae_crue"].mean(); maec_s = df_r["mae_crue"].std()
        n_folds = len(df_r)

        xgb_results[h][label] = {
            "n_folds": n_folds,
            "mae_mean": mae_m, "mae_std": mae_s,
            "r2_mean": r2_m,   "r2_std": r2_s,
            "maec_mean": maec_m, "maec_std": maec_s,
            "fold_maes": df_r["mae"].values,
        }
        print(f"  {label:<12}  {n_folds:>5}  {mae_m:>8.2f} {mae_s:>8.2f}  "
              f"{r2_m:>7.3f} {r2_s:>7.3f}  {maec_m:>9.2f} {maec_s:>9.2f}")

# ══════════════════════════════════════════════════════════════
#  5. SLIDING WINDOW CV — CNN 1D (évaluation sans ré-entraînement)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. SLIDING WINDOW CV — CNN 1D H=3h (évaluation pré-entraîné)")
print("   Note : le modèle est FIXE (pas de ré-entraînement par fold)")
print("   → mesure la robustesse de généralisation temporelle")
print("=" * 70)

CNN_SEQ_FEATS = [
    "debit", "rain", "temperature_2m",
    "slope_1h", "slope_3h",
    "hour_sin", "hour_cos", "cyclic_custom",
    "ressuyage_exp", "en_crue", "intensite_crue",
    "is_weekend", "is_monday_morning", "is_holiday",
    "rain_sum_3h", "rain_sum_6h", "rain_sum_24h",
    "debit_min_6h", "debit_rise_6h",
    "rain_lag_6h", "rain_lag_12h", "rain_lag_24h",
    "crue_ret_H3",
]
CNN_STATIC_FEATS = [
    "debit_lag24", "debit_lag48", "debit_lag7d",
    "cyclic_custom",
    "rain_sum_3d", "rain_sum_7d",
    "rain_lag_9h", "rain_lag_15h", "rain_lag_18h",
    "rain_lag_30h", "rain_lag_36h", "rain_lag_48h",
    "rain_lag_24h_x_rise_6h", "rain_lag_12h_x_rise_6h",
    "debit_min_12h", "debit_min_24h",
    "debit_rise_12h", "debit_rise_24h",
    "rain_exp_norm", "is_raining_hard", "rain_max_3h",
]
CNN_ORACLE_FEATS = ["prev_pluie_1h", "prev_pluie_2h", "prev_pluie_3h"]

# Architecture CNN (identique à cnn1d_H3.py v2)
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=self.pad)
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.2):
        super().__init__()
        self.conv = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        return self.drop(self.relu(self.bn(self.conv(x)))) + self.skip(x)

class TCNHybrid(nn.Module):
    def __init__(self, n_seq, n_static, n_oracle,
                 n_filters=64, kernel=3, dilations=(1,2,4,8,16), dropout=0.2):
        super().__init__()
        layers, in_ch = [], n_seq
        for d in dilations:
            layers.append(TCNBlock(in_ch, n_filters, kernel, d, dropout))
            in_ch = n_filters
        self.tcn  = nn.Sequential(*layers)
        dense_in  = n_filters + n_static + n_oracle
        self.head = nn.Sequential(
            nn.Linear(dense_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )
    def forward(self, x_seq, x_sta, x_orc):
        x = self.tcn(x_seq.permute(0, 2, 1))[:, :, -1]
        return self.head(torch.cat([x, x_sta, x_orc], dim=1)).squeeze(1)

# Scaler entraîné sur les données d'entraînement du CNN (mêmes que à l'entraînement)
# On refit sur df_train (CV - 3 dernières semaines)
W_VAL_TRAIN = 3 * 7 * 24
df_cnn_train = df_cv.iloc[: len(df_cv) - W_VAL_TRAIN].dropna(
    subset=CNN_SEQ_FEATS + CNN_STATIC_FEATS + CNN_ORACLE_FEATS + ["target_3h"])

sc_seq    = StandardScaler().fit(df_cnn_train[CNN_SEQ_FEATS].values)
sc_static = StandardScaler().fit(df_cnn_train[CNN_STATIC_FEATS].values)
sc_oracle = StandardScaler().fit(df_cnn_train[CNN_ORACLE_FEATS].values)
sc_target = StandardScaler().fit(df_cnn_train[["target_3h"]].values)

cnn_model = None
if CNN_WEIGHTS.exists():
    cnn_model = TCNHybrid(len(CNN_SEQ_FEATS), len(CNN_STATIC_FEATS), len(CNN_ORACLE_FEATS))
    cnn_model.load_state_dict(torch.load(CNN_WEIGHTS, map_location="cpu"))
    cnn_model.eval()
    print(f"  Poids CNN chargés depuis {CNN_WEIGHTS.name}")
else:
    print(f"  ATTENTION : {CNN_WEIGHTS.name} non trouvé — CNN ignoré")

def cnn_eval_window(df_window):
    """Évalue le CNN pré-entraîné sur une fenêtre de données."""
    needed = CNN_SEQ_FEATS + CNN_STATIC_FEATS + CNN_ORACLE_FEATS + ["target_3h"]
    df_w = df_window.dropna(subset=needed)
    if len(df_w) < CNN_LOOKBACK + 10:
        return None
    seq = sc_seq.transform(df_w[CNN_SEQ_FEATS].values).astype(np.float32)
    sta = sc_static.transform(df_w[CNN_STATIC_FEATS].values).astype(np.float32)
    orc = sc_oracle.transform(df_w[CNN_ORACLE_FEATS].values).astype(np.float32)
    tgt = df_w["target_3h"].values

    preds_sc, n = [], len(df_w)
    with torch.no_grad():
        for i in range(0, n - CNN_LOOKBACK, 128):
            end = min(i + 128, n - CNN_LOOKBACK)
            xs = torch.from_numpy(np.stack([seq[j:j+CNN_LOOKBACK] for j in range(i, end)]))
            xst = torch.from_numpy(sta[i+CNN_LOOKBACK-1: end+CNN_LOOKBACK-1])
            xo  = torch.from_numpy(orc[i+CNN_LOOKBACK-1: end+CNN_LOOKBACK-1])
            preds_sc.append(cnn_model(xs, xst, xo).numpy())

    preds_sc = np.concatenate(preds_sc)
    preds = sc_target.inverse_transform(preds_sc.reshape(-1, 1)).ravel()
    trues = tgt[CNN_LOOKBACK - 1: CNN_LOOKBACK - 1 + len(preds)]
    naive = df_w["debit"].values[CNN_LOOKBACK - 1: CNN_LOOKBACK - 1 + len(preds)]

    mae  = mean_absolute_error(trues, preds)
    r2   = r2_score(trues, preds)
    mask = trues > SEUIL_CRUE
    mae_c = mean_absolute_error(trues[mask], preds[mask]) if mask.sum() > 0 else np.nan
    return {"mae": mae, "r2": r2, "mae_crue": mae_c}

cnn_results = {}
if cnn_model is not None:
    print(f"\n  {'Config':<12}  {'Folds':>5}  {'MAE moy':>8} {'MAE std':>8}  "
          f"{'R² moy':>7} {'R² std':>7}  {'MAEc moy':>9} {'MAEc std':>9}")
    print("  " + "─" * 74)

    for (w_tr, w_va, step, label) in WINDOW_CONFIGS:
        fold_results = []
        start = 0
        while start + w_tr + w_va <= len(df_cv):
            va_part = df_cv.iloc[start + w_tr: start + w_tr + w_va]
            res = cnn_eval_window(va_part)
            if res:
                fold_results.append(res)
            start += step

        if not fold_results:
            continue
        df_r   = pd.DataFrame(fold_results)
        mae_m  = df_r["mae"].mean();      mae_s  = df_r["mae"].std()
        r2_m   = df_r["r2"].mean();       r2_s   = df_r["r2"].std()
        maec_m = df_r["mae_crue"].mean(); maec_s = df_r["mae_crue"].std()
        cnn_results[label] = {
            "n_folds": len(df_r), "mae_mean": mae_m, "mae_std": mae_s,
            "r2_mean": r2_m, "r2_std": r2_s, "maec_mean": maec_m, "maec_std": maec_s,
        }
        print(f"  {label:<12}  {len(df_r):>5}  {mae_m:>8.2f} {mae_s:>8.2f}  "
              f"{r2_m:>7.3f} {r2_s:>7.3f}  {maec_m:>9.2f} {maec_s:>9.2f}")

# ══════════════════════════════════════════════════════════════
#  6. SYNTHÈSE FINALE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. SYNTHÈSE FINALE — MAE moy ± std par config")
print("   (XGBoost Direct tous horizons + CNN H=3h + Récursif référence)")
print("=" * 70)

ref_rec = {
    "8sem/2sem":  {3: (48.4, None), 6: (49.3, None), 12: (49.3, None), 24: (49.3, None)},
    "10sem/2sem": {3: (46.6, None), 6: (47.4, None), 12: (47.5, None), 24: (47.9, None)},
    "12sem/2sem": {3: (48.4, None), 6: (49.8, None), 12: (49.4, None), 24: (49.8, None)},
    "12sem/3sem": {3: (49.2, None), 6: (50.9, None), 12: (50.2, None), 24: (50.5, None)},
}

for label in ["8sem/2sem", "10sem/2sem", "12sem/2sem", "12sem/3sem"]:
    print(f"\n  ── Config {label} ──────────────────────────────────────────────────")
    print(f"  {'Modèle':<28}  {'MAE moy':>8} {'± std':>7}  {'R² moy':>7}  {'MAEc moy':>9}")
    print("  " + "─" * 65)
    for h in [3, 6, 12, 24]:
        r = xgb_results[h].get(label)
        if r:
            print(f"  {'XGBoost direct H=' + str(h) + 'h':<28}  {r['mae_mean']:>8.2f} {r['mae_std']:>7.2f}  "
                  f"{r['r2_mean']:>7.3f}  {r['maec_mean']:>9.2f}")
    if label in ref_rec and 3 in ref_rec[label]:
        mae_r3 = ref_rec[label][3][0]
        print(f"  {'Récursif H=3h (réf.)':<28}  {mae_r3:>8.1f} {'(std N/A)':>7}  {'—':>7}  {'—':>9}")
    if label in cnn_results:
        r = cnn_results[label]
        print(f"  {'CNN 1D v2 H=3h':<28}  {r['mae_mean']:>8.2f} {r['mae_std']:>7.2f}  "
              f"{r['r2_mean']:>7.3f}  {r['maec_mean']:>9.2f}")

# ══════════════════════════════════════════════════════════════
#  7. GRAPHIQUE RÉSUMÉ
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. GRAPHIQUE RÉSUMÉ")
print("=" * 70)

configs = ["8sem/2sem", "10sem/2sem", "12sem/2sem", "12sem/3sem"]
colors  = {"H=3h": "#2e86ab", "H=6h": "#a23b72", "H=12h": "#f18f01", "H=24h": "#c73e1d", "CNN": "#3b1f2b"}

fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=False)
fig.suptitle("Robustesse Sliding Window CV — MAE moyenne ± std par config\n"
             "XGBoost Direct (H=3/6/12/24h) + CNN 1D H=3h", fontsize=13, fontweight="bold")

for ax, label in zip(axes, configs):
    x_pos = 0
    for h, color in [(3, "#2e86ab"), (6, "#a23b72"), (12, "#f18f01"), (24, "#c73e1d")]:
        r = xgb_results[h].get(label)
        if r:
            ax.bar(x_pos, r["mae_mean"], color=color, alpha=0.85,
                   label=f"XGB H={h}h", width=0.7)
            ax.errorbar(x_pos, r["mae_mean"], yerr=r["mae_std"],
                        fmt="none", color="black", capsize=4, lw=1.5)
            x_pos += 1

    if label in cnn_results:
        r = cnn_results[label]
        ax.bar(x_pos, r["mae_mean"], color="#3b1f2b", alpha=0.85,
               label="CNN H=3h", width=0.7)
        ax.errorbar(x_pos, r["mae_mean"], yerr=r["mae_std"],
                    fmt="none", color="black", capsize=4, lw=1.5)

    ax.set_title(label, fontsize=11)
    ax.set_ylabel("MAE (m³/h)" if label == "8sem/2sem" else "")
    ax.set_xticks([])
    ax.set_ylim(0, max(70, ax.get_ylim()[1]))

# Légende commune
handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.85)
           for c in ["#2e86ab","#a23b72","#f18f01","#c73e1d","#3b1f2b"]]
labels_ = ["XGB H=3h","XGB H=6h","XGB H=12h","XGB H=24h","CNN H=3h"]
fig.legend(handles, labels_, loc="lower center", ncol=5, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.06, 1, 1])
out_path = OUT / "comparison_sliding_window.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  -> {out_path.name}")

print("\n" + "=" * 70)
print("FIN comparison_sliding_window.py")
print("=" * 70)
