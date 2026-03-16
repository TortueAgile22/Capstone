#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
modele_recursif.py — XGBoost prédiction récursive multi-horizon  |  Station P
==============================================================================
Un seul modèle entraîné à prédire t+3h est utilisé de façon récursive :
  t → prédire t+3 → (utilise cette pred) → prédire t+6 → ... → t+24h

══ COMPARAISON ÉQUITABLE ══════════════════════════════════════════════════════

  Oracle pluie — les deux approches ont le MÊME accès :
  ┌─────────────────────────────────────────────────────────────────┐
  │ Récursif : au pas s, voit rain(τ+1h..τ+3h) avec τ = t+3s      │
  │   → au total voit rain(t+1h) ... rain(t+24h) par fenêtres de 3h│
  │ Direct H=Xh : voit prev_pluie_1h ... prev_pluie_Xh en une fois │
  │   → exactement le même contenu d'information oracle             │
  └─────────────────────────────────────────────────────────────────┘

  Features directes — adaptées à chaque horizon (comme dans horizon_Xh.py) :
  ┌───────────┬──────────────────────────────────────────────────────┐
  │ H=3h      │ G6(30) + G12(18)  = 48 features (= horizon_3H.py)   │
  │ H=6h      │ G6_6h(36) + G12_6h(22) = 58 features                │
  │ H=9h      │ G6_6h + prev_pluie_7-9h étendu  ≈ 61 features       │
  │ H=12h     │ G6_12h(47) + G12_12h(26) = 73 features              │
  │ H=15/18h  │ G6_12h + prev_pluie jusqu'à l'horizon ≈ 76-79 feat  │
  │ H=21h     │ G6_12h + prev_pluie jusqu'à 21h   ≈ 82 features     │
  │ H=24h     │ G6_24h(63) — G12 non retenu (ablation)              │
  └───────────┴──────────────────────────────────────────────────────┘

Questions de la tutrice :
  1. "Le modèle dérive-t-il ?" → MAE à chaque horizon rec vs direct équitable
  2. "Robustesse en changeant la sliding window et l'horizon"
     → heatmap CV : 4 fenêtres × 8 horizons
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

BASE             = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\bronze\bronze")
OUT              = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\Modèles\XGBoost_Recursif")
POST_SHIFT_START = "2025-04-03"
STEP             = 3
MAX_STEPS        = 8          # jusqu'à H=24h
W_TEST_FINAL     = 8 * 7 * 24

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
    d = d.set_index("ts"); dfs_w.append(d[["rain", "temperature_2m"]])
df_weather = pd.concat(dfs_w).groupby(level=0)[["rain", "temperature_2m"]].mean()
df_weather = df_weather[~df_weather.index.duplicated(keep="first")]
df = pd.DataFrame({"debit": debit}).join(df_weather, how="inner")
df["rain"] = df["rain"].fillna(0)
df["temperature_2m"] = df["temperature_2m"].ffill().fillna(15)
print(f"  Débit  : {debit.index.min().date()} -> {debit.index.max().date()}")
print(f"  Météo  : {len(weather_files)} stations | Fusionné : {len(df)} heures")

# ══════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING — TOUTES LES FEATURES (tous horizons)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. FEATURE ENGINEERING  (features pour tous les horizons)")
print("=" * 70)

h_arr = df.index.hour.astype(float)
SEUIL_CRUE = float(df["debit"].quantile(0.90))

# ── Lags de débit ────────────────────────────────────────────
for lag in [1, 3, 6, 12, 24, 48]:
    df[f"debit_lag{lag}"] = df["debit"].shift(lag)
df["debit_lag7d"] = df["debit"].shift(7 * 24)

# ── Pentes ───────────────────────────────────────────────────
for w in [1, 3, 6, 12, 24]:
    df[f"slope_{w}h"] = df["debit"] - df["debit"].shift(w)

# ── Cumuls et max pluie passée ────────────────────────────────
for w in [3, 6, 12, 24]:
    df[f"rain_sum_{w}h"] = df["rain"].rolling(w).sum()
    df[f"rain_max_{w}h"] = df["rain"].rolling(w).max()
df["rain_sum_3d"] = df["rain"].rolling(72).sum()
df["rain_sum_7d"] = df["rain"].rolling(168).sum()

# ── Pluie future oracle : 1h → 24h ───────────────────────────
for i in range(1, 25):
    df[f"prev_pluie_{i}h"] = df["rain"].shift(-i)
df["prev_pluie_cumul_3h"]  = sum(df[f"prev_pluie_{i}h"] for i in range(1, 4))
df["prev_pluie_cumul_6h"]  = sum(df[f"prev_pluie_{i}h"] for i in range(1, 7))
df["prev_pluie_cumul_9h"]  = sum(df[f"prev_pluie_{i}h"] for i in range(1, 10))
df["prev_pluie_cumul_12h"] = sum(df[f"prev_pluie_{i}h"] for i in range(1, 13))
df["prev_pluie_cumul_24h"] = sum(df[f"prev_pluie_{i}h"] for i in range(1, 25))

# ── Cyclic + calendrier ───────────────────────────────────────
df["hour_sin"]        = np.sin(2 * np.pi * h_arr / 24)
df["hour_cos"]        = np.cos(2 * np.pi * h_arr / 24)
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

CYCLIC_OK = False
if HAS_SCIPY:
    profile_h = df.groupby(df.index.hour)["debit"].mean()
    def cyclic_func(h, beta, alpha, phi, alpha2):
        return beta + alpha * np.cos(h*2*np.pi/24+phi)**2 + alpha2*np.sin(h*2*np.pi/24+phi)
    try:
        popt, _ = curve_fit(cyclic_func, profile_h.index.values.astype(float),
                            profile_h.values, p0=[profile_h.mean(), 50., -2.7, 80.], maxfev=20000)
        df["cyclic_custom"] = cyclic_func(h_arr, *popt)
        r2_fit = r2_score(profile_h.values,
                          cyclic_func(profile_h.index.values.astype(float), *popt))
        print(f"  Cyclic fit  R2={r2_fit:.3f}  beta={popt[0]:.1f}  alpha={popt[1]:.1f}")
        CYCLIC_OK = True
    except Exception:
        pass
if not CYCLIC_OK:
    df["cyclic_custom"] = df["hour_sin"] * 100 + 200

# Features TARGET DAY (H=24h)
target_ts_24 = df.index + pd.Timedelta(hours=24)
df["target_is_weekend"]        = (target_ts_24.dayofweek >= 5).astype(int)
df["target_is_monday_morning"] = ((target_ts_24.dayofweek == 0) & (target_ts_24.hour < 10)).astype(int)
df["target_is_holiday"] = 0
if HAS_HOLIDAYS and fr_hol is not None:
    df["target_is_holiday"] = target_ts_24.normalize().isin(fr_hol).astype(int)

# ── G12 (H=3h base) ──────────────────────────────────────────
for lag in [6, 9, 12, 15, 18, 24, 30, 36, 48]:
    df[f"rain_lag_{lag}h"] = df["rain"].shift(lag)
for win in [6, 12, 24]:
    df[f"debit_min_{win}h"]  = df["debit"].rolling(win).min()
    df[f"debit_rise_{win}h"] = df["debit"] - df[f"debit_min_{win}h"]
df["rain_lag_24h_x_rise_6h"]  = df["rain_lag_24h"] * df["debit_rise_6h"]
df["rain_lag_12h_x_rise_6h"]  = df["rain_lag_12h"] * df["debit_rise_6h"]
df["crue_retardee_signal_3h"] = (
    (df["debit_rise_6h"] > 50) & (df["rain_sum_6h"] < 1.0)
).astype(float)

# ── G12 élargi (H=6h) ────────────────────────────────────────
for lag in [60, 72]:
    df[f"rain_lag_{lag}h"] = df["rain"].shift(lag)
df["debit_min_48h"]  = df["debit"].rolling(48).min()
df["debit_rise_48h"] = df["debit"] - df["debit_min_48h"]
df["rain_lag_12h_x_rise_12h"] = df["rain_lag_12h"] * df["debit_rise_12h"]
df["crue_retardee_signal_6h"] = (
    (df["debit_rise_12h"] > 80) & (df["rain_sum_6h"] < 1.0)
).astype(float)

# ── G12 élargi (H=12h) ───────────────────────────────────────
for lag in [84, 96]:
    df[f"rain_lag_{lag}h"] = df["rain"].shift(lag)
df["debit_min_72h"]  = df["debit"].rolling(72).min()
df["debit_rise_72h"] = df["debit"] - df["debit_min_72h"]
df["rain_lag_48h_x_rise_24h"] = df["rain_lag_48h"] * df["debit_rise_24h"]
df["crue_retardee_signal_12h"] = (
    (df["debit_rise_24h"] > 120) & (df["rain_sum_12h"] < 2.0)
).astype(float)

# ── Targets ───────────────────────────────────────────────────
HORIZONS_H = [STEP * s for s in range(1, MAX_STEPS + 1)]
for h in HORIZONS_H:
    df[f"target_{h}h"] = df["debit"].shift(-h)

print(f"  Seuil crue P90 = {SEUIL_CRUE:.1f} m³/h")

# ══════════════════════════════════════════════════════════════
#  DÉFINITION DES FEATURES PAR HORIZON (pour les modèles directs)
# ══════════════════════════════════════════════════════════════

# Base commune G6 H=3h (30 features)
G6_BASE = [
    "debit","debit_lag1","debit_lag3","debit_lag6","debit_lag24","debit_lag48","debit_lag7d",
    "slope_1h","slope_3h",
    "rain_sum_3h","rain_sum_6h","rain_sum_24h","rain_sum_3d","rain_sum_7d","rain_max_3h",
    "prev_pluie_1h","prev_pluie_2h","prev_pluie_3h","prev_pluie_cumul_3h",
    "hour_sin","hour_cos","cyclic_custom",
    "ressuyage_exp","rain_exp_norm","is_raining_hard",
    "is_weekend","is_holiday","is_monday_morning",
    "en_crue","intensite_crue",
]
G12_3H = [
    "rain_lag_6h","rain_lag_9h","rain_lag_12h","rain_lag_15h",
    "rain_lag_18h","rain_lag_24h","rain_lag_30h","rain_lag_36h","rain_lag_48h",
    "debit_min_6h","debit_min_12h","debit_min_24h",
    "debit_rise_6h","debit_rise_12h","debit_rise_24h",
    "rain_lag_24h_x_rise_6h","rain_lag_12h_x_rise_6h","crue_retardee_signal_3h",
]
# G6 étendu H=6h (ajoute slope_6h/12h, rain_max_6h/12h, prev_pluie_4-6h, debit_lag12)
G6_EXTRA_6H = [
    "debit_lag12","slope_6h","slope_12h",
    "rain_sum_12h","rain_max_6h","rain_max_12h",
    "prev_pluie_4h","prev_pluie_5h","prev_pluie_6h","prev_pluie_cumul_6h",
]
G12_6H = G12_3H + [
    "rain_lag_60h","rain_lag_72h",
    "debit_min_48h","debit_rise_48h",
    "rain_lag_12h_x_rise_12h","crue_retardee_signal_6h",
]
# G6 étendu H=12h
G6_EXTRA_12H = G6_EXTRA_6H + [
    "slope_24h","rain_sum_12h","rain_max_24h",
    "prev_pluie_7h","prev_pluie_8h","prev_pluie_9h","prev_pluie_10h",
    "prev_pluie_11h","prev_pluie_12h","prev_pluie_cumul_12h",
]
G12_12H = G12_6H + [
    "rain_lag_84h","rain_lag_96h",
    "debit_min_72h","debit_rise_72h",
    "rain_lag_48h_x_rise_24h","crue_retardee_signal_12h",
]
# G6 pour H=24h (pas de G12, résultat ablation)
G6_EXTRA_24H = [
    "debit_lag12","slope_6h","slope_12h","slope_24h",
    "rain_sum_12h","rain_sum_24h","rain_max_6h","rain_max_12h","rain_max_24h",
] + [f"prev_pluie_{i}h" for i in range(4, 25)] + [
    "prev_pluie_cumul_12h","prev_pluie_cumul_24h",
    "target_is_weekend","target_is_monday_morning","target_is_holiday",
]

def dedup(lst):
    return list(dict.fromkeys(lst))

FEAT_3H  = dedup(G6_BASE + G12_3H)
FEAT_6H  = dedup(G6_BASE + G6_EXTRA_6H + G12_6H)
FEAT_9H  = dedup(G6_BASE + G6_EXTRA_6H + ["prev_pluie_7h","prev_pluie_8h","prev_pluie_9h",
                 "prev_pluie_cumul_9h"] + G12_6H)
FEAT_12H = dedup(G6_BASE + G6_EXTRA_12H + G12_12H)
FEAT_15H = dedup(G6_BASE + G6_EXTRA_12H + [f"prev_pluie_{i}h" for i in range(13, 16)] + G12_12H)
FEAT_18H = dedup(G6_BASE + G6_EXTRA_12H + [f"prev_pluie_{i}h" for i in range(13, 19)] + G12_12H)
FEAT_21H = dedup(G6_BASE + G6_EXTRA_12H + [f"prev_pluie_{i}h" for i in range(13, 22)] + G12_12H)
FEAT_24H = dedup(G6_BASE + G6_EXTRA_24H)

FEAT_DIRECT = {3: FEAT_3H, 6: FEAT_6H, 9: FEAT_9H, 12: FEAT_12H,
               15: FEAT_15H, 18: FEAT_18H, 21: FEAT_21H, 24: FEAT_24H}

# Features pour le modèle récursif (base H=3h)
FEAT_REC = FEAT_3H  # même features que horizon_3H.py

all_feat_needed = set(FEAT_REC)
for feats in FEAT_DIRECT.values():
    all_feat_needed.update(feats)
all_feat_needed.update({f"target_{h}h" for h in HORIZONS_H})

df_clean = df.dropna(subset=list(all_feat_needed)).copy()
n = len(df_clean)
df_cv   = df_clean.iloc[: n - W_TEST_FINAL]
df_test = df_clean.iloc[n - W_TEST_FINAL :]

for h, feats in FEAT_DIRECT.items():
    print(f"  H={h:>2}h direct : {len(feats)} features")
print(f"  Récursif base : {len(FEAT_REC)} features (H=3h, utilisées à chaque pas)")
print(f"\n  CV   : {df_cv.index.min().date()} -> {df_cv.index.max().date()}  ({len(df_cv)} h)")
print(f"  Test : {df_test.index.min().date()} -> {df_test.index.max().date()}  ({len(df_test)} h)")

# ══════════════════════════════════════════════════════════════
#  3. GRIDSEARCH  (modèle récursif de base H=3h)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. GRIDSEARCH  (modèle récursif de base H=3h)")
print("=" * 70)

W_GS_TRAIN = 10*7*24; W_GS_VAL = 2*7*24
gs_tr = df_cv.iloc[:W_GS_TRAIN].dropna(subset=FEAT_REC + ["target_3h"])
gs_va = df_cv.iloc[W_GS_TRAIN: W_GS_TRAIN+W_GS_VAL].dropna(subset=FEAT_REC + ["target_3h"])

PARAM_GRID = {"max_depth":[4,5,6], "learning_rate":[0.03,0.05,0.10],
              "subsample":[0.7,0.9], "colsample_bytree":[0.7,0.9]}
best_mae_gs = np.inf; best_params = None; best_n_est_gs = None

print(f"\n  {'max_d':>5} {'lr':>5} {'sub':>5} {'col':>5} | {'MAE_val':>8} {'n_est':>6}")
print("  " + "─" * 42)
for md, lr, sub, col in product(PARAM_GRID["max_depth"], PARAM_GRID["learning_rate"],
                                 PARAM_GRID["subsample"], PARAM_GRID["colsample_bytree"]):
    m = xgb.XGBRegressor(max_depth=md, learning_rate=lr, n_estimators=800,
                         subsample=sub, colsample_bytree=col, early_stopping_rounds=30,
                         eval_metric="mae", random_state=42, n_jobs=-1, verbosity=0)
    m.fit(gs_tr[FEAT_REC], gs_tr["target_3h"],
          eval_set=[(gs_va[FEAT_REC], gs_va["target_3h"])], verbose=False)
    ne = m.best_iteration + 1
    mae = mean_absolute_error(gs_va["target_3h"],
                              m.predict(gs_va[FEAT_REC]).clip(min=100))
    tag = " ← best" if mae < best_mae_gs else ""
    print(f"  {md:>5} {lr:>5.2f} {sub:>5.1f} {col:>5.1f} | {mae:>8.2f} {ne:>6}{tag}")
    if mae < best_mae_gs:
        best_mae_gs = mae; best_params = dict(max_depth=md, learning_rate=lr,
                                               subsample=sub, colsample_bytree=col)
        best_n_est_gs = ne

scale       = len(df_cv) / W_GS_TRAIN
best_n_est  = max(300, int(best_n_est_gs * scale))
print(f"\n  Best params : {best_params}")
print(f"  n_estimators (scaled ×{scale:.1f}) : {best_n_est}")

XGB_BEST = dict(**best_params, n_estimators=best_n_est,
                random_state=42, n_jobs=-1, verbosity=0)

# ══════════════════════════════════════════════════════════════
#  4. FONCTION DE PRÉDICTION RÉCURSIVE
# ══════════════════════════════════════════════════════════════
def recursive_predict(model, df_data, idx_eval, features, seuil_crue,
                      step_h=3, n_steps=8):
    """
    Prédiction récursive multi-horizon.

    Pour chaque point t dans idx_eval, prédit debit(t+3), ..., debit(t+3*n_steps).

    Mise à jour des features débit à chaque pas s (0-indexed) :
      - Les features PLUIE (prev_pluie_*, rain_sum_*, rain_lag_*) viennent
        automatiquement du bon pas τ = t+3s dans df_data (= valeurs réelles/oracle).
      - Les features DÉBIT sont mises à jour avec les prédictions :
        · debit           ← prédiction du pas précédent
        · slope_1h        ← pred_prev - real(τ-1h) [valeur réelle disponible]
        · slope_3h        ← pred_prev - pred_prev_prev (ou real si s<2)
        · debit_lag3      ← pred d'il y a 2 pas (si s≥2)
        · debit_lag6      ← pred d'il y a 3 pas (si s≥3)
        · en_crue, intensite_crue ← depuis pred_prev
        · debit_rise_*    ← pred_prev − debit_min_* réel (approx. conservatrice)
        · interactions    ← recalculées
    """
    all_preds = np.full((len(idx_eval), n_steps), np.nan)
    for i, t in enumerate(idx_eval):
        step_preds = []
        for s in range(n_steps):
            tau = t + pd.Timedelta(hours=step_h * s)
            if tau not in df_data.index:
                break
            row = df_data.loc[tau, features].copy().astype(float)
            if s >= 1:
                prev_pred = step_preds[s - 1]
                row["debit"]        = prev_pred
                row["slope_1h"]     = prev_pred - df_data.loc[tau, "debit_lag1"]
                if s >= 2:
                    row["slope_3h"]   = prev_pred - step_preds[s - 2]
                    row["debit_lag3"] = step_preds[s - 2]
                else:
                    row["slope_3h"] = prev_pred - df_data.loc[tau, "debit_lag3"]
                if s >= 3:
                    row["debit_lag6"] = step_preds[s - 3]
                row["en_crue"]        = 1.0 if prev_pred > seuil_crue else 0.0
                row["intensite_crue"] = max(prev_pred - seuil_crue, 0.0)
                dmin6  = df_data.loc[tau, "debit_min_6h"]
                dmin12 = df_data.loc[tau, "debit_min_12h"]
                dmin24 = df_data.loc[tau, "debit_min_24h"]
                rise6  = max(prev_pred - dmin6,  0.)
                rise12 = max(prev_pred - dmin12, 0.)
                rise24 = max(prev_pred - dmin24, 0.)
                row["debit_rise_6h"]           = rise6
                row["debit_rise_12h"]          = rise12
                row["debit_rise_24h"]          = rise24
                row["rain_lag_24h_x_rise_6h"]  = df_data.loc[tau, "rain_lag_24h"] * rise6
                row["rain_lag_12h_x_rise_6h"]  = df_data.loc[tau, "rain_lag_12h"] * rise6
                row["crue_retardee_signal_3h"]  = float((rise6 > 50) and
                                                        (df_data.loc[tau, "rain_sum_6h"] < 1.0))
            pred = float(model.predict(row[features].values.reshape(1, -1))[0])
            pred = max(pred, 100.)
            step_preds.append(pred)
        all_preds[i, :len(step_preds)] = step_preds
    return all_preds

# ══════════════════════════════════════════════════════════════
#  5. SLIDING WINDOW CV — ROBUSTESSE (récursif vs direct adapté)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. ROBUSTESSE SLIDING WINDOW CV")
print("   Comparaison ÉQUITABLE : direct avec features adaptées à chaque horizon")
print("=" * 70)

WINDOW_CONFIGS = [
    (8  * 7 * 24, 2 * 7 * 24, 2 * 7 * 24, "8sem/2sem"),
    (10 * 7 * 24, 2 * 7 * 24, 2 * 7 * 24, "10sem/2sem"),
    (12 * 7 * 24, 2 * 7 * 24, 2 * 7 * 24, "12sem/2sem"),
    (12 * 7 * 24, 3 * 7 * 24, 3 * 7 * 24, "12sem/3sem"),
]
wlabels = [wc[3] for wc in WINDOW_CONFIGS]

cv_rec  = {h: {wl: [] for wl in wlabels} for h in HORIZONS_H}
cv_dir  = {h: {wl: [] for wl in wlabels} for h in HORIZONS_H}
cv_naif = {h: {wl: [] for wl in wlabels} for h in HORIZONS_H}

print(f"\n  Fenêtres : {wlabels}")
print(f"  Horizons : {HORIZONS_H}h\n")

target_cols = [f"target_{h}h" for h in HORIZONS_H]
all_needed_cv = list(set(FEAT_REC) | {f for fs in FEAT_DIRECT.values() for f in fs}
                     | set(target_cols))
df_cv_clean = df_cv.dropna(subset=all_needed_cv)

for w_tr, w_va, step_w, wlabel in WINDOW_CONFIGS:
    folds = 0; start = 0
    while start + w_tr + w_va <= len(df_cv_clean):
        tr = df_cv_clean.iloc[start: start + w_tr]
        va = df_cv_clean.iloc[start + w_tr: start + w_tr + w_va]
        if len(tr) < 200 or len(va) < 20:
            start += step_w; continue
        # Modèle récursif (base H=3h)
        m_rec = xgb.XGBRegressor(**XGB_BEST)
        m_rec.fit(tr[FEAT_REC], tr["target_3h"], verbose=False)
        preds_rec = recursive_predict(m_rec, df_cv_clean, va.index,
                                      FEAT_REC, SEUIL_CRUE, STEP, MAX_STEPS)
        for hi, h in enumerate(HORIZONS_H):
            col_t = f"target_{h}h"
            act   = va[col_t].values
            p_rec = preds_rec[:, hi]
            valid = ~np.isnan(p_rec) & ~np.isnan(act)
            if valid.sum() < 5: continue
            cv_rec[h][wlabel].append(mean_absolute_error(act[valid], p_rec[valid]))
            # Modèle direct avec features adaptées
            feats_h = FEAT_DIRECT[h]
            m_dir = xgb.XGBRegressor(**XGB_BEST)
            m_dir.fit(tr[feats_h], tr[col_t], verbose=False)
            p_dir = m_dir.predict(va[feats_h]).clip(min=100)
            cv_dir[h][wlabel].append(mean_absolute_error(act, p_dir))
            cv_naif[h][wlabel].append(mean_absolute_error(act, va["debit"].clip(lower=100)))
        folds += 1; start += step_w
    print(f"  {wlabel} : {folds} folds")

print(f"\n  TABLEAU — MAErec / MAEdir_adapté / MAEnaïf")
print(f"\n  {'H':>5}", end="")
for wl in wlabels: print(f"  {wl:>26}", end="")
print()
print("  " + "─" * (5 + 28 * len(wlabels)))
for h in HORIZONS_H:
    print(f"  H={h:>2}h", end="")
    for wl in wlabels:
        r = cv_rec[h][wl]; d = cv_dir[h][wl]; n = cv_naif[h][wl]
        if r:
            print(f"  {np.mean(r):5.1f}/{np.mean(d):5.1f}/{np.mean(n):5.1f}     ", end="")
        else:
            print(f"  {'—':>26}", end="")
    print()

# ══════════════════════════════════════════════════════════════
#  6. MODELE FINAL — TEST SET
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. MODELE FINAL — TEST SET (Nov-Jan)")
print("   Comparaison équitable : direct avec features adaptées à chaque horizon")
print("=" * 70)

df_cv_f = df_cv.dropna(subset=all_needed_cv)
df_te_f = df_test.dropna(subset=all_needed_cv)

model_final = xgb.XGBRegressor(**XGB_BEST)
model_final.fit(df_cv_f[FEAT_REC], df_cv_f["target_3h"], verbose=False)

preds_rec_te = recursive_predict(model_final, df_te_f, df_te_f.index,
                                  FEAT_REC, SEUIL_CRUE, STEP, MAX_STEPS)

preds_dir_te = {}
for h in HORIZONS_H:
    feats_h = FEAT_DIRECT[h]
    m_d = xgb.XGBRegressor(**XGB_BEST)
    m_d.fit(df_cv_f[feats_h], df_cv_f[f"target_{h}h"], verbose=False)
    preds_dir_te[h] = m_d.predict(df_te_f[feats_h]).clip(min=100)

print(f"\n  {'H':>5} | {'MAE Récursif':>13} | {'MAE Direct*':>11} | "
      f"{'MAE Naïf':>9} | {'Gain rec':>9} | {'Dérive (rec−dir)':>16}")
print(f"  {'':>5}   {'':>13}   {'(*feat. adapt.)':>11}")
print("  " + "─" * 82)

mae_rec_list = []; mae_dir_list = []; mae_naif_list = []
for hi, h in enumerate(HORIZONS_H):
    act_h  = df_te_f[f"target_{h}h"].values
    p_rec  = preds_rec_te[:, hi]
    p_dir  = preds_dir_te[h]
    p_naif = df_te_f["debit"].clip(lower=100).values
    valid  = ~np.isnan(p_rec)
    mae_r  = mean_absolute_error(act_h[valid], p_rec[valid])
    mae_d  = mean_absolute_error(act_h, p_dir)
    mae_n  = mean_absolute_error(act_h, p_naif)
    gain   = (1 - mae_r / mae_n) * 100
    derive = mae_r - mae_d
    mae_rec_list.append(mae_r); mae_dir_list.append(mae_d); mae_naif_list.append(mae_n)
    arrow  = "↓ dérive" if derive > 2 else ("↑ récursif+" if derive < -2 else "≈ équivalent")
    print(f"  H={h:>2}h | {mae_r:>12.2f}  | {mae_d:>10.2f}  | {mae_n:>8.2f}  | "
          f"{gain:>8.1f}%  | {derive:>+8.2f}  {arrow}")

print(f"\n  MAE sur les crues (débit > {SEUIL_CRUE:.0f} m³/h) :")
print(f"  {'H':>5} | {'MAE Récursif':>13} | {'MAE Direct*':>11} | n_pics")
print("  " + "─" * 50)
for hi, h in enumerate(HORIZONS_H):
    act_h = df_te_f[f"target_{h}h"].values
    p_rec = preds_rec_te[:, hi]
    valid = ~np.isnan(p_rec)
    mask  = (act_h > SEUIL_CRUE) & valid
    if mask.sum() > 0:
        mae_r = mean_absolute_error(act_h[mask], p_rec[mask])
        mae_d = mean_absolute_error(act_h[mask], preds_dir_te[h][mask])
        print(f"  H={h:>2}h | {mae_r:>12.2f}  | {mae_d:>10.2f}  | {mask.sum()}")

# ══════════════════════════════════════════════════════════════
#  7. GRAPHIQUES
# ══════════════════════════════════════════════════════════════
print(f"\nGénération des graphiques...")

# ── Fig 1 : MAE vs horizon + dérive ──────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle("Modèle récursif vs Direct (features adaptées par horizon)",
              fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(HORIZONS_H, mae_rec_list,  "o-",  color="#e74c3c", lw=2, ms=7,
        label="Récursif (1 modèle H=3h)")
ax.plot(HORIZONS_H, mae_dir_list,  "s--", color="#2980b9", lw=2, ms=6,
        label="Direct (features adaptées)")
ax.plot(HORIZONS_H, mae_naif_list, "^:",  color="#7f8c8d", lw=1.5, ms=5,
        label="Naïf (persistence)")
ax.fill_between(HORIZONS_H, mae_dir_list, mae_rec_list,
                alpha=0.15, color="#e74c3c",
                label="Coût récursif (+)" if any(r>d for r,d in zip(mae_rec_list,mae_dir_list))
                      else "Avantage récursif (-)")
ax.set_xlabel("Horizon (heures)"); ax.set_ylabel("MAE (m³/h)")
ax.set_title("MAE vs Horizon  (test set)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xticks(HORIZONS_H)

ax = axes[1]
derive = [r - d for r, d in zip(mae_rec_list, mae_dir_list)]
colors_b = ["#e74c3c" if x > 2 else ("#27ae60" if x < -2 else "#f39c12") for x in derive]
bars = ax.bar(HORIZONS_H, derive, color=colors_b, alpha=0.8, width=2)
ax.axhline(0, color="black", lw=1)
ax.axhline(+5, color="#e74c3c", lw=0.8, ls="--", alpha=0.5, label="±5 m³/h seuil")
ax.axhline(-5, color="#27ae60", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Horizon (heures)"); ax.set_ylabel("MAErécursif − MAEdirect (m³/h)")
ax.set_title("Dérive : rouge=direct meilleur, vert=récursif meilleur")
ax.grid(True, alpha=0.3, axis="y"); ax.set_xticks(HORIZONS_H); ax.legend(fontsize=8)

fig1.tight_layout()
fig1.savefig(OUT / "recursif_derive_MAE.png", dpi=130, bbox_inches="tight")
plt.close(fig1)
print(f"  -> recursif_derive_MAE.png")

# ── Fig 2 : Séries temporelles ────────────────────────────────
fig2, axes2 = plt.subplots(4, 1, figsize=(18, 16), sharex=True)
fig2.suptitle("Prédictions récursives vs directes — Extrait test (Nov-Jan)",
              fontsize=13, fontweight="bold")
start_ex = pd.Timestamp("2025-11-20"); end_ex = pd.Timestamp("2026-01-05")
mask_ex = (df_te_f.index >= start_ex) & (df_te_f.index <= end_ex)
idx_ex  = df_te_f.index[mask_ex]

for ax_i, (h, col) in enumerate(zip([3, 6, 12, 24],
                                     ["#2ecc71","#3498db","#e67e22","#e74c3c"])):
    hi  = HORIZONS_H.index(h)
    ax  = axes2[ax_i]
    act = df_te_f.loc[idx_ex, f"target_{h}h"].values
    p_r = preds_rec_te[mask_ex, hi]
    p_d = preds_dir_te[h][mask_ex]
    ax.plot(idx_ex, act, color="steelblue", lw=1.3, label="Réel", alpha=0.9)
    ax.plot(idx_ex, p_r, color=col,    lw=1.0, label=f"Récursif H={h}h", alpha=0.85)
    ax.plot(idx_ex, p_d, color="gray", lw=0.8, ls="--", label=f"Direct H={h}h (feat. adapt.)", alpha=0.65)
    ax.axhline(SEUIL_CRUE, color="red", lw=0.7, ls=":", alpha=0.5)
    vld = ~np.isnan(p_r)
    mae_r = mean_absolute_error(act[vld], p_r[vld])
    mae_d = mean_absolute_error(act, p_d)
    ax.set_title(f"H={h}h  —  MAE Récursif={mae_r:.1f}  vs  MAE Direct(adapt.)={mae_d:.1f}  m³/h",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Débit (m³/h)"); ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
axes2[-1].set_xlabel("Date")
fig2.tight_layout()
fig2.savefig(OUT / "recursif_series_temporelles.png", dpi=130, bbox_inches="tight")
plt.close(fig2)
print(f"  -> recursif_series_temporelles.png")

# ── Fig 3 : Heatmap robustesse CV ────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle("Robustesse Sliding Window CV — Récursif vs Direct (features adaptées)",
              fontsize=13, fontweight="bold")
for ax_idx, (results, title) in enumerate([
        (cv_rec, "Récursif (MAE CV)"),
        (cv_dir, "Direct — features adaptées (MAE CV)")]):
    mat = np.array([[np.mean(results[h][wl]) if results[h][wl] else np.nan
                     for wl in wlabels] for h in HORIZONS_H])
    ax = axes3[ax_idx]
    vmin = np.nanmin(mat) - 2; vmax = np.nanmax(mat) + 2
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="MAE (m³/h)")
    ax.set_xticks(range(len(wlabels))); ax.set_xticklabels(wlabels, rotation=20, ha="right")
    ax.set_yticks(range(len(HORIZONS_H))); ax.set_yticklabels([f"H={h}h" for h in HORIZONS_H])
    ax.set_title(title, fontweight="bold")
    for i in range(len(HORIZONS_H)):
        for j in range(len(wlabels)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", fontsize=8)
fig3.tight_layout()
fig3.savefig(OUT / "recursif_robustesse_CV.png", dpi=130, bbox_inches="tight")
plt.close(fig3)
print(f"  -> recursif_robustesse_CV.png")

# ── Fig 4 : Distribution erreurs ─────────────────────────────
fig4, axes4 = plt.subplots(2, 4, figsize=(20, 9))
fig4.suptitle("Distribution erreurs absolues — Récursif (rouge) vs Direct adapté (bleu)",
              fontsize=13, fontweight="bold")
for hi, h in enumerate(HORIZONS_H):
    ax   = axes4[hi // 4][hi % 4]
    act  = df_te_f[f"target_{h}h"].values
    p_r  = preds_rec_te[:, hi]; valid = ~np.isnan(p_r)
    err_r = np.abs(act[valid] - p_r[valid])
    err_d = np.abs(act - preds_dir_te[h])
    lim   = max(np.percentile(err_r, 97), np.percentile(err_d, 97))
    bins  = np.linspace(0, lim, 40)
    ax.hist(err_r, bins=bins, alpha=0.55, color="#e74c3c",
            label=f"Récursif  MAE={np.mean(err_r):.0f}")
    ax.hist(err_d, bins=bins, alpha=0.55, color="#2980b9",
            label=f"Direct    MAE={np.mean(err_d):.0f}")
    ax.axvline(np.median(err_r), color="#e74c3c", lw=1.5, ls="--")
    ax.axvline(np.median(err_d), color="#2980b9", lw=1.5, ls="--")
    ax.set_title(f"H={h}h", fontweight="bold")
    ax.set_xlabel("|Erreur| (m³/h)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig(OUT / "recursif_distribution_erreurs.png", dpi=130, bbox_inches="tight")
plt.close(fig4)
print(f"  -> recursif_distribution_erreurs.png")

print("\n" + "=" * 70)
print("FIN modele_recursif.py")
print("=" * 70)
