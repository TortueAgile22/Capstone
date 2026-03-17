#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluation_finale.py — Comparaison équitable : Direct / Récursif / CNN
=======================================================================
Tout est recalculé dynamiquement — aucun résultat hardcodé.
Seuls les hyperparamètres XGBoost sont fixes (issus des GridSearch préalables).

Métriques calculées pour TOUS les modèles :
  MAE  |  R²  |  MAE crues (P90)  |  Gain vs naïf

Modèles comparés :
  - XGBoost Direct   H=3/6/12/24h  (1 modèle par horizon, features adaptées)
  - XGBoost Récursif H=3→24h par pas de 3h (1 modèle H=3h, appliqué récursivement)
  - CNN 1D v2        H=3h          (poids pré-entraînés)
  - Naïf (persistence)
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
from matplotlib.gridspec import GridSpec

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

import torch
import torch.nn as nn

try:
    from scipy.optimize import curve_fit; HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
try:
    import holidays as hol_lib; HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

# ══════════════════════════════════════════════════════════════
#  CHEMINS ET CONSTANTES
# ══════════════════════════════════════════════════════════════
BASE         = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\bronze\bronze")
OUT          = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\Modèles")
CNN_WEIGHTS  = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\Modèles\CNN1d_DIRECT\cnn1d_H3_weights.pt")
POST_SHIFT   = "2025-04-03"
W_TEST       = 8 * 7 * 24    # 8 semaines test set
W_CV_VAL     = 3 * 7 * 24    # val CNN (pour refitter les scalers)
CNN_LOOKBACK = 48
REC_STEP     = 3              # pas du modèle récursif
REC_STEPS    = 8              # jusqu'à H=24h (3×8=24)

# Hyperparamètres XGBoost — issus des GridSearch préalables par horizon
# (constantes de configuration, pas des résultats)
BEST_PARAMS = {
    3:  {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "n_estimators": 300},
    6:  {"max_depth": 6, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.7, "n_estimators": 300},
    12: {"max_depth": 6, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.7, "n_estimators": 300},
    24: {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.9, "n_estimators": 300},
}
# Modèle récursif = même config que le modèle H=3h
REC_PARAMS = dict(BEST_PARAMS[3])

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
debit = debit[debit.index >= POST_SHIFT]

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
#  2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. FEATURE ENGINEERING")
print("=" * 70)

SEUIL_CRUE = float(df["debit"].quantile(0.90))
h_arr = df.index.hour.astype(float)

# Cyclic custom
if HAS_SCIPY:
    profile_h = df.groupby(df.index.hour)["debit"].mean()
    def cyclic_func(h, beta, alpha, phi, alpha2):
        return beta + alpha * np.cos(h * 2*np.pi/24 + phi)**2 + alpha2 * np.sin(h * 2*np.pi/24 + phi)
    try:
        popt, _ = curve_fit(cyclic_func, profile_h.index.values.astype(float),
                            profile_h.values, p0=[profile_h.mean(), 50., -2.7, 80.], maxfev=20000)
        df["cyclic_custom"] = cyclic_func(h_arr, *popt)
        r2_fit = r2_score(profile_h.values, cyclic_func(profile_h.index.values.astype(float), *popt))
        print(f"  Cyclic fit  R2={r2_fit:.3f}")
    except Exception:
        df["cyclic_custom"] = np.sin(2 * np.pi * h_arr / 24) * 100 + 200
else:
    df["cyclic_custom"] = np.sin(2 * np.pi * h_arr / 24) * 100 + 200

# Lags débit
for lag in [1, 3, 6, 12, 24, 48]:
    df[f"debit_lag{lag}"] = df["debit"].shift(lag)
df["debit_lag7d"] = df["debit"].shift(7 * 24)

# Pentes
for w in [1, 3, 6, 12, 24]:
    df[f"slope_{w}h"] = df["debit"] - df["debit"].shift(w)

# Pluie passée (cumuls, max)
for w in [3, 6, 12, 24]:
    df[f"rain_sum_{w}h"] = df["rain"].rolling(w).sum()
    df[f"rain_max_{w}h"] = df["rain"].rolling(w).max()
df["rain_sum_3d"] = df["rain"].rolling(72).sum()
df["rain_sum_7d"] = df["rain"].rolling(168).sum()

# Pluie future oracle (1h → 24h)
for i in range(1, 25):
    df[f"prev_pluie_{i}h"] = df["rain"].shift(-i)
for n in [3, 6, 9, 12, 24]:
    df[f"prev_pluie_cumul_{n}h"] = sum(df[f"prev_pluie_{i}h"] for i in range(1, n + 1))

# Lags pluie passée
for lag in [6, 9, 12, 15, 18, 24, 30, 36, 48, 60, 72, 84, 96]:
    df[f"rain_lag_{lag}h"] = df["rain"].shift(lag)

# Débit min / rise (fenêtres multiples)
for win in [6, 12, 24, 48, 72]:
    df[f"debit_min_{win}h"]  = df["debit"].rolling(win).min()
    df[f"debit_rise_{win}h"] = df["debit"] - df[f"debit_min_{win}h"]

# Calendrier
df["hour_sin"]          = np.sin(2 * np.pi * h_arr / 24)
df["hour_cos"]          = np.cos(2 * np.pi * h_arr / 24)
df["ressuyage_exp"]     = df["rain"].ewm(halflife=48, adjust=False).mean()
rain_std                = df["rain"].std()
df["rain_exp_norm"]     = np.expm1(df["rain"] / rain_std) if rain_std > 0 else 0.
df["is_raining_hard"]   = (df["rain"] > 1.0).astype(float)
df["is_weekend"]        = (df.index.dayofweek >= 5).astype(float)
df["is_monday_morning"] = ((df.index.dayofweek == 0) & (df.index.hour < 10)).astype(float)
df["is_holiday"]        = 0.
fr_hol = None
if HAS_HOLIDAYS:
    fr_hol = hol_lib.France(years=range(2023, 2027))
    df["is_holiday"] = df.index.normalize().isin(fr_hol).astype(float)
df["en_crue"]        = (df["debit"] > SEUIL_CRUE).astype(float)
df["intensite_crue"] = np.maximum(df["debit"] - SEUIL_CRUE, 0.)

# Features jour cible (H=24h)
target_ts_24 = df.index + pd.Timedelta(hours=24)
df["target_is_weekend"]        = (target_ts_24.dayofweek >= 5).astype(int)
df["target_is_monday_morning"] = ((target_ts_24.dayofweek == 0) & (target_ts_24.hour < 10)).astype(int)
df["target_is_holiday"]        = 0
if HAS_HOLIDAYS and fr_hol is not None:
    df["target_is_holiday"] = target_ts_24.normalize().isin(fr_hol).astype(int)

# Interactions et signaux crue retardée
df["rain_lag_24h_x_rise_6h"]   = df["rain_lag_24h"] * df["debit_rise_6h"]
df["rain_lag_12h_x_rise_6h"]   = df["rain_lag_12h"] * df["debit_rise_6h"]
df["rain_lag_12h_x_rise_12h"]  = df["rain_lag_12h"] * df["debit_rise_12h"]
df["rain_lag_48h_x_rise_24h"]  = df["rain_lag_48h"] * df["debit_rise_24h"]
df["crue_retardee_signal_3h"]  = ((df["debit_rise_6h"]  > 50)  & (df["rain_sum_6h"]  < 1.0)).astype(float)
df["crue_retardee_signal_6h"]  = ((df["debit_rise_12h"] > 80)  & (df["rain_sum_6h"]  < 1.0)).astype(float)
df["crue_retardee_signal_12h"] = ((df["debit_rise_24h"] > 120) & (df["rain_sum_12h"] < 2.0)).astype(float)

# Targets pour tous les horizons récursifs
HORIZONS_REC = [REC_STEP * s for s in range(1, REC_STEPS + 1)]  # [3,6,9,12,15,18,21,24]
HORIZONS_DIR = [3, 6, 12, 24]
for h in HORIZONS_REC:
    df[f"target_{h}h"] = df["debit"].shift(-h)

print(f"  Seuil crue P90 = {SEUIL_CRUE:.1f} m³/h")

# ══════════════════════════════════════════════════════════════
#  3. DÉFINITION DES FEATURE SETS
# ══════════════════════════════════════════════════════════════

def dedup(lst):
    return list(dict.fromkeys(lst))

# Base commune G6 (H=3h)
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
G6_EXTRA_12H = dedup(G6_EXTRA_6H + [
    "slope_24h","rain_max_24h",
    "prev_pluie_7h","prev_pluie_8h","prev_pluie_9h","prev_pluie_10h",
    "prev_pluie_11h","prev_pluie_12h","prev_pluie_cumul_12h",
])
G12_12H = G12_6H + [
    "rain_lag_84h","rain_lag_96h",
    "debit_min_72h","debit_rise_72h",
    "rain_lag_48h_x_rise_24h","crue_retardee_signal_12h",
]
G6_EXTRA_24H = [
    "debit_lag12","slope_6h","slope_12h","slope_24h",
    "rain_sum_12h","rain_sum_24h","rain_max_6h","rain_max_12h","rain_max_24h",
] + [f"prev_pluie_{i}h" for i in range(4, 25)] + [
    "prev_pluie_cumul_12h","prev_pluie_cumul_24h",
    "target_is_weekend","target_is_monday_morning","target_is_holiday",
]

FEAT_DIRECT = {
    3:  dedup(G6_BASE + G12_3H),
    6:  dedup(G6_BASE + G6_EXTRA_6H + G12_6H),
    12: dedup(G6_BASE + G6_EXTRA_12H + G12_12H),
    24: dedup(G6_BASE + G6_EXTRA_24H),
}
FEAT_REC = FEAT_DIRECT[3]  # modèle récursif = même base que H=3h

# Toutes les features nécessaires (pour dropna global)
all_feats_needed = set(FEAT_REC)
for feats in FEAT_DIRECT.values():
    all_feats_needed.update(feats)
all_feats_needed.update({f"target_{h}h" for h in HORIZONS_REC})

df_clean = df.dropna(subset=list(all_feats_needed)).copy()
n = len(df_clean)
df_cv   = df_clean.iloc[: n - W_TEST]
df_test = df_clean.iloc[n - W_TEST:]
print(f"  CV   : {df_cv.index.min().date()} → {df_cv.index.max().date()}  ({len(df_cv)} h)")
print(f"  Test : {df_test.index.min().date()} → {df_test.index.max().date()}  ({len(df_test)} h)")


def compute_metrics(trues, preds, naive, seuil):
    """Retourne (MAE, R², MAE_crues, gain_vs_naif)."""
    mae   = mean_absolute_error(trues, preds)
    r2    = r2_score(trues, preds)
    mae_n = mean_absolute_error(trues, naive)
    mask  = trues > seuil
    mae_c = mean_absolute_error(trues[mask], preds[mask]) if mask.sum() > 0 else np.nan
    gain  = (1 - mae / mae_n) * 100
    return mae, r2, mae_c, gain


# ══════════════════════════════════════════════════════════════
#  4. XGBoost DIRECT — TEST SET
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. XGBoost DIRECT — TEST SET (Nov-Jan)")
print("=" * 70)
print(f"  {'H':>5}  {'MAE':>7}  {'R²':>6}  {'MAEc':>7}  {'Gain':>6}  {'n_crues':>7}")
print("  " + "─" * 50)

xgb_test = {}
for h in HORIZONS_DIR:
    feats  = FEAT_DIRECT[h]
    target = f"target_{h}h"
    cv_f   = df_cv.dropna(subset=feats + [target])
    te_f   = df_test.dropna(subset=feats + [target])
    m = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0, **BEST_PARAMS[h])
    m.fit(cv_f[feats], cv_f[target], verbose=False)
    preds = m.predict(te_f[feats]).clip(min=100)
    trues = te_f[target].values
    naive = te_f["debit"].clip(lower=100).values
    mae, r2, mae_c, gain = compute_metrics(trues, preds, naive, SEUIL_CRUE)
    n_crues = (trues > SEUIL_CRUE).sum()
    xgb_test[h] = {"mae": mae, "r2": r2, "mae_crue": mae_c, "gain": gain,
                   "preds": preds, "trues": trues, "naive": naive,
                   "idx": te_f.index}
    print(f"  H={h:>2}h  {mae:>7.2f}  {r2:>6.3f}  {mae_c:>7.1f}  {gain:>5.1f}%  {n_crues:>7}")


# ══════════════════════════════════════════════════════════════
#  5. XGBoost RÉCURSIF — TEST SET
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. XGBoost RÉCURSIF — TEST SET (Nov-Jan)")
print("=" * 70)

def recursive_predict(model, df_data, idx_eval, features, seuil_crue,
                      step_h=3, n_steps=8):
    """
    Prédiction récursive multi-horizon.
    Pour chaque t dans idx_eval, prédit debit(t+3h), ..., debit(t+24h).
    Les features débit sont mises à jour avec les prédictions précédentes.
    Les features pluie (oracle) viennent des vraies valeurs.
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
                prev = step_preds[s - 1]
                row["debit"]    = prev
                row["slope_1h"] = prev - df_data.loc[tau, "debit_lag1"]
                row["slope_3h"] = (prev - step_preds[s-2]) if s >= 2 else (prev - df_data.loc[tau, "debit_lag3"])
                if s >= 2: row["debit_lag3"] = step_preds[s-2]
                if s >= 3: row["debit_lag6"] = step_preds[s-3]
                row["en_crue"]        = 1.0 if prev > seuil_crue else 0.0
                row["intensite_crue"] = max(prev - seuil_crue, 0.0)
                dmin6  = df_data.loc[tau, "debit_min_6h"]
                dmin12 = df_data.loc[tau, "debit_min_12h"]
                dmin24 = df_data.loc[tau, "debit_min_24h"]
                rise6  = max(prev - dmin6,  0.)
                rise12 = max(prev - dmin12, 0.)
                rise24 = max(prev - dmin24, 0.)
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

# Entraîner sur CV complet
model_rec = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0, **REC_PARAMS)
model_rec.fit(df_cv[FEAT_REC], df_cv["target_3h"], verbose=False)

print("  Prédiction récursive sur le test set...")
preds_rec_te = recursive_predict(model_rec, df_test, df_test.index,
                                  FEAT_REC, SEUIL_CRUE, REC_STEP, REC_STEPS)

rec_test = {}
print(f"\n  {'H':>5}  {'MAE':>7}  {'R²':>6}  {'MAEc':>7}  {'Gain':>6}  {'n_crues':>7}")
print("  " + "─" * 50)
for hi, h in enumerate(HORIZONS_REC):
    target = f"target_{h}h"
    trues  = df_test[target].values
    preds  = preds_rec_te[:, hi]
    naive  = df_test["debit"].clip(lower=100).values
    valid  = ~np.isnan(preds)
    if valid.sum() < 10:
        continue
    mae, r2, mae_c, gain = compute_metrics(trues[valid], preds[valid], naive[valid], SEUIL_CRUE)
    n_crues = (trues[valid] > SEUIL_CRUE).sum()
    rec_test[h] = {"mae": mae, "r2": r2, "mae_crue": mae_c, "gain": gain,
                   "preds": preds, "trues": trues, "naive": naive,
                   "valid": valid}
    print(f"  H={h:>2}h  {mae:>7.2f}  {r2:>6.3f}  {mae_c:>7.1f}  {gain:>5.1f}%  {n_crues:>7}")


# ══════════════════════════════════════════════════════════════
#  6. CNN 1D v2 — TEST SET (H=3h)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. CNN 1D v2 — TEST SET (H=3h)")
print("=" * 70)

CNN_SEQ = ["debit","rain","temperature_2m","slope_1h","slope_3h",
           "hour_sin","hour_cos","cyclic_custom","ressuyage_exp",
           "en_crue","intensite_crue","is_weekend","is_monday_morning","is_holiday",
           "rain_sum_3h","rain_sum_6h","rain_sum_24h","debit_min_6h","debit_rise_6h",
           "rain_lag_6h","rain_lag_12h","rain_lag_24h","crue_retardee_signal_3h"]
CNN_STA = ["debit_lag24","debit_lag48","debit_lag7d","cyclic_custom",
           "rain_sum_3d","rain_sum_7d","rain_lag_9h","rain_lag_15h","rain_lag_18h",
           "rain_lag_30h","rain_lag_36h","rain_lag_48h",
           "rain_lag_24h_x_rise_6h","rain_lag_12h_x_rise_6h",
           "debit_min_12h","debit_min_24h","debit_rise_12h","debit_rise_24h",
           "rain_exp_norm","is_raining_hard","rain_max_3h"]
CNN_ORC = ["prev_pluie_1h","prev_pluie_2h","prev_pluie_3h"]

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
    def __init__(self, n_seq, n_sta, n_orc, n_filters=64, kernel=3,
                 dilations=(1,2,4,8,16), dropout=0.2):
        super().__init__()
        layers, in_ch = [], n_seq
        for d in dilations:
            layers.append(TCNBlock(in_ch, n_filters, kernel, d, dropout))
            in_ch = n_filters
        self.tcn  = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(n_filters + n_sta + n_orc, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )
    def forward(self, x_seq, x_sta, x_orc):
        x = self.tcn(x_seq.permute(0, 2, 1))[:, :, -1]
        return self.head(torch.cat([x, x_sta, x_orc], dim=1)).squeeze(1)

cnn_test = None
if CNN_WEIGHTS.exists():
    df_cnn_tr = df_cv.iloc[: len(df_cv) - W_CV_VAL].dropna(
        subset=CNN_SEQ + CNN_STA + CNN_ORC + ["target_3h"])
    sc_seq = StandardScaler().fit(df_cnn_tr[CNN_SEQ].values)
    sc_sta = StandardScaler().fit(df_cnn_tr[CNN_STA].values)
    sc_orc = StandardScaler().fit(df_cnn_tr[CNN_ORC].values)
    sc_tgt = StandardScaler().fit(df_cnn_tr[["target_3h"]].values)

    cnn_model = TCNHybrid(len(CNN_SEQ), len(CNN_STA), len(CNN_ORC))
    cnn_model.load_state_dict(torch.load(CNN_WEIGHTS, map_location="cpu"))
    cnn_model.eval()

    def cnn_predict_on(df_part):
        needed = CNN_SEQ + CNN_STA + CNN_ORC + ["target_3h"]
        dp = df_part.dropna(subset=needed)
        if len(dp) < CNN_LOOKBACK + 5:
            return None
        seq = sc_seq.transform(dp[CNN_SEQ].values).astype(np.float32)
        sta = sc_sta.transform(dp[CNN_STA].values).astype(np.float32)
        orc = sc_orc.transform(dp[CNN_ORC].values).astype(np.float32)
        ps  = []
        with torch.no_grad():
            for i in range(0, len(dp) - CNN_LOOKBACK, 256):
                end = min(i + 256, len(dp) - CNN_LOOKBACK)
                xs  = torch.from_numpy(np.stack([seq[j:j+CNN_LOOKBACK] for j in range(i, end)]))
                xst = torch.from_numpy(sta[i+CNN_LOOKBACK-1: end+CNN_LOOKBACK-1])
                xo  = torch.from_numpy(orc[i+CNN_LOOKBACK-1: end+CNN_LOOKBACK-1])
                ps.append(cnn_model(xs, xst, xo).numpy())
        ps    = np.concatenate(ps)
        preds = sc_tgt.inverse_transform(ps.reshape(-1, 1)).ravel()
        offset = CNN_LOOKBACK - 1
        trues = dp["target_3h"].values[offset: offset + len(preds)]
        naive = dp["debit"].values[offset: offset + len(preds)]
        idx   = dp.index[offset: offset + len(preds)]
        return preds, trues, naive, idx

    result = cnn_predict_on(df_test)
    if result is not None:
        preds_c, trues_c, naive_c, idx_c = result
        mae, r2, mae_c, gain = compute_metrics(trues_c, preds_c, naive_c, SEUIL_CRUE)
        n_crues = (trues_c > SEUIL_CRUE).sum()
        cnn_test = {"mae": mae, "r2": r2, "mae_crue": mae_c, "gain": gain,
                    "preds": preds_c, "trues": trues_c, "naive": naive_c, "idx": idx_c}
        print(f"  MAE={mae:.2f}  R²={r2:.3f}  MAEc={mae_c:.1f}  "
              f"gain={gain:.1f}%  n_crues={n_crues}")
    else:
        print("  Données insuffisantes.")
else:
    print("  Poids CNN non trouvés — CNN ignoré.")


# ══════════════════════════════════════════════════════════════
#  7. ROBUSTESSE — Sliding Window 10sem/2sem (tous horizons directs)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. ROBUSTESSE TEMPORELLE — Sliding Window 10sem/2sem")
print("   XGBoost Direct H=3/6/12/24h  (~10 folds)")
print("=" * 70)

W_TR, W_VA, STEP_CV = 10*7*24, 2*7*24, 2*7*24

# Collecter les fold MAEs pour chaque horizon direct
cv_folds = {h: [] for h in HORIZONS_DIR}   # {h: [mae_fold1, mae_fold2, ...]}
fold_dates = []

start = 0
while start + W_TR + W_VA <= len(df_cv):
    tr_raw = df_cv.iloc[start: start + W_TR]
    va_raw = df_cv.iloc[start + W_TR: start + W_TR + W_VA]
    fold_dates_needed = start == 0  # n'enregistrer les dates qu'une seule fois
    for h in HORIZONS_DIR:
        target = f"target_{h}h"
        feats  = FEAT_DIRECT[h]
        tr = tr_raw.dropna(subset=feats + [target])
        va = va_raw.dropna(subset=feats + [target])
        if len(tr) < 100 or len(va) < 10:
            continue
        m = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0, **BEST_PARAMS[h])
        m.fit(tr[feats], tr[target], verbose=False)
        preds_f = m.predict(va[feats]).clip(min=100)
        trues_f = va[target].values
        cv_folds[h].append(mean_absolute_error(trues_f, preds_f))
        if h == 3:
            fold_dates.append(va.index[0].strftime("%Y-%m"))
    start += STEP_CV

# Récursif H=3h = même que Direct H=3h au premier pas
cv_folds_rec = {3: list(cv_folds[3])}

cv_stats = {}
for h in HORIZONS_DIR:
    arr = np.array(cv_folds[h])
    cv_stats[h] = {"mean": arr.mean(), "std": arr.std(),
                   "min": arr.min(), "max": arr.max(), "folds": arr}

fold_maes = cv_stats[3]["folds"]   # garde pour le boxplot

print(f"\n  {'H':>5}  {'Folds':>6}  {'MAE moy':>8}  {'± std':>7}  {'min':>6}  {'max':>6}")
print("  " + "─" * 50)
for h in HORIZONS_DIR:
    s = cv_stats[h]
    print(f"  H={h:>2}h  {len(s['folds']):>6}  {s['mean']:>8.2f}  {s['std']:>7.2f}  "
          f"{s['min']:>6.2f}  {s['max']:>6.2f}")

print(f"\n  MAE par fold — H=3h (val début) :")
for date, mae_f in zip(fold_dates, fold_maes):
    bar = "█" * int(mae_f / 3)
    print(f"    {date}  {mae_f:>6.1f} m³/h  {bar}")


# ══════════════════════════════════════════════════════════════
#  8. TABLEAU RÉCAPITULATIF FINAL
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. TABLEAU RÉCAPITULATIF — TEST SET (Nov-Jan)")
print("=" * 70)

mae_naive_3 = xgb_test[3]["naive"]
mae_naive_3_val = mean_absolute_error(xgb_test[3]["trues"], mae_naive_3)

print(f"\n  {'Modèle':<32}  {'MAE test':>8}  {'R²':>6}  {'MAEc':>7}  {'MAE CV':>8}  {'± std':>7}")
print("  " + "─" * 78)
print(f"  {'Naïf H=3h (persistence)':<32}  {mae_naive_3_val:>8.2f}  {'—':>6}  {'—':>7}  {'—':>8}  {'—':>7}")

print(f"\n  ── XGBoost Direct ──────────────────────────────────────────────────────")
for h in HORIZONS_DIR:
    r  = xgb_test[h]
    cv = cv_stats.get(h, {})
    cv_str  = f"{cv['mean']:>8.2f}" if cv else f"{'—':>8}"
    std_str = f"{cv['std']:>7.2f}"  if cv else f"{'—':>7}"
    print(f"  {'XGBoost Direct H=' + str(h) + 'h':<32}  {r['mae']:>8.2f}  {r['r2']:>6.3f}  "
          f"{r['mae_crue']:>7.1f}  {cv_str}  {std_str}")

print(f"\n  ── XGBoost Récursif (MAE CV = idem Direct H=3h pour premier pas) ──────")
for h in HORIZONS_REC:
    if h not in rec_test: continue
    r = rec_test[h]
    # CV sliding window disponible seulement pour H=3h (= même que direct H=3h)
    if h == 3 and 3 in cv_stats:
        cv_str  = f"{cv_stats[3]['mean']:>8.2f}"
        std_str = f"{cv_stats[3]['std']:>7.2f}"
    else:
        cv_str, std_str = f"{'—':>8}", f"{'—':>7}"
    print(f"  {'XGBoost Récursif H=' + str(h) + 'h':<32}  {r['mae']:>8.2f}  {r['r2']:>6.3f}  "
          f"{r['mae_crue']:>7.1f}  {cv_str}  {std_str}")

if cnn_test:
    print(f"\n  ── CNN 1D v2 (pas de ré-entraînement par fold) ────────────────────────")
    r = cnn_test
    print(f"  {'CNN 1D v2 H=3h':<32}  {r['mae']:>8.2f}  {r['r2']:>6.3f}  "
          f"{r['mae_crue']:>7.1f}  {'—':>8}  {'—':>7}")


# ══════════════════════════════════════════════════════════════
#  9. GRAPHIQUES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("9. GÉNÉRATION DES GRAPHIQUES")
print("=" * 70)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Évaluation finale — Station P\n"
             "XGBoost Direct / Récursif / CNN 1D", fontsize=14, fontweight="bold")
gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30,
              top=0.91, bottom=0.08, left=0.08, right=0.97)

# ── (A) MAE par horizon ───────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
H_rec_plot  = sorted(rec_test.keys())
mae_rec_v   = [rec_test[h]["mae"] for h in H_rec_plot]
naive_rec_v = [mean_absolute_error(rec_test[h]["trues"][rec_test[h]["valid"]],
                                    rec_test[h]["naive"][rec_test[h]["valid"]])
               for h in H_rec_plot]
mae_dir_v   = [xgb_test[h]["mae"] for h in HORIZONS_DIR]

ax_a.plot(H_rec_plot, naive_rec_v, "k--", lw=1.5, marker="s", ms=4, label="Naïf (persistence)")
ax_a.plot(H_rec_plot, mae_rec_v, "o-", color="#e07b39", lw=2, ms=6, label="XGBoost Récursif")
ax_a.plot(HORIZONS_DIR, mae_dir_v, "^-", color="#2e86ab", lw=2, ms=8, label="XGBoost Direct")
if cnn_test:
    ax_a.scatter([3], [cnn_test["mae"]], color="#c73e1d", s=120, zorder=5,
                 marker="D", label=f"CNN 1D H=3h ({cnn_test['mae']:.1f})")
ax_a.set_title("(A) MAE par horizon — Test set", fontsize=11)
ax_a.set_xlabel("Horizon (heures)")
ax_a.set_ylabel("MAE (m³/h)")
ax_a.set_xticks(HORIZONS_REC)
ax_a.legend(fontsize=9)
ax_a.grid(alpha=0.3)

# ── (B) R² par horizon ───────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
r2_rec_v = [rec_test[h]["r2"] for h in H_rec_plot]
r2_dir_v = [xgb_test[h]["r2"] for h in HORIZONS_DIR]
ax_b.plot(H_rec_plot, r2_rec_v, "o-", color="#e07b39", lw=2, ms=6, label="XGBoost Récursif")
ax_b.plot(HORIZONS_DIR, r2_dir_v, "^-", color="#2e86ab", lw=2, ms=8, label="XGBoost Direct")
if cnn_test:
    ax_b.scatter([3], [cnn_test["r2"]], color="#c73e1d", s=120, zorder=5,
                 marker="D", label=f"CNN 1D H=3h ({cnn_test['r2']:.3f})")
ax_b.axhline(0, color="black", lw=0.8, ls="--")
ax_b.set_title("(B) R² par horizon — Test set", fontsize=11)
ax_b.set_xlabel("Horizon (heures)")
ax_b.set_ylabel("R²")
ax_b.set_xticks(HORIZONS_REC)
ax_b.legend(fontsize=9)
ax_b.grid(alpha=0.3)

# ── (C) Boxplot robustesse — distribution des folds ──────────
ax_c = fig.add_subplot(gs[1, 0])
bp = ax_c.boxplot(fold_maes, vert=True, patch_artist=True, widths=0.5,
                  medianprops=dict(color="black", lw=2.5),
                  flierprops=dict(marker="o", ms=6, alpha=0.6))
bp["boxes"][0].set_facecolor("#2e86ab99")
x_jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(fold_maes))
ax_c.scatter(1 + x_jitter, fold_maes, color="#2e86ab", s=50, zorder=5, alpha=0.8)
ax_c.axhline(xgb_test[3]["mae"], color="tomato", lw=2, ls="--",
             label=f"MAE test set = {xgb_test[3]['mae']:.1f} m³/h")
ax_c.set_title(f"(C) Robustesse XGBoost Direct H=3h\n"
               f"Sliding Window 10sem/2sem — {len(fold_maes)} folds\n"
               f"moy={fold_maes.mean():.1f}  std={fold_maes.std():.1f}  "
               f"[{fold_maes.min():.1f}–{fold_maes.max():.1f}] m³/h", fontsize=10)
ax_c.set_ylabel("MAE (m³/h)")
ax_c.set_xticks([])
ax_c.legend(fontsize=9)
ax_c.set_ylim(0, max(fold_maes.max(), xgb_test[3]["mae"]) * 1.25)

# ── (D) Séries temporelles test — H=3h ───────────────────────
ax_d = fig.add_subplot(gs[1, 1])
idx_aligned = df_test.dropna(subset=FEAT_DIRECT[3] + ["target_3h"]).index
trues_3 = xgb_test[3]["trues"]
preds_3 = xgb_test[3]["preds"]
rec_valid_3 = rec_test.get(3, {}).get("valid", np.zeros(len(df_test), dtype=bool))
preds_rec_3 = rec_test.get(3, {}).get("preds", np.full(len(df_test), np.nan))
ax_d.plot(idx_aligned, trues_3, color="steelblue", lw=1.2, label="Réel", alpha=0.9)
ax_d.plot(idx_aligned, preds_3, color="#e07b39", lw=1.0, alpha=0.8, label="XGB Direct H=3h")
if 3 in rec_test:
    ax_d.plot(df_test.index[rec_valid_3], preds_rec_3[rec_valid_3],
              color="#2ca02c", lw=0.9, alpha=0.7, ls="--", label="XGB Récursif H=3h")
if cnn_test and len(cnn_test["preds"]) > 0:
    ax_d.plot(cnn_test["idx"], cnn_test["preds"],
              color="#c73e1d", lw=0.8, alpha=0.7, label="CNN 1D H=3h")
ax_d.axhline(SEUIL_CRUE, color="orange", lw=1, ls="--",
             label=f"Seuil crue P90 ({SEUIL_CRUE:.0f} m³/h)")
ax_d.set_title("(D) Séries temporelles — Test set H=3h", fontsize=11)
ax_d.set_ylabel("Débit (m³/h)")
ax_d.legend(fontsize=8)
ax_d.grid(alpha=0.3)

out_path = OUT / "evaluation_finale.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  -> {out_path.name}")

print("\n" + "=" * 70)
print("FIN evaluation_finale.py")
print("=" * 70)
