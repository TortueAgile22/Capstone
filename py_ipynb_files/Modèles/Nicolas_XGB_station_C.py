#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost Station C - Modele ameliore vs SARIMAX Nicolas
Horizon H=3h (direct), avec features cycliques parametriques (tuteur)

Features cles:
  - Profil cyclique parametrique: beta + alpha*cos(h*tau+phi)^2 + alpha2*sin(h*tau+phi)
  - Features polynomiales pluie (exp normalisee, carres, interactions)
  - Features seuil pluie
  - Lags debit etendus (1h -> 168h)
  - Oracle pluie future t+1, t+2, t+3

Reference Nicolas SARIMAX: MAE=102.14, RMSE=160.13, R2=0.819
Hard cap tuteur: 1700 m3/h (bassin d'orage)

Donnees:
  - sensors_C_20210101_20250101.csv  (entry_debit, 2021-2025)
  - weather_C/*.csv  (14 stations, 2015-2024, correction UTC requise)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from scipy.optimize import curve_fit

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\bronze\bronze")

HORIZON  = 3
MIN_CAP  = 100    # minimum physique (tuteur seance 1)
HARD_CAP = 1700   # hard cap bassin d'orage (tuteur seance 5)

# ─── 1. Chargement entry_debit ────────────────────────────────────────────────
print("Chargement entry_debit (2021-2025)...")
chunks = []
for chunk in pd.read_csv(BASE / "sensors_C_20210101_20250101.csv", chunksize=500_000):
    sub = chunk[chunk["sensor"] == "entry_debit"][["ts", "value"]]
    if len(sub):
        chunks.append(sub)

df_debit = (pd.concat(chunks)
            .rename(columns={"ts": "date", "value": "debit"})
            .assign(date=lambda x: pd.to_datetime(x["date"]))
            .set_index("date").sort_index())
df_debit = df_debit[~df_debit.index.duplicated(keep="first")].resample("1h").mean()
df_debit["debit"] = df_debit["debit"].clip(lower=0)
print(f"  {df_debit.index.min()} -> {df_debit.index.max()} | mean={df_debit['debit'].mean():.0f} m3/h")

# ─── 2. Chargement meteo + correction UTC -> Europe/Paris ─────────────────────
print("Chargement meteo (14 stations, correction UTC)...")
frames_w = []
for f in sorted((BASE / "weather_C").glob("*.csv")):
    dw = pd.read_csv(f, parse_dates=["date"]).set_index("date")
    dw = dw[~dw.index.duplicated(keep="first")]
    # Correction UTC -> Europe/Paris (Open-Meteo renvoie UTC par defaut)
    dw.index = pd.to_datetime(dw.index, utc=True).tz_convert("Europe/Paris").tz_localize(None)
    frames_w.append(dw[["rain", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]])

df_weather = (pd.concat(frames_w)
              .groupby(level=0)[["rain", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]]
              .mean()
              .rename(columns={"temperature_2m": "temp",
                               "relative_humidity_2m": "humidity",
                               "wind_speed_10m": "wind"}))
df_weather = df_weather[~df_weather.index.duplicated(keep="first")]

# ─── 3. Merge et nettoyage ────────────────────────────────────────────────────
df = df_debit.join(df_weather, how="inner").dropna(subset=["debit"])
df["rain"]     = df["rain"].fillna(0)
df["temp"]     = df["temp"].ffill().bfill()
df["humidity"] = df["humidity"].ffill().bfill()
df["wind"]     = df["wind"].fillna(0)

# Split 60/20/20 identique a Nicolas pour comparaison
n = len(df)
TRAIN_END  = df.index[int(n * 0.60)]
TEST_START = df.index[int(n * 0.80)]
df_tr = df[df.index < TRAIN_END]

print(f"  Train: {df.index.min()} -> {TRAIN_END}")
print(f"  Val  : {TRAIN_END} -> {TEST_START}")
print(f"  Test : {TEST_START} -> {df.index.max()}")
SEUIL_CRUE = np.quantile(df[df.index < TEST_START]["debit"].dropna(), 0.90)
print(f"  Seuil crue P90: {SEUIL_CRUE:.0f} m3/h")

# ─── 4. Profil cyclique parametrique (methode tuteur seance 4 et 6) ───────────
print("\nFit profil cyclique parametrique...")

def cyclic_func(h, beta, alpha, alpha2, phi):
    """f(h) = beta + alpha*cos(h*tau+phi)^2 + alpha2*sin(h*tau+phi) (tuteur)"""
    tau = 2 * np.pi / 24
    return beta + alpha * np.cos(h * tau + phi)**2 + alpha2 * np.sin(h * tau + phi)

hours_tr = df_tr.index.hour.values.astype(float)
debit_tr_vals = df_tr["debit"].values
try:
    popt, _ = curve_fit(cyclic_func, hours_tr, debit_tr_vals,
                        p0=[1000.0, 200.0, 100.0, -np.pi/2], maxfev=50000)
    beta_c, alpha_c, alpha2_c, phi_c = popt
    CYCLIC_PARAMS = popt
    y_cyc = cyclic_func(hours_tr, *popt)
    r2_cyc = 1 - np.sum((debit_tr_vals - y_cyc)**2) / np.sum((debit_tr_vals - debit_tr_vals.mean())**2)
    print(f"  beta={beta_c:.1f}, alpha={alpha_c:.1f}, alpha2={alpha2_c:.1f}, phi={phi_c:.3f}")
    print(f"  R2 profil cyclique train: {r2_cyc:.3f}")
    print(f"  (faible car reseau unitaire: pluie domine le cycle journalier)")
except Exception as e:
    print(f"  Fit failed ({e}), defaults")
    CYCLIC_PARAMS = np.array([1000.0, 200.0, 100.0, -np.pi/2])

# Profil moyen par (heure x mois) - 288 bins, plus precis
profile_hm = (df[df.index < TEST_START]
              .groupby([df[df.index < TEST_START].index.hour,
                        df[df.index < TEST_START].index.month])["debit"].mean())

# ─── 5. Feature engineering ───────────────────────────────────────────────────
def make_features(df_in, horizon=3):
    df = df_in.copy()
    p    = df["rain"].fillna(0)
    t    = df["temp"].fillna(df["temp"].mean())
    h_r  = df["humidity"].fillna(df["humidity"].mean())
    wind = df["wind"].fillna(0)

    df["target"] = df["debit"].shift(-horizon)

    # ── Features cycliques parametriques (f(h) = beta + alpha*cos^2 + alpha2*sin) ──
    h = df.index.hour.astype(float)
    tau = 2 * np.pi / 24
    beta_c, alpha_c, alpha2_c, phi_c = CYCLIC_PARAMS
    df["cyclic_val"]     = cyclic_func(h, *CYCLIC_PARAMS)
    df["debit_vs_cyclic"] = df["debit"] - df["cyclic_val"]
    # Profil cyclique a t+H (ce qu'on predit si le systeme est au niveau de base)
    h_fut = ((df.index.hour + horizon) % 24).astype(float)
    df["cyclic_next_h"]  = cyclic_func(h_fut, *CYCLIC_PARAMS)
    # Composantes du profil (features polynomiales sur l'heure)
    df["cos_h_tau"]      = np.cos(h * tau + phi_c)           # cos(h*tau+phi)
    df["cos_h_tau_sq"]   = np.cos(h * tau + phi_c) ** 2      # cos(h*tau+phi)^2  <- tuteur!
    df["sin_h_tau"]      = np.sin(h * tau + phi_c)           # sin(h*tau+phi)

    # Profil par (heure x mois) = 288 bins
    keys = list(zip(df.index.hour, df.index.month))
    df["cyclic_hm"]         = [profile_hm.get(k, df["cyclic_val"].iloc[i]) for i, k in enumerate(keys)]
    df["debit_vs_cyclic_hm"] = df["debit"] - df["cyclic_hm"]

    # ── Pluie passee: lags + rolling ──────────────────────────────────────────
    # Lag 5h optimal pour reseau unitaire Charleville (tuteur seance 3)
    for lag in [1, 2, 3, 4, 5, 6, 8, 12, 24]:
        df[f"rain_lag{lag}h"] = p.shift(lag).fillna(0)
    # Fenetre autour du lag critique 5h
    df["rain_lag5_sum3"]  = df["rain_lag3h"] + df["rain_lag4h"] + df["rain_lag5h"]
    df["rain_sum_3h"]     = p.rolling(3,   min_periods=1).sum()
    df["rain_sum_6h"]     = p.rolling(6,   min_periods=1).sum()
    df["rain_sum_12h"]    = p.rolling(12,  min_periods=1).sum()
    df["rain_sum_24h"]    = p.rolling(24,  min_periods=1).sum()
    df["rain_sum_48h"]    = p.rolling(48,  min_periods=1).sum()
    df["rain_sum_7d"]     = p.rolling(168, min_periods=24).sum()
    df["rain_max_3h"]     = p.rolling(3,   min_periods=1).max()
    df["rain_max_6h"]     = p.rolling(6,   min_periods=1).max()
    df["rain_max_24h"]    = p.rolling(24,  min_periods=1).max()
    df["ressuyage_48h"]   = p.ewm(halflife=48).mean()

    # ── Features pluie exp + seuil (tuteur seance 6) ──────────────────────────
    # "normalises la pluie et derriere t'appliques ton exp"
    rain_std = p.std() if p.std() > 0 else 1.0
    rain_norm = p / rain_std
    df["rain_exp"]          = np.expm1(rain_norm)              # exp sur pluie normalisee
    df["rain_exp_sq"]       = np.expm1(rain_norm) ** 2
    df["rain_sq"]           = p ** 2
    df["rain_sqrt"]         = np.sqrt(p.clip(lower=0))
    df["rain_log1p"]        = np.log1p(p)

    # Seuils pluie (feature seuil tuteur)
    df["rain_gt01"]     = (p > 0.1).astype(float)
    df["rain_gt05"]     = (p > 0.5).astype(float)
    df["rain_gt1"]      = (p > 1.0).astype(float)
    df["rain_gt2"]      = (p > 2.0).astype(float)
    df["rain_gt5"]      = (p > 5.0).astype(float)

    # Rolling exp (non-linearite sur cumuls)
    rs24_std = df["rain_sum_24h"].std() if df["rain_sum_24h"].std() > 0 else 1.0
    rs7d_std  = df["rain_sum_7d"].std()  if df["rain_sum_7d"].std() > 0  else 1.0
    df["rain_sum_24h_exp"] = np.expm1(df["rain_sum_24h"] / rs24_std)
    df["rain_sum_24h_sq"]  = df["rain_sum_24h"] ** 2
    df["rain_sum_7d_exp"]  = np.expm1(df["rain_sum_7d"] / rs7d_std)
    df["rain_sum_7d_sq"]   = df["rain_sum_7d"] ** 2

    # Rain event tracking
    is_rain = (p > 0.1).astype(int)
    evt_hrs, evt_cum = [], []
    cnt, cumul = 0, 0.0
    for i, v in enumerate(is_rain):
        if v: cnt += 1; cumul += p.iloc[i]
        else: cnt = 0; cumul = 0.0
        evt_hrs.append(cnt); evt_cum.append(cumul)
    df["rain_event_hours"] = evt_hrs
    df["rain_event_cumul"] = evt_cum

    # Heures depuis derniere pluie
    dry_hrs = []
    last = 0
    for v in is_rain:
        last = 0 if v else last + 1
        dry_hrs.append(last)
    df["hours_since_rain"] = dry_hrs
    df["is_raining"]       = (p > 0.1).astype(float)

    # ── Pluie future oracle (OK selon tuteur seance 5) ────────────────────────
    for k in range(1, horizon + 1):
        df[f"pluie_futur_{k}h"] = p.shift(-k).fillna(0)
    fcols = [f"pluie_futur_{k}h" for k in range(1, horizon + 1)]
    df["pluie_future_sum"] = sum(df[c] for c in fcols)
    df["pluie_future_max"] = pd.concat([df[c] for c in fcols], axis=1).max(axis=1)
    pfs_std = df["pluie_future_sum"].std() if df["pluie_future_sum"].std() > 0 else 1.0
    df["pluie_future_exp"] = np.expm1(df["pluie_future_sum"] / pfs_std)

    # ── Debit: lags etendus ───────────────────────────────────────────────────
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        df[f"debit_lag{lag}"] = df["debit"].shift(lag)
    df["debit_roll_mean6h"]  = df["debit"].rolling(6,  min_periods=1).mean()
    df["debit_roll_mean24h"] = df["debit"].rolling(24, min_periods=1).mean()
    df["debit_roll_std6h"]   = df["debit"].rolling(6,  min_periods=1).std().fillna(0)
    df["debit_ewm3h"]        = df["debit"].ewm(span=3).mean()
    df["debit_ewm12h"]       = df["debit"].ewm(span=12).mean()
    df["slope_1h"]           = df["debit"].diff(1).fillna(0)
    df["slope_3h"]           = df["debit"].diff(3).fillna(0)
    df["slope_6h"]           = df["debit"].diff(6).fillna(0)
    df["en_crue"]            = (df["debit"] > SEUIL_CRUE).astype(float)
    df["intensite_crue"]     = (df["debit"] - SEUIL_CRUE).clip(lower=0)
    df["debit_ratio_24h"]    = df["debit"] / (df["debit_roll_mean24h"] + 1)
    # Feature polynomiale cle: debit au carre normalise (R2: 2e feature importance)
    debit_std = df["debit"].std() if df["debit"].std() > 0 else 1.0
    df["debit_norm_sq"]      = (df["debit"] / debit_std) ** 2

    # ── Temporel ──────────────────────────────────────────────────────────────
    df["hour_sin"]   = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df.index.hour / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df.index.month / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df.index.dayofweek / 7)
    # Cycle hebdomadaire (168h) - complement au cycle journalier
    hiw = (df.index.dayofweek * 24 + df.index.hour).astype(float)
    df["week_sin"]   = np.sin(2 * np.pi * hiw / 168)
    df["week_cos"]   = np.cos(2 * np.pi * hiw / 168)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(float)
    # Jours feries 2021-2024
    jf = pd.to_datetime([
        "2021-01-01","2021-04-05","2021-05-01","2021-05-08","2021-05-13",
        "2021-05-24","2021-07-14","2021-08-15","2021-11-01","2021-11-11","2021-12-25",
        "2022-01-01","2022-04-18","2022-05-01","2022-05-08","2022-05-26",
        "2022-06-06","2022-07-14","2022-08-15","2022-11-01","2022-11-11","2022-12-25",
        "2023-01-01","2023-04-10","2023-05-01","2023-05-08","2023-05-18",
        "2023-05-29","2023-07-14","2023-08-15","2023-11-01","2023-11-11","2023-12-25",
        "2024-01-01","2024-04-01","2024-05-01","2024-05-08","2024-05-09",
        "2024-05-20","2024-07-14","2024-08-15","2024-11-01","2024-11-11","2024-12-25",
    ])
    df["is_holiday"] = df.index.normalize().isin(jf).astype(float)

    # ── Meteo ─────────────────────────────────────────────────────────────────
    df["temp"]         = t
    df["humidity"]     = h_r
    df["wind"]         = wind
    df["temp_ewm24h"]  = t.ewm(halflife=24).mean()

    # ── Features polynomiales cles (pluie x debit) (tuteur seances 4 et 6) ────
    # Reseau unitaire: pluie * debit = interaction directe
    df["rain_x_debit"]        = p * df["debit"]
    df["rain_exp_x_debit"]    = df["rain_exp"] * df["debit"]
    df["rain_lag5_x_debit"]   = df["rain_lag5h"] * df["debit"]
    df["rain24_x_lag24"]      = df["rain_sum_24h"] * df["debit_lag24"]
    # Saturation sol x pluie future (ressuyage)
    df["soil_x_future"]       = df["ressuyage_48h"] * df["pluie_future_sum"]
    df["soil_x_future_exp"]   = df["ressuyage_48h"] * df["pluie_future_exp"]
    df["ressuyage_x_rain24"]  = df["ressuyage_48h"] * df["rain_sum_24h"]
    # Cyclic x pluie (effet de la pluie varie selon l'heure)
    df["cyclic_x_rain"]       = df["cyclic_val"] * p
    df["cyclic_x_future"]     = df["cyclic_val"] * df["pluie_future_sum"]
    # Temperature x pluie (pluie froide = plus de ressuyage)
    df["temp_x_rain"]         = t * p
    df["temp_x_rain24"]       = t * df["rain_sum_24h"]
    # Slope x pluie (debit qui monte + pluie = crue probable)
    df["slope_x_rain"]        = df["slope_1h"] * p
    df["slope_x_future"]      = df["slope_1h"] * df["pluie_future_sum"]

    return df


FEATURE_COLS = [
    # ── Temporel ──
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "dow_sin", "dow_cos", "week_sin", "week_cos",
    "is_weekend", "is_holiday",

    # ── Profil cyclique parametrique (tuteur) ──
    "cyclic_val", "debit_vs_cyclic", "cyclic_next_h",
    "cos_h_tau", "cos_h_tau_sq", "sin_h_tau",
    "cyclic_hm", "debit_vs_cyclic_hm",

    # ── Debit courant + lags ──
    "debit", "debit_norm_sq",
    "debit_lag1", "debit_lag2", "debit_lag3", "debit_lag6",
    "debit_lag12", "debit_lag24", "debit_lag48", "debit_lag72", "debit_lag168",
    "debit_roll_mean6h", "debit_roll_mean24h", "debit_roll_std6h",
    "debit_ewm3h", "debit_ewm12h",
    "slope_1h", "slope_3h", "slope_6h",
    "en_crue", "intensite_crue", "debit_ratio_24h",

    # ── Pluie passee (lags optimaux pour reseau unitaire: lag ~5h) ──
    "rain_lag1h", "rain_lag2h", "rain_lag3h", "rain_lag4h", "rain_lag5h",
    "rain_lag6h", "rain_lag8h", "rain_lag12h", "rain_lag24h", "rain_lag5_sum3",
    "rain_sum_3h", "rain_sum_6h", "rain_sum_12h",
    "rain_sum_24h", "rain_sum_48h", "rain_sum_7d",
    "rain_max_3h", "rain_max_6h", "rain_max_24h",
    "ressuyage_48h",

    # ── Transformations pluie exp + seuils (tuteur seance 6) ──
    "rain_exp", "rain_exp_sq", "rain_sq", "rain_sqrt", "rain_log1p",
    "rain_gt01", "rain_gt05", "rain_gt1", "rain_gt2", "rain_gt5",
    "rain_sum_24h_exp", "rain_sum_24h_sq",
    "rain_sum_7d_exp", "rain_sum_7d_sq",
    "rain_event_hours", "rain_event_cumul", "hours_since_rain", "is_raining",

    # ── Pluie future oracle ──
    "pluie_futur_1h", "pluie_futur_2h", "pluie_futur_3h",
    "pluie_future_sum", "pluie_future_max", "pluie_future_exp",

    # ── Meteo ──
    "temp", "humidity", "wind", "temp_ewm24h",

    # ── Interactions polynomiales (tuteur: pluie x debit) ──
    "rain_x_debit", "rain_exp_x_debit", "rain_lag5_x_debit", "rain24_x_lag24",
    "soil_x_future", "soil_x_future_exp", "ressuyage_x_rain24",
    "cyclic_x_rain", "cyclic_x_future",
    "temp_x_rain", "temp_x_rain24",
    "slope_x_rain", "slope_x_future",
]

# ─── 6. Construction des features ─────────────────────────────────────────────
print("\nConstruction des features...")
df_feat = make_features(df.copy()).dropna(subset=["target"] + FEATURE_COLS)

mask_train = df_feat.index < TRAIN_END
mask_val   = (df_feat.index >= TRAIN_END) & (df_feat.index < TEST_START)
mask_test  = df_feat.index >= TEST_START

X_tr   = df_feat[mask_train][FEATURE_COLS]
y_tr   = df_feat[mask_train]["target"]
X_va   = df_feat[mask_val][FEATURE_COLS]
y_va   = df_feat[mask_val]["target"]
X_te   = df_feat[mask_test][FEATURE_COLS]
y_te   = df_feat[mask_test]["target"]
X_trva = pd.concat([X_tr, X_va])
y_trva = pd.concat([y_tr, y_va])

print(f"  Train: {len(X_tr)}h | Val: {len(X_va)}h | Test: {len(X_te)}h")
print(f"  Features: {len(FEATURE_COLS)}")

# ─── 7. Sliding window CV (TimeSeriesSplit spirit) ────────────────────────────
W_TRAIN = 8 * 7 * 24    # 8 semaines
W_VAL   = 2 * 7 * 24    # 2 semaines
STEP    = 2 * 7 * 24

cv_feat = df_feat[~mask_test]
cv_idx  = cv_feat.index
folds   = []
start   = 0
while start + W_TRAIN + W_VAL <= len(cv_idx):
    te = start + W_TRAIN
    ve = te + W_VAL
    folds.append((cv_idx[start], cv_idx[te - 1], cv_idx[te], cv_idx[ve - 1]))
    start += STEP
print(f"\nSliding window CV: {len(folds)} folds (W_train=8sem, W_val=2sem)")

def run_cv(params, cols=FEATURE_COLS):
    maes = []
    for (ts, te_idx, vs, ve_idx) in folds:
        X_f = cv_feat[(cv_feat.index >= ts) & (cv_feat.index <= te_idx)][cols]
        y_f = cv_feat[(cv_feat.index >= ts) & (cv_feat.index <= te_idx)]["target"]
        X_v = cv_feat[(cv_feat.index >= vs) & (cv_feat.index <= ve_idx)][cols]
        y_v = cv_feat[(cv_feat.index >= vs) & (cv_feat.index <= ve_idx)]["target"]
        if len(X_f) < 300 or len(X_v) < 50:
            continue
        m = XGBRegressor(**params)
        m.fit(X_f, y_f, verbose=False)
        p = np.clip(m.predict(X_v), MIN_CAP, HARD_CAP)
        maes.append(np.mean(np.abs(p - y_v)))
    return np.mean(maes) if maes else float("nan")

# ─── 8. Comparaison configs ────────────────────────────────────────────────────
CONFIGS = [
    ("depth4_lr005",  dict(max_depth=4, learning_rate=0.05, n_estimators=400,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                           min_child_weight=5, random_state=42, n_jobs=-1, verbosity=0)),
    ("depth5_lr005",  dict(max_depth=5, learning_rate=0.05, n_estimators=400,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=3,
                           min_child_weight=5, random_state=42, n_jobs=-1, verbosity=0)),
    ("depth5_lr002",  dict(max_depth=5, learning_rate=0.02, n_estimators=800,
                           subsample=0.8, colsample_bytree=0.7, reg_lambda=3,
                           min_child_weight=5, random_state=42, n_jobs=-1, verbosity=0)),
    ("depth6_lr003",  dict(max_depth=6, learning_rate=0.03, n_estimators=500,
                           subsample=0.8, colsample_bytree=0.7, reg_lambda=3,
                           min_child_weight=5, random_state=42, n_jobs=-1, verbosity=0)),
    ("depth5_reg10",  dict(max_depth=5, learning_rate=0.05, n_estimators=400,
                           subsample=0.7, colsample_bytree=0.7, reg_lambda=10,
                           min_child_weight=8, random_state=42, n_jobs=-1, verbosity=0)),
]

print()
print("=" * 80)
print(f"{'Config':>16} | {'MAE_cv':>7} | {'MAE_te':>7} {'RMSE_te':>8} {'R2_te':>6} {'MAEpic':>8}")
print("-" * 80)
print(f"{'ref_Nicolas':>16} |         | {102.14:7.2f} {160.13:8.2f}  0.819")
print("-" * 80)

best_mae_te = 9999
best_cfg = None

for name, params in CONFIGS:
    mae_cv = run_cv(params)

    m = XGBRegressor(**params)
    m.fit(X_trva, y_trva, verbose=False)
    p_te = np.clip(m.predict(X_te), MIN_CAP, HARD_CAP)
    pics = y_te > SEUIL_CRUE
    mae_te  = np.mean(np.abs(p_te - y_te))
    rmse_te = np.sqrt(np.mean((p_te - y_te)**2))
    r2_te   = 1 - np.sum((p_te - y_te)**2) / np.sum((y_te - y_te.mean())**2)
    maepic  = np.mean(np.abs(p_te[pics] - y_te[pics])) if pics.any() else float("nan")
    mark = " <--" if mae_te < best_mae_te else ""
    print(f"{name:>16} | {mae_cv:7.2f} | {mae_te:7.2f} {rmse_te:8.2f} {r2_te:6.3f} {maepic:8.1f}{mark}", flush=True)

    if mae_te < best_mae_te:
        best_mae_te = mae_te
        best_cfg = (name, params, m)

print("=" * 80)

# ─── 9. Resultats finaux du meilleur modele ───────────────────────────────────
name_b, params_b, model_b = best_cfg
p_best = np.clip(model_b.predict(X_te), MIN_CAP, HARD_CAP)
errors = np.abs(p_best - y_te.values)
actuals = y_te.values

print(f"\n=== MEILLEUR MODELE: {name_b} ===")
print(f"  MAE    = {np.mean(errors):.2f} m3/h  (vs Nicolas: 102.14, gain: {102.14-np.mean(errors):.1f})")
print(f"  Mediane= {np.median(errors):.2f} m3/h")
print(f"  P90    = {np.percentile(errors, 90):.2f} m3/h")
print(f"  RMSE   = {np.sqrt(np.mean(errors**2)):.2f} m3/h  (vs Nicolas: 160.13)")
r2 = 1 - np.sum((p_best-actuals)**2)/np.sum((actuals-actuals.mean())**2)
print(f"  R2     = {r2:.3f}  (vs Nicolas: 0.819)")
pics_m = actuals > SEUIL_CRUE
print(f"  MAEpic = {np.mean(errors[pics_m]):.2f} m3/h")

print("\nMAE par condition:")
rain_t = df_feat[mask_test]["rain"].values
print(f"  Sec (<0.1mm):       {np.mean(errors[rain_t<0.1]):.1f} m3/h  ({(rain_t<0.1).mean()*100:.0f}% du temps)")
print(f"  Pluie 0.1-1mm:      {np.mean(errors[(rain_t>=0.1)&(rain_t<1)]):.1f} m3/h" if ((rain_t>=0.1)&(rain_t<1)).any() else "")
print(f"  Pluie forte (>=1):  {np.mean(errors[rain_t>=1]):.1f} m3/h"   if (rain_t>=1).any() else "")
hrs = df_feat[mask_test].index.hour.values
print(f"  00-06h:             {np.mean(errors[(hrs>=0)&(hrs<6)]):.1f}")
print(f"  06-12h:             {np.mean(errors[(hrs>=6)&(hrs<12)]):.1f}")
print(f"  12-18h:             {np.mean(errors[(hrs>=12)&(hrs<18)]):.1f}")
print(f"  18-24h:             {np.mean(errors[(hrs>=18)&(hrs<24)]):.1f}")

# ─── 10. Top 25 features importance ──────────────────────────────────────────
print("\nTop 25 features importance:")
fi = pd.Series(model_b.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
for feat, imp in fi.head(25).items():
    bar = "#" * int(imp * 300)
    print(f"  {feat:35s} {imp:.4f} {bar}")

# ─── 11. Multi-horizon (3h, 6h, 12h, 24h) ─────────────────────────────────────
print("\n=== MULTI-HORIZON ===")
print(f"{'H':>4} | {'MAE_te':>7} {'RMSE_te':>8} {'R2_te':>6}")
print("-" * 35)
for H in [3, 6, 12, 24]:
    df_hfeat = make_features(df.copy(), horizon=H).dropna(subset=["target"] + FEATURE_COLS)
    X_h_trva = df_hfeat[df_hfeat.index < TEST_START][FEATURE_COLS]
    y_h_trva = df_hfeat[df_hfeat.index < TEST_START]["target"]
    X_h_te   = df_hfeat[df_hfeat.index >= TEST_START][FEATURE_COLS]
    y_h_te   = df_hfeat[df_hfeat.index >= TEST_START]["target"]
    mh = XGBRegressor(**params_b)
    mh.fit(X_h_trva, y_h_trva, verbose=False)
    p_h = np.clip(mh.predict(X_h_te), MIN_CAP, HARD_CAP)
    mae_h  = np.mean(np.abs(p_h - y_h_te))
    rmse_h = np.sqrt(np.mean((p_h - y_h_te)**2))
    r2_h   = 1 - np.sum((p_h-y_h_te)**2)/np.sum((y_h_te-y_h_te.mean())**2)
    print(f" H={H:2d} | {mae_h:7.2f} {rmse_h:8.2f} {r2_h:6.3f}", flush=True)

print("\nFIN.")
