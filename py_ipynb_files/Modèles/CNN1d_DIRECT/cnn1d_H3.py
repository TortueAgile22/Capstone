#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cnn1d_H3.py — Modèle CNN 1D (TCN) hybride  |  Station P  |  H+3h
==================================================================
Prédiction directe du débit à l'horizon H=3h.

Architecture hybride :
  ┌─ Séquence 48h (23 features brutes) ─► TCN (5 blocs dilatés) ─► vecteur 64-dim ─┐
  ├─ Features artisanales XGBoost (21 features snapshot à t) ────────────────────────┤─► Dense(64)─►Dense(1)
  └─ Oracle rain (prev_pluie_1h/2h/3h) ──────────────────────────────────────────────┘

Améliorations v2 :
  - Dilations [1,2,4,8,16] → champ récepteur = 63h (couvre le lookback 48h)
  - Features artisanales G6+G12 injectées dans la tête dense (idem XGBoost)
  - Filtres 64 (vs 32), patience 30 (vs 20)
  - Tête Dense(64+21+3, 64) → Dense(32) → Dense(1)

Comparaison cible (XGBoost H=3h) :
  MAE test = 40.54 m³/h  |  MAE crues = 134.27 m³/h  |  R² = 0.740
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
#  CHEMINS & HYPERPARAMÈTRES
# ══════════════════════════════════════════════════════════════
BASE             = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\bronze\bronze")
OUT              = Path(r"C:\Users\Cecil\Documents\Cours 3A\M2 Data Science\CAPSTONE\Capstone\py_ipynb_files\Modèles\CNN1d_DIRECT")
POST_SHIFT_START = "2025-04-03"
HORIZON          = 3
LOOKBACK         = 48
W_TEST_FINAL     = 8 * 7 * 24
W_VAL_TRAIN      = 3 * 7 * 24

# Hyperparamètres v2
N_FILTERS    = 64
KERNEL_SIZE  = 3
DILATIONS    = [1, 2, 4, 8, 16]   # champ récepteur = 1 + 2*(1+2+4+8+16) = 63h
DROPOUT      = 0.2
BATCH_SIZE   = 128
MAX_EPOCHS   = 200
LR           = 1e-3
PATIENCE     = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
print(f"  Device : {DEVICE}")

# ══════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("2. FEATURE ENGINEERING")
print("=" * 65)

SEUIL_CRUE = float(df["debit"].quantile(0.90))
h_arr = df.index.hour.astype(float)

# ── Cyclic custom ──────────────────────────────────────────────
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
    except Exception:
        pass
if not CYCLIC_OK:
    df["cyclic_custom"] = np.sin(2 * np.pi * h_arr / 24) * 100 + 200

# ── Features de base (calcul pour toutes les features) ────────
df["slope_1h"]          = df["debit"] - df["debit"].shift(1)
df["slope_3h"]          = df["debit"] - df["debit"].shift(3)
df["debit_lag1"]        = df["debit"].shift(1)
df["debit_lag3"]        = df["debit"].shift(3)
df["debit_lag6"]        = df["debit"].shift(6)
df["debit_lag24"]       = df["debit"].shift(24)
df["debit_lag48"]       = df["debit"].shift(48)
df["debit_lag7d"]       = df["debit"].shift(7 * 24)
df["hour_sin"]          = np.sin(2 * np.pi * h_arr / 24)
df["hour_cos"]          = np.cos(2 * np.pi * h_arr / 24)
df["ressuyage_exp"]     = df["rain"].ewm(halflife=48, adjust=False).mean()
rain_std                = df["rain"].std()
df["rain_exp_norm"]     = np.expm1(df["rain"] / rain_std) if rain_std > 0 else 0.
df["is_raining_hard"]   = (df["rain"] > 1.0).astype(float)
df["en_crue"]           = (df["debit"] > SEUIL_CRUE).astype(float)
df["intensite_crue"]    = np.maximum(df["debit"] - SEUIL_CRUE, 0.)
df["is_weekend"]        = (df.index.dayofweek >= 5).astype(float)
df["is_monday_morning"] = ((df.index.dayofweek == 0) & (df.index.hour < 10)).astype(float)
df["is_holiday"]        = 0.
if HAS_HOLIDAYS:
    fr_hol = hol_lib.France(years=range(2023, 2027))
    df["is_holiday"] = df.index.normalize().isin(fr_hol).astype(float)
df["rain_sum_3h"]       = df["rain"].rolling(3).sum()
df["rain_sum_6h"]       = df["rain"].rolling(6).sum()
df["rain_sum_24h"]      = df["rain"].rolling(24).sum()
df["rain_sum_3d"]       = df["rain"].rolling(72).sum()
df["rain_sum_7d"]       = df["rain"].rolling(168).sum()
df["rain_max_3h"]       = df["rain"].rolling(3).max()
df["debit_min_6h"]      = df["debit"].rolling(6).min()
df["debit_min_12h"]     = df["debit"].rolling(12).min()
df["debit_min_24h"]     = df["debit"].rolling(24).min()
df["debit_rise_6h"]     = df["debit"] - df["debit_min_6h"]
df["debit_rise_12h"]    = df["debit"] - df["debit_min_12h"]
df["debit_rise_24h"]    = df["debit"] - df["debit_min_24h"]
for lag in [6, 9, 12, 15, 18, 24, 30, 36, 48]:
    df[f"rain_lag_{lag}h"] = df["rain"].shift(lag)
df["rain_lag_24h_x_rise_6h"]  = df["rain_lag_24h"] * df["debit_rise_6h"]
df["rain_lag_12h_x_rise_6h"]  = df["rain_lag_12h"] * df["debit_rise_6h"]
df["crue_retardee_signal"]     = (
    (df["debit_rise_6h"] > 50) & (df["rain_sum_6h"] < 1.0)
).astype(float)

# Oracle rain
df["prev_pluie_1h"]  = df["rain"].shift(-1)
df["prev_pluie_2h"]  = df["rain"].shift(-2)
df["prev_pluie_3h"]  = df["rain"].shift(-3)

# Target
df["target"] = df["debit"].shift(-HORIZON)

# ── Définition des trois groupes de features ──────────────────

# SEQ_FEATS : features présentes à chaque timestep de la fenêtre 48h
# (features "brutes" que le TCN traite séquentiellement)
SEQ_FEATS = [
    "debit", "rain", "temperature_2m",
    "slope_1h", "slope_3h",
    "hour_sin", "hour_cos", "cyclic_custom",
    "ressuyage_exp", "en_crue", "intensite_crue",
    "is_weekend", "is_monday_morning", "is_holiday",
    "rain_sum_3h", "rain_sum_6h", "rain_sum_24h",
    "debit_min_6h", "debit_rise_6h",
    "rain_lag_6h", "rain_lag_12h", "rain_lag_24h",
    "crue_retardee_signal",
]

# STATIC_FEATS : features artisanales snapshot à t (injectées dans la tête dense)
# Mêmes que XGBoost G6+G12 — le CNN n'a pas à les redécouvrir seul
STATIC_FEATS = [
    # Lags long terme (au bord ou hors de la fenêtre 48h)
    "debit_lag24", "debit_lag48", "debit_lag7d",
    # Cyclic explicite (pattern journalier — top feature XGBoost)
    "cyclic_custom",
    # Rolling long terme (hors fenêtre 48h)
    "rain_sum_3d", "rain_sum_7d",
    # Lags pluie G12
    "rain_lag_9h", "rain_lag_15h", "rain_lag_18h",
    "rain_lag_30h", "rain_lag_36h", "rain_lag_48h",
    # Interactions pluie × crue (features non-linéaires clés)
    "rain_lag_24h_x_rise_6h", "rain_lag_12h_x_rise_6h",
    # Baseflow étendu
    "debit_min_12h", "debit_min_24h",
    "debit_rise_12h", "debit_rise_24h",
    # Features diverses
    "rain_exp_norm", "is_raining_hard", "rain_max_3h",
]

# ORACLE_FEATS : pluie future connue au moment de la prédiction
ORACLE_FEATS = ["prev_pluie_1h", "prev_pluie_2h", "prev_pluie_3h"]

ALL_NEEDED = SEQ_FEATS + STATIC_FEATS + ORACLE_FEATS + ["target"]
df_clean = df.dropna(subset=list(set(ALL_NEEDED))).copy()

n = len(df_clean)
df_cv   = df_clean.iloc[: n - W_TEST_FINAL]
df_test = df_clean.iloc[n - W_TEST_FINAL :]
df_train = df_cv.iloc[: len(df_cv) - W_VAL_TRAIN]
df_val   = df_cv.iloc[len(df_cv) - W_VAL_TRAIN :]

print(f"  Seuil crue P90 = {SEUIL_CRUE:.1f} m³/h")
print(f"  SEQ features    : {len(SEQ_FEATS)}")
print(f"  STATIC features : {len(STATIC_FEATS)}  (artisanales XGBoost)")
print(f"  ORACLE features : {len(ORACLE_FEATS)}")
print(f"  CV   : {df_cv.index.min().date()} -> {df_cv.index.max().date()}  ({len(df_cv)} h)")
print(f"  Test : {df_test.index.min().date()} -> {df_test.index.max().date()}  ({len(df_test)} h)")

# ══════════════════════════════════════════════════════════════
#  3. NORMALISATION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("3. NORMALISATION (fitté sur train uniquement)")
print("=" * 65)

sc_seq    = StandardScaler()
sc_static = StandardScaler()
sc_oracle = StandardScaler()
sc_target = StandardScaler()

sc_seq.fit(df_train[SEQ_FEATS].values)
sc_static.fit(df_train[STATIC_FEATS].values)
sc_oracle.fit(df_train[ORACLE_FEATS].values)
sc_target.fit(df_train[["target"]].values)

def scale_part(df_part):
    seq = sc_seq.transform(df_part[SEQ_FEATS].values).astype(np.float32)
    sta = sc_static.transform(df_part[STATIC_FEATS].values).astype(np.float32)
    orc = sc_oracle.transform(df_part[ORACLE_FEATS].values).astype(np.float32)
    tgt = sc_target.transform(df_part[["target"]].values).ravel().astype(np.float32)
    return seq, sta, orc, tgt

seq_train, sta_train, orc_train, tgt_train = scale_part(df_train)
seq_val,   sta_val,   orc_val,   tgt_val   = scale_part(df_val)
seq_test,  sta_test,  orc_test,  tgt_test  = scale_part(df_test)

print(f"  Fitté sur {len(df_train)} h  |  3 scalers (seq / static / oracle)")

# ══════════════════════════════════════════════════════════════
#  4. DATASET
# ══════════════════════════════════════════════════════════════
class DebitDataset(Dataset):
    def __init__(self, seq, sta, orc, tgt, lookback):
        self.seq = seq
        self.sta = sta
        self.orc = orc
        self.tgt = tgt
        self.L   = lookback

    def __len__(self):
        return len(self.tgt) - self.L

    def __getitem__(self, i):
        x_seq = torch.from_numpy(self.seq[i: i + self.L])       # (L, n_seq)
        x_sta = torch.from_numpy(self.sta[i + self.L - 1])      # snapshot à t
        x_orc = torch.from_numpy(self.orc[i + self.L - 1])      # snapshot à t
        y     = torch.tensor(self.tgt[i + self.L - 1])
        return x_seq, x_sta, x_orc, y

ds_train = DebitDataset(seq_train, sta_train, orc_train, tgt_train, LOOKBACK)
ds_val   = DebitDataset(seq_val,   sta_val,   orc_val,   tgt_val,   LOOKBACK)
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False)

print(f"  Samples train : {len(ds_train):,}   val : {len(ds_val):,}")

# ══════════════════════════════════════════════════════════════
#  5. ARCHITECTURE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("5. ARCHITECTURE CNN 1D hybride (TCN + features artisanales)")
print("=" * 65)

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
                 n_filters=64, kernel=3, dilations=(1, 2, 4, 8, 16), dropout=0.2):
        super().__init__()
        layers = []
        in_ch = n_seq
        for d in dilations:
            layers.append(TCNBlock(in_ch, n_filters, kernel, d, dropout))
            in_ch = n_filters
        self.tcn = nn.Sequential(*layers)

        # Tête hybride : TCN_out + features artisanales + oracle rain
        dense_in = n_filters + n_static + n_oracle
        self.head = nn.Sequential(
            nn.Linear(dense_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq, x_sta, x_orc):
        x = self.tcn(x_seq.permute(0, 2, 1))   # (B, n_filters, L)
        x = x[:, :, -1]                         # dernier timestep (B, n_filters)
        x = torch.cat([x, x_sta, x_orc], dim=1)
        return self.head(x).squeeze(1)


N_SEQ    = len(SEQ_FEATS)
N_STATIC = len(STATIC_FEATS)
N_ORC    = len(ORACLE_FEATS)

model = TCNHybrid(N_SEQ, N_STATIC, N_ORC,
                  n_filters=N_FILTERS,
                  kernel=KERNEL_SIZE,
                  dilations=DILATIONS,
                  dropout=DROPOUT).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
receptive_field = 1 + (KERNEL_SIZE - 1) * sum(DILATIONS)
dense_in = N_FILTERS + N_STATIC + N_ORC
print(f"  Paramètres entraînables : {n_params:,}")
print(f"  Champ récepteur TCN     : {receptive_field}h  (lookback = {LOOKBACK}h) ✓")
print(f"  Entrée tête dense       : {N_FILTERS} (TCN) + {N_STATIC} (static) + {N_ORC} (oracle) = {dense_in}")
print(f"  Dilations               : {DILATIONS}")

# ══════════════════════════════════════════════════════════════
#  6. ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("6. ENTRAÎNEMENT")
print("=" * 65)

optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5, min_lr=1e-5)
criterion  = nn.HuberLoss(delta=1.0)

best_val   = float("inf")
best_state = None
patience_c = 0
train_hist, val_hist = [], []

for epoch in range(1, MAX_EPOCHS + 1):
    # ── Train ────────────────────────────────────────────────
    model.train()
    ep_loss = 0.
    for x_seq, x_sta, x_orc, y in dl_train:
        x_seq, x_sta, x_orc, y = (x_seq.to(DEVICE), x_sta.to(DEVICE),
                                   x_orc.to(DEVICE), y.to(DEVICE))
        optimizer.zero_grad()
        loss = criterion(model(x_seq, x_sta, x_orc), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ep_loss += loss.item() * len(y)
    ep_loss /= len(ds_train)

    # ── Validation ───────────────────────────────────────────
    model.eval()
    v_loss = 0.
    with torch.no_grad():
        for x_seq, x_sta, x_orc, y in dl_val:
            x_seq, x_sta, x_orc, y = (x_seq.to(DEVICE), x_sta.to(DEVICE),
                                       x_orc.to(DEVICE), y.to(DEVICE))
            v_loss += criterion(model(x_seq, x_sta, x_orc), y).item() * len(y)
    v_loss /= len(ds_val)

    scheduler.step(v_loss)
    train_hist.append(ep_loss)
    val_hist.append(v_loss)

    if v_loss < best_val:
        best_val   = v_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_c = 0
    else:
        patience_c += 1

    if epoch % 10 == 0 or epoch == 1:
        lr_cur = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{MAX_EPOCHS}  "
              f"train={ep_loss:.4f}  val={v_loss:.4f}  "
              f"lr={lr_cur:.1e}  patience={patience_c}/{PATIENCE}")

    if patience_c >= PATIENCE:
        print(f"\n  Early stopping à epoch {epoch}")
        break

model.load_state_dict(best_state)
torch.save(best_state, OUT / "cnn1d_H3_weights.pt")
print(f"\n  Meilleur val_loss = {best_val:.4f}")
print(f"  Poids sauvegardés -> cnn1d_H3_weights.pt")

# ══════════════════════════════════════════════════════════════
#  7. PRÉDICTIONS
# ══════════════════════════════════════════════════════════════
def predict_full(seq_arr, sta_arr, orc_arr, n_rows):
    model.eval()
    preds_sc = []
    with torch.no_grad():
        for i in range(0, n_rows - LOOKBACK, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n_rows - LOOKBACK)
            xs = torch.from_numpy(
                np.stack([seq_arr[j: j + LOOKBACK] for j in range(i, end)])
            ).to(DEVICE)
            xst = torch.from_numpy(sta_arr[i + LOOKBACK - 1: end + LOOKBACK - 1]).to(DEVICE)
            xo  = torch.from_numpy(orc_arr[i + LOOKBACK - 1: end + LOOKBACK - 1]).to(DEVICE)
            preds_sc.append(model(xs, xst, xo).cpu().numpy())
    return np.concatenate(preds_sc)

def get_preds(seq_arr, sta_arr, orc_arr, df_part):
    preds_sc = predict_full(seq_arr, sta_arr, orc_arr, len(df_part))
    preds = sc_target.inverse_transform(preds_sc.reshape(-1, 1)).ravel()
    n     = len(preds)
    trues = df_part["target"].values[LOOKBACK - 1: LOOKBACK - 1 + n]
    naive = df_part["debit"].values[LOOKBACK - 1: LOOKBACK - 1 + n]
    return preds, trues, naive

preds_test, y_test, n_test = get_preds(seq_test, sta_test, orc_test, df_test)

# ══════════════════════════════════════════════════════════════
#  8. MÉTRIQUES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("8. MÉTRIQUES — TEST SET (Nov-Jan)")
print("=" * 65)

mae_test  = mean_absolute_error(y_test, preds_test)
rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
r2_test   = r2_score(y_test, preds_test)
mae_naive = mean_absolute_error(y_test, n_test)
gain      = (1 - mae_test / mae_naive) * 100

mask_crue = y_test > SEUIL_CRUE
mae_crue  = mean_absolute_error(y_test[mask_crue], preds_test[mask_crue]) if mask_crue.sum() else np.nan
n_pics    = mask_crue.sum()

rain_test = df_test["rain"].values[LOOKBACK - 1: LOOKBACK - 1 + len(preds_test)]
mask_rain = rain_test > 0.1
mae_pluie = mean_absolute_error(y_test[mask_rain], preds_test[mask_rain]) if mask_rain.sum() else np.nan

print(f"\n  MAE  test  = {mae_test:.2f} m³/h")
print(f"  RMSE test  = {rmse_test:.2f} m³/h")
print(f"  R²   test  = {r2_test:.3f}")
print(f"  MAE  naïf  = {mae_naive:.2f} m³/h   gain = {gain:.1f}%")
print(f"  MAE  crues = {mae_crue:.2f} m³/h   (n_pics = {n_pics})")
print(f"  MAE  pluie = {mae_pluie:.2f} m³/h")

print(f"\n  ── Comparaison ──────────────────────────────────────────")
print(f"  {'Modèle':<26}  {'MAE':>7}  {'MAE crues':>10}  {'R²':>6}")
print(f"  {'─'*55}")
print(f"  {'XGBoost H=3h (direct)':<26}  {'40.54':>7}  {'134.27':>10}  {'0.740':>6}")
print(f"  {'CNN 1D v1 (sans artis.)':<26}  {'46.95':>7}  {'166.94':>10}  {'0.593':>6}")
print(f"  {'CNN 1D v2 (+ artisanales)':<26}  {mae_test:>7.2f}  {mae_crue:>10.2f}  {r2_test:>6.3f}")
print(f"  {'Naïf (persistence)':<26}  {mae_naive:>7.2f}  {'—':>10}  {'—':>6}")

# ══════════════════════════════════════════════════════════════
#  9. GRAPHIQUES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("9. GÉNÉRATION DES GRAPHIQUES")
print("=" * 65)

idx_test = df_test.index[LOOKBACK - 1: LOOKBACK - 1 + len(preds_test)]

fig = plt.figure(figsize=(18, 14))
fig.suptitle("CNN 1D hybride (TCN + features artisanales) — Station P — H+3h",
             fontsize=14, fontweight="bold")
gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32,
              top=0.94, bottom=0.07, left=0.07, right=0.97)

# (A) Série temporelle test
ax_ts = fig.add_subplot(gs[0, :])
ax_ts.plot(idx_test, y_test,     color="steelblue", lw=1.2, label="Réel",       alpha=0.9)
ax_ts.plot(idx_test, preds_test, color="tomato",    lw=1.0, label="CNN 1D v2",  alpha=0.85)
ax_ts.axhline(SEUIL_CRUE, color="orange", lw=1, ls="--",
              label=f"Seuil crue ({SEUIL_CRUE:.0f} m³/h)")
ax_ts.set_title(f"(A) Test set  |  MAE={mae_test:.2f} m³/h  R²={r2_test:.3f}", fontsize=11)
ax_ts.set_ylabel("Débit (m³/h)")
ax_ts.legend(fontsize=9)

# (B) Scatter
ax_sc = fig.add_subplot(gs[1, 0])
c_val = np.where(mask_crue, "tomato", "steelblue")
ax_sc.scatter(y_test, preds_test, c=c_val, s=4, alpha=0.4)
lim = max(y_test.max(), preds_test.max())
ax_sc.plot([0, lim], [0, lim], "k--", lw=1)
ax_sc.set_title(f"(B) Réel vs Prédit  |  R²={r2_test:.3f}", fontsize=11)
ax_sc.set_xlabel("Débit réel (m³/h)")
ax_sc.set_ylabel("Débit prédit (m³/h)")
ax_sc.legend(handles=[mpatches.Patch(color="tomato", label="Crue"),
                       mpatches.Patch(color="steelblue", label="Normal")], fontsize=9)

# (C) Distribution erreurs
ax_err = fig.add_subplot(gs[1, 1])
errors = preds_test - y_test
ax_err.hist(errors, bins=60, color="steelblue", alpha=0.75, edgecolor="white")
ax_err.axvline(0,             color="black",  lw=1.5)
ax_err.axvline(errors.mean(), color="tomato", lw=1.5, ls="--",
               label=f"Biais = {errors.mean():.1f} m³/h")
ax_err.set_title(f"(C) Erreurs  |  std={errors.std():.1f} m³/h", fontsize=11)
ax_err.set_xlabel("Erreur (m³/h)")
ax_err.legend(fontsize=9)

# (D) Courbe d'apprentissage
ax_lc = fig.add_subplot(gs[2, 0])
ep_x = list(range(1, len(train_hist) + 1))
ax_lc.plot(ep_x, train_hist, label="Train loss", color="steelblue")
ax_lc.plot(ep_x, val_hist,   label="Val loss",   color="tomato")
best_ep = int(np.argmin(val_hist)) + 1
ax_lc.axvline(best_ep, color="green", lw=1.5, ls="--", label=f"Best epoch={best_ep}")
ax_lc.set_title("(D) Courbe d'apprentissage", fontsize=11)
ax_lc.set_xlabel("Epoch")
ax_lc.set_ylabel("Huber Loss")
ax_lc.legend(fontsize=9)

# (E) Zoom crue
ax_z = fig.add_subplot(gs[2, 1])
if mask_crue.sum() > 48:
    fc = np.where(mask_crue)[0][0]
    z0, z1 = max(0, fc - 24), min(len(y_test), fc + 96)
    ax_z.plot(idx_test[z0:z1], y_test[z0:z1],     color="steelblue", lw=1.5, label="Réel")
    ax_z.plot(idx_test[z0:z1], preds_test[z0:z1], color="tomato",    lw=1.5, label="CNN 1D v2")
    ax_z.axhline(SEUIL_CRUE, color="orange", lw=1, ls="--")
    ax_z.set_title(f"(E) Zoom crue  |  MAE crues={mae_crue:.0f} m³/h", fontsize=11)
else:
    ax_z.text(0.5, 0.5, "Pas assez de crues", ha="center", va="center",
              transform=ax_z.transAxes)
ax_z.set_ylabel("Débit (m³/h)")
ax_z.legend(fontsize=9)

out_path = OUT / "cnn1d_H3_v2_evaluation.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  -> {out_path.name}")

print("\n" + "=" * 65)
print("FIN cnn1d_H3.py  (v2 hybride)")
print("=" * 65)
