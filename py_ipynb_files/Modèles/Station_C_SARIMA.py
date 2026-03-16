import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from itertools import product
import time

# Ignorer les warnings (convergence, etc.)
warnings.filterwarnings("ignore")

# --- 1. Chargement et Préparation ---
file_path = './Dataset/bronze/sensors_C_20250101_20260108.csv'

print("Chargement des données...")
df = pd.read_csv(file_path)

# Conversion de la date
date_col = 'ts'
df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

# Sélection du signal unique
series = df.loc[df['sensor'] == 'entry_debit', 'value']

# Rééchantillonnage horaire (Moyenne)
series_resampled = series.resample('1h').mean().ffill()

# Split Train / Test
# On garde 3 mois d'historique pour l'entraînement pour capter les dynamiques sans être trop lourd
three_months_ago = series_resampled.index.max() - pd.DateOffset(months=3)
series_recent = series_resampled[series_resampled.index >= three_months_ago]

test_size = 24 * 14 # Test sur 2 semaines (336 heures)
train = series_recent[:-test_size]
test = series_recent[-test_size:]

print(f"Train size: {len(train)} heures")
print(f"Test size: {len(test)} heures")

# --- 2. Grid Search SARIMAX (Saisonnalité Journalière) ---

# Définition de la grille
# Attention : SARIMAX avec s=24 est très lent. On limite l'espace de recherche.
# Ordre de tendance (Trend)
p_values = [1, 2]    # Auto-régressif court terme
d_values = [0, 1]    # Différenciation simple
q_values = [1, 2]    # Moyenne mobile

# Ordre saisonnier (Seasonal) - s=24 pour le cycle journalier
P_values = [0, 1]    # Auto-régressif saisonnier (ex: lien avec hier à la même heure)
D_values = [1]       # Différenciation saisonnière (quasi obligatoire pour le cycle 24h)
Q_values = [0, 1]    # Moyenne mobile saisonnière
s_value = 24         # Période de 24h

best_score = float("inf")
best_cfg = None
best_model = None

print(f"\nDémarrage de la Grid Search SARIMAX (s={s_value})...")
print("Cela peut prendre du temps en fonction de la puissance de calcul.")

# Création de toutes les combinaisons
combinations = list(product(p_values, d_values, q_values, P_values, D_values, Q_values))
total_combos = len(combinations)

start_time = time.time()

for i, (p, d, q, P, D, Q) in enumerate(combinations):
    order = (p, d, q)
    seasonal_order = (P, D, Q, s_value)
    
    try:
        # Affichage de progression
        if i % 5 == 0:
            print(f"Test combinaison {i}/{total_combos} : {order} x {seasonal_order}...")

        model = SARIMAX(train, 
                        order=order, 
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False, 
                        enforce_invertibility=False)
        
        # disp=False pour ne pas polluer la console
        model_fit = model.fit(disp=False, maxiter=50) 
        
        # Prédiction
        forecast = model_fit.get_forecast(steps=len(test))
        predictions = forecast.predicted_mean
        
        # Évaluation RMSE
        rmse = np.sqrt(mean_squared_error(test, predictions))
        
        if rmse < best_score:
            best_score = rmse
            best_cfg = (order, seasonal_order)
            best_model = model_fit
            print(f"--> NOUVEAU RECORD : RMSE={rmse:.3f} avec {order}x{seasonal_order}")

    except Exception as e:
        continue

elapsed_time = (time.time() - start_time) / 60
print(f"\nTerminé en {elapsed_time:.1f} minutes.")
print(f'Meilleur Modèle : SARIMAX{best_cfg} RMSE={best_score:.3f}')

# --- 3. Visualisation ---
final_forecast = best_model.get_forecast(steps=len(test))
final_predictions = final_forecast.predicted_mean
conf_int = final_forecast.conf_int()

plt.figure(figsize=(15, 6))

# On affiche la fin du train
plt.plot(train.index[-168:], train[-168:], label='Train (Dernière semaine)', color='gray', alpha=0.5)

# La réalité
plt.plot(test.index, test, label='Réalité (Test)', color='blue', linewidth=1.5)

# La prédiction
plt.plot(test.index, final_predictions, label='Prédiction SARIMAX', color='red', linestyle='--', linewidth=1.5)

# Intervalle de confiance (optionnel mais utile)
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.1)

plt.title(f"Prédiction Débit Entrant (Station C) - Best SARIMAX {best_cfg[0]}x{best_cfg[1]}")
plt.ylabel("Débit (m3/h)")
plt.xlabel("Date")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# --- 4. Métriques ---
mae = mean_absolute_error(test, final_predictions)
print("\n--- Résultats Finaux ---")
print(f"Meilleurs Paramètres : Order={best_cfg[0]}, Seasonal={best_cfg[1]}")
print(f"RMSE : {best_score:.2f} m3/h")
print(f"MAE  : {mae:.2f} m3/h")