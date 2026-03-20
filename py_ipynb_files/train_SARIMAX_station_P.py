import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. FONCTION DE CHARGEMENT ET SPLIT
# ==========================================
def load_and_split_gold_data(file_path):
    print("1. Chargement du dataset Gold...")
    # Chargement avec indexation temporelle
    df = pd.read_csv(file_path)
    time_col = 'ts' if 'ts' in df.columns else 'date' if 'date' in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)
    
    df = df.asfreq('h') 
    
    print("2. Découpage Train / Test (Test = 2 dernières semaines)...")
    # 2 semaines = 2 * 7 * 24 = 336 heures
    test_size = 336
    
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    
    # Séparation Endogène (Débit) / Exogène (Pluie laggée)
    train_endog = train_df['debit_entrant']
    train_exog = train_df[['rain_lag_1h']]
    
    test_endog = test_df['debit_entrant']
    test_exog = test_df[['rain_lag_1h']]
    
    print(f"   Taille du Train : {len(train_endog)} heures")
    print(f"   Taille du Test  : {len(test_endog)} heures")
    
    return train_endog, test_endog, train_exog, test_exog

# ==========================================
# 2. FONCTION D'ENTRAÎNEMENT ET D'ÉVALUATION
# ==========================================
def train_evaluate_sarimax(train_endog, test_endog, train_exog, test_exog, show_plots=True):
    print("\n3. Lancement de l'entraînement SARIMAX - Station P...")
    
    # --- Modélisation ---
    model = SARIMAX(
        endog=train_endog,
        exog=train_exog,
        order=(0, 1, 2),
        seasonal_order=(0, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    model_fit = model.fit(disp=False)
    print("   Entraînement terminé !")
    
    # --- Prédictions ---
    print("4. Génération des prédictions sur le jeu de test...")
    forecast = model_fit.get_forecast(steps=len(test_endog), exog=test_exog)
    preds = forecast.predicted_mean.values
    
    # --- Métriques ---
    rmse = np.sqrt(mean_squared_error(test_endog.values, preds))
    mae = mean_absolute_error(test_endog.values, preds)
    r2 = r2_score(test_endog.values, preds)
    
    print("\n" + "="*50)
    print(f"RÉSULTATS SARIMAX (Modèle 4 - Station P)")
    print("="*50)
    print(f"RMSE : {rmse:.2f} m3/h")
    print(f"MAE  : {mae:.2f} m3/h")
    print(f"R²   : {r2:.4f}")
    
    # --- DataFrame de résultats ---
    df_results = pd.DataFrame({
        'Date': test_endog.index,
        'Réel': test_endog.values,
        'Prédiction': preds,
        'Pluie_Exog': test_exog['rain_lag_1h'].values 
    }).set_index('Date')
    
    # --- Visualisation ---
    if show_plots:
        print("\n5. Génération des graphiques...")
        
        # 1. Graphique Principal : Débit Réel vs Prédiction (+ Pluie)
        fig, ax1 = plt.subplots(figsize=(18, 6))

        ax1.plot(df_results.index, df_results['Réel'], label='Débit Réel', color='blue', alpha=0.6, linewidth=1)
        ax1.plot(df_results.index, df_results['Prédiction'], label='Prédiction SARIMAX', color='orange', alpha=0.7, linewidth=1, linestyle='--')
        ax1.set_xlabel('Date', fontweight='bold')
        ax1.set_ylabel('Débit Entrant (m3/h)', color='blue', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')
        ax1.grid(True, which='both', linestyle='--', alpha=0.7) 

        # Axe secondaire pour la pluie
        ax2 = ax1.twinx()
        ax2.bar(df_results.index, df_results['Pluie_Exog'], alpha=0.3, color='cyan', width=0.03, label='Pluie (lag 1h)')
        ax2.set_ylabel('Pluie laggée (mm)', color='cyan', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='cyan')
        ax2.legend(loc='upper right')   

        plt.title(f"Performance SARIMAX (Station P) - Zoom 2 semaines de Test\nRMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}", fontsize=14, pad=15)
        plt.tight_layout()
        plt.show()

        # 2. Graphique des Résidus
        plt.figure(figsize=(18, 4))
        residus = df_results['Réel'] - df_results['Prédiction']
        plt.plot(df_results.index, residus, color='green', alpha=0.7)
        plt.axhline(0, color='black', linestyle='--')
        plt.title("Analyse des Résidus (Réel - Prédiction)", pad=10)
        plt.ylabel("Erreur (m3/h)", fontweight='bold')
        plt.xlabel("Date", fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
    return model_fit, df_results

# ==========================================
# 3. EXÉCUTION DU SCRIPT
# ==========================================
if __name__ == "__main__":
    # Chemin vers le dataset Gold généré à l'étape précédente
    CHEMIN_GOLD_P = 'Dataset/gold/station_P_sarimax_ready.csv'
    
    # 1. Chargement et découpage
    train_endog, test_endog, train_exog, test_exog = load_and_split_gold_data(CHEMIN_GOLD_P)
    
    # 2. Entraînement et évaluation (Mets show_plots=False pour ne pas afficher les graphiques)
    modele_final, resultats = train_evaluate_sarimax(
        train_endog=train_endog, 
        test_endog=test_endog, 
        train_exog=train_exog, 
        test_exog=test_exog, 
        show_plots=True
    )