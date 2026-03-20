import pandas as pd
import numpy as np
import os

def load_and_average_weather_data(cities_list, base_path):
    """
    Charge les données de pluie pour plusieurs villes et retourne la moyenne horaire.
    """
    features_to_keep = ['date', 'rain']
    list_df_cities = []
    
    for city in cities_list:
        file_path = f"{base_path}/{city}_hourly.csv"
        
        if os.path.exists(file_path):
            df_city = pd.read_csv(file_path, usecols=features_to_keep)
            df_city['date'] = pd.to_datetime(df_city['date'])
            df_city.set_index('date', inplace=True)
            list_df_cities.append(df_city)
        else:
            print(f"⚠️ Fichier introuvable : {file_path}")
            
    if not list_df_cities:
        raise ValueError("Aucune donnée météo trouvée pour les villes spécifiées.")

    # Concaténation et calcul de la moyenne par heure sur toutes les villes
    df_weather_mean = pd.concat(list_df_cities).groupby(level=0).mean()
    return df_weather_mean

def compute_rain_lag(df, lag_hours):
    """
    Applique un décalage temporel (lag) sur la colonne de pluie.
    """
    df_lagged = df.copy()
    new_col_name = f'rain_lag_{lag_hours}h'
    
    df_lagged[new_col_name] = df_lagged['rain'].shift(lag_hours)
    
    # Remplissage des premières valeurs manquantes par la première valeur connue
    first_value = df_lagged['rain'].iloc[0]
    df_lagged[new_col_name] = df_lagged[new_col_name].fillna(first_value)
    
    df_lagged = df_lagged.drop(columns=['rain'])
    return df_lagged

def build_engineered_features(df_exog, rain_col_name, threshold=2.0):
    """
    Crée de nouvelles variables (Feature Engineering) : transformation exponentielle et flag.
    """
    df_engineered = pd.DataFrame(index=df_exog.index)
    
    # Transformation mathématique (expm1)
    df_engineered['rain_lag_transformed'] = np.expm1(df_exog[rain_col_name])
    
    # Création d'un flag binaire pour les fortes pluies
    df_engineered['rain_flag'] = (df_exog[rain_col_name] >= threshold).astype(int)
    
    return df_engineered

def create_gold_dataset_station_c(sensor_file_path, weather_base_path, cities_list, output_gold_path, lag_hours=4, rain_threshold=2.0):
    print("--- Début du Preprocessing (Station C - Modèle 5 SARIMAX avec FE) ---")
    
    # ==========================================
    # 1. TRAITEMENT DES DONNÉES CAPTEURS (Endogène)
    # ==========================================
    print("1. Chargement et traitement des données de débit (Station C)...")
    df_sensor = pd.read_csv(sensor_file_path)
    df_sensor['ts'] = pd.to_datetime(df_sensor['ts'])
    df_sensor.set_index('ts', inplace=True)
    
    series_debit = df_sensor.loc[df_sensor['sensor'] == 'entry_debit_f1', 'value']
    df_endo = series_debit.resample('1h').mean().ffill().to_frame(name='debit_entrant')
    
    print(f"   Plage de données : du {df_endo.index.min()} au {df_endo.index.max()}")
    
    # ==========================================
    # 2. TRAITEMENT DES DONNÉES MÉTÉO (Exogène + FE)
    # ==========================================
    print(f"2. Chargement de la météo pour {len(cities_list)} villes et calcul de la moyenne...")
    df_weather_mean = load_and_average_weather_data(cities_list, weather_base_path)
    
    print(f"   Application du décalage (lag) de {lag_hours} heures...")
    df_exo_lagged = compute_rain_lag(df_weather_mean, lag_hours=lag_hours)
    
    # Alignement avant le Feature Engineering pour éviter les décalages d'index
    df_exo_aligned = df_exo_lagged.reindex(df_endo.index).ffill().bfill()
    
    print("   Application du Feature Engineering (expm1 et rain_flag)...")
    df_exo_engineered = build_engineered_features(
        df_exog=df_exo_aligned, 
        rain_col_name=f'rain_lag_{lag_hours}h', 
        threshold=rain_threshold
    )
    
    # ==========================================
    # 3. FUSION ET ALIGNEMENT FINAL
    # ==========================================
    print("3. Fusion des datasets endogènes et exogènes...")
    df_gold_sarimax = df_endo.join(df_exo_engineered)
    
    # ==========================================
    # 4. SAUVEGARDE
    # ==========================================
    print(f"4. Sauvegarde du fichier Gold vers : {output_gold_path}")
    os.makedirs(os.path.dirname(output_gold_path), exist_ok=True)
    df_gold_sarimax.to_csv(output_gold_path)
    
    print(f"   Dimensions du fichier final : {df_gold_sarimax.shape}")
    print("--- Preprocessing terminé avec succès ! ---")
    
    return df_gold_sarimax

# ==========================================
# EXÉCUTION DU SCRIPT
# ==========================================
if __name__ == "__main__":
    # Paramètres de configuration
    CHEMIN_BRONZE_C = 'Dataset/bronze/sensors_C_20250101_20260108.csv'
    CHEMIN_METEO_DIR_C = 'Dataset/bronze/weather_C_latest'
    CHEMIN_GOLD_C = 'Dataset/gold/station_C_sarimax_ready.csv'
    
    VILLES_C = [
        'Aiglemont', 'Chalandry-Elaire', 'Damouzy', 'Dom-le-Mesnil', 
        'Flize', 'La-Francheville', 'La-Grandville', 'Les-Ayvelles', 
        'Montcy-Notre-Dame', 'Prix-les-Mezieres', 'Saint-Laurent', 
        'Ville-sur-Lumes', 'Villers-Semeuse', 'Warcq'
    ]
    
    LAG = 4
    THRESHOLD = 2.0
    
    # Lancement
    df_final = create_gold_dataset_station_c(
        sensor_file_path=CHEMIN_BRONZE_C, 
        weather_base_path=CHEMIN_METEO_DIR_C, 
        cities_list=VILLES_C, 
        output_gold_path=CHEMIN_GOLD_C,
        lag_hours=LAG,
        rain_threshold=THRESHOLD
    )
    
    print("\nAperçu des données Gold prêtes pour SARIMAX :")
    print(df_final.head())