import pandas as pd
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

def compute_rain_lag(df, lag_hours=1):
    """
    Applique un décalage temporel (lag) sur la colonne de pluie.
    """
    df_lagged = df.copy()
    new_col_name = f'rain_lag_{lag_hours}h'
    
    # Création du décalage temporel
    df_lagged[new_col_name] = df_lagged['rain'].shift(lag_hours)
    
    # Remplissage des 'lag_hours' premières valeurs par la toute première valeur de 'rain'
    first_value = df_lagged['rain'].iloc[0]
    df_lagged[new_col_name] = df_lagged[new_col_name].fillna(first_value)
    
    # Suppression de la colonne 'rain' initiale (non laggée)
    df_lagged = df_lagged.drop(columns=['rain'])
    
    return df_lagged

def create_gold_dataset_station_p(sensor_file_path, weather_base_path, cities_list, output_gold_path, start_date='2025-05-01', lag=1):
    print("--- Début du Preprocessing (Station P - Modèle 4 SARIMAX) ---")
    
    # ==========================================
    # 1. TRAITEMENT DES DONNÉES CAPTEURS (Endogène)
    # ==========================================
    print("1. Chargement et traitement des données de débit...")
    df_sensor = pd.read_csv(sensor_file_path)
    df_sensor['ts'] = pd.to_datetime(df_sensor['ts'])
    df_sensor.set_index('ts', inplace=True)
    
    # Filtrage du capteur et rééchantillonnage horaire
    series_debit = df_sensor.loc[df_sensor['sensor'] == 'entry_debit_f1', 'value']
    df_endo = series_debit.resample('1h').mean().ffill().to_frame(name='debit_entrant')
    
    # Filtrage à partir du 1er mai 2025 (Période des données ajustées)
    df_endo = df_endo.loc[df_endo.index >= start_date]
    
    # ==========================================
    # 2. TRAITEMENT DES DONNÉES MÉTÉO (Exogène)
    # ==========================================
    print(f"2. Chargement de la météo pour {len(cities_list)} villes et calcul de la moyenne...")
    df_weather_mean = load_and_average_weather_data(cities_list, weather_base_path)
    
    print("   Application du décalage (lag) de 1 heure...")
    df_exo = compute_rain_lag(df_weather_mean, lag_hours=lag)
    
    # ==========================================
    # 3. FUSION ET ALIGNEMENT FINAL
    # ==========================================
    print("3. Alignement temporel des données exogènes sur les données endogènes...")
    # On réindexe la météo pour qu'elle matche exactement les dates du débit filtré
    df_exo_aligned = df_exo.reindex(df_endo.index).ffill().bfill()
    
    # Fusion finale (Création du dataset Gold unique)
    df_gold_sarimax = df_endo.join(df_exo_aligned)
    
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
    CHEMIN_BRONZE_P = 'Dataset/bronze/sensor_P_20231110_20260106.csv'
    CHEMIN_METEO_DIR = 'Dataset/bronze/weather_P_latest'
    CHEMIN_GOLD_P = 'Dataset/gold/station_P_sarimax_ready.csv'
    
    VILLES_P = ['Beaumont-sur-Oise', 'Bernes-sur-Oise', 'Chambly', 'Mours', 'Nointel', 'Persan-Beaumont', 'Ronquerolles']
    DATE_DEBUT_ANALYSE = '2025-05-01'

    LAG = 1
    
    # Lancement
    df_final = create_gold_dataset_station_p(
        sensor_file_path=CHEMIN_BRONZE_P, 
        weather_base_path=CHEMIN_METEO_DIR, 
        cities_list=VILLES_P, 
        output_gold_path=CHEMIN_GOLD_P,
        start_date=DATE_DEBUT_ANALYSE,
        lag=LAG
    )
    
    print("\nAperçu des données Gold prêtes pour SARIMAX :")
    print(df_final.head())