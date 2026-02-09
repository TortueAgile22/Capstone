import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os
import sys

# --- 1. GESTION DES ARGUMENTS ---

if len(sys.argv) < 2:
    print("❌ Erreur : Argument manquant.")
    print("Usage : python fetch_weather_data.py [station_C|station_P]")
    sys.exit(1)

station_choice = sys.argv[1]

# Configuration dynamique selon l'argument
if station_choice == "station_C":
    T_START = "2025-01-01"
    T_END = "2026-01-08"
    OUTPUT_PATH = "../Dataset/bronze/weather_C_latest/"
    CITIES = [
        {"name": "Aiglemont", "lat": 49.7803, "lon": 4.7648},
        {"name": "Chalandry-Elaire", "lat": 49.7128, "lon": 4.7478},
        {"name": "Damouzy", "lat": 49.7981, "lon": 4.6739},
        {"name": "Dom-le-Mesnil", "lat": 49.6917, "lon": 4.8050},
        {"name": "Flize", "lat": 49.6983, "lon": 4.7739},
        {"name": "La-Francheville", "lat": 49.7289, "lon": 4.7144},
        {"name": "La-Grandville", "lat": 49.7806, "lon": 4.7933},
        {"name": "Les-Ayvelles", "lat": 49.7153, "lon": 4.7578},
        {"name": "Montcy-Notre-Dame", "lat": 49.7758, "lon": 4.7431},
        {"name": "Prix-les-Mezieres", "lat": 49.7547, "lon": 4.6853},
        {"name": "Saint-Laurent", "lat": 49.7644, "lon": 4.7631},
        {"name": "Ville-sur-Lumes", "lat": 49.7511, "lon": 4.7906},
        {"name": "Villers-Semeuse", "lat": 49.7394, "lon": 4.7514},
        {"name": "Warcq", "lat": 49.7700, "lon": 4.6800}
    ]
elif station_choice == "station_P":
    T_START = "2023-11-10" 
    T_END = "2026-01-06"
    OUTPUT_PATH = "../Dataset/bronze/weather_P_latest/"
    CITIES = [
        {"name": "Beaumont-sur-Oise", "lat": 49.1431, "lon": 2.2856},
        {"name": "Bernes-sur-Oise", "lat": 49.1622, "lon": 2.3006},
        {"name": "Chambly", "lat": 49.1658, "lon": 2.2472},
        {"name": "Mours", "lat": 49.1347, "lon": 2.2619},
        {"name": "Nointel", "lat": 49.1286, "lon": 2.2922},
        {"name": "Persan-Beaumont", "lat": 49.1539, "lon": 2.2706},
        {"name": "Ronquerolles", "lat": 49.1658, "lon": 2.2153}
    ]
else:
    print(f"❌ Erreur : '{station_choice}' n'est pas un argument valide.")
    print("Veuillez choisir entre 'station_C' ou 'station_P'.")
    sys.exit(1)

# --- 2. INITIALISATION ---

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Création du dossier de sortie absolu pour éviter les erreurs de chemin
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_PATH)
os.makedirs(FULL_OUTPUT_PATH, exist_ok=True)

url = "https://archive-api.open-meteo.com/v1/archive"

# --- 3. BOUCLE DE RÉCUPÉRATION ---

print(f"🚀 Lancement du scraping pour {station_choice}")
print(f"📂 Destination : {FULL_OUTPUT_PATH}\n")

for city_info in CITIES:
    city_name = city_info["name"]
    print(f"📥 Récupération : {city_name}...")

    params = {
        "latitude": city_info["lat"],
        "longitude": city_info["lon"],
        "start_date": T_START,
        "end_date": T_END,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "rain", "weather_code", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m"]
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        
        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}

        # Extraction dynamique des colonnes
        for i, col in enumerate(params["hourly"]):
            hourly_data[col] = hourly.Variables(i).ValuesAsNumpy()

        df = pd.DataFrame(data = hourly_data)
        df["latitude"] = response.Latitude()
        df["longitude"] = response.Longitude()
        df["city"] = city_name
        df["date"] = df["date"].dt.strftime('%Y-%m-%d %H:%M:%S')

        file_path = os.path.join(FULL_OUTPUT_PATH, f"{city_name}_hourly.csv")
        df.to_csv(file_path, index=False)
        print(f"   ✅ Sauvegardé.")

    except Exception as e:
        print(f"   ⚠️ Erreur pour {city_name} : {e}")

print(f"\n✨ Terminé pour {station_choice}.")