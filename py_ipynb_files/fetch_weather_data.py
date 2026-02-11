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
        {"name": "Aiglemont", "lat": 49.79022, "lon": 4.770087},
        {"name": "Chalandry-Elaire", "lat": 49.707747, "lon": 4.756461},
        {"name": "Damouzy", "lat": 49.807862, "lon": 4.682754},
        {"name": "Dom-le-Mesnil", "lat": 49.685858, "lon": 4.809973},
        {"name": "Flize", "lat": 49.695297, "lon": 4.775173},
        {"name": "La-Francheville", "lat": 49.7333, "lon": 4.7167},
        {"name": "La-Grandville", "lat": 49.783, "lon": 4.795},
        {"name": "Les-Ayvelles", "lat": 49.7, "lon": 4.7667},
        {"name": "Montcy-Notre-Dame", "lat": 49.790675, "lon": 4.743459},
        {"name": "Prix-les-Mezieres", "lat": 49.756, "lon": 4.687},
        {"name": "Saint-Laurent", "lat": 49.759138, "lon": 4.776074},
        {"name": "Ville-sur-Lumes", "lat": 49.755706, "lon": 4.79606849833},
        {"name": "Villers-Semeuse", "lat": 49.741144, "lon": 4.751054},
        {"name": "Warcq", "lat": 49.768287, "lon": 4.666435}
    ]
elif station_choice == "station_P":
    T_START = "2023-11-10" 
    T_END = "2026-01-06"
    OUTPUT_PATH = "../Dataset/bronze/weather_P_latest/"
    CITIES = [
        {"name": "Beaumont-sur-Oise", "lat": 49.140098, "lon": 2.300213},
        {"name": "Bernes-sur-Oise", "lat": 49.165413, "lon": 2.301276},
        {"name": "Chambly", "lat": 49.17181, "lon": 2.24657},
        {"name": "Mours", "lat": 49.130395, "lon": 2.264151},
        {"name": "Nointel", "lat": 49.128869, "lon": 2.294343},
        {"name": "Persan-Beaumont", "lat": 49.150002, "lon": 2.26667},
        {"name": "Ronquerolles", "lat": 49.167704, "lon": 2.213417}
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