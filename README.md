## 📁 Structure du projet

* **`py_ipynb_files/`** : Regroupe l'ensemble des notebooks Jupyter (`.ipynb`) et les scripts Python (`.py`) utilisés pour le traitement, l'agrégation et la visualisation des données.
* **`Dataset/bronze/`** : Répertoire destiné à l'accueil des données brutes.

## 🚀 Installation et Configuration

### 1. Environnement Virtuel
Pour garantir la reproductibilité des analyses, il est fortement recommandé d'utiliser un environnement virtuel. Installez les dépendances requises avec la commande suivante :
```bash
pip install -r requirements.txt
```

### 2. Organisation des données

Avant d'exécuter les notebooks, vous devez structurer vos fichiers de données comme suit :

1. Placez vos fichiers sources dans le dossier `./Dataset/bronze/`.

2. Pour les données météorologiques spécifiques, créez des sous-dossiers respectant la nomenclature suivante :
    • `./Dataset/bronze/weather_{nom_de_la_station}/`

    • Note : `{nom_de_la_station}` doit être remplacé par **C** ou **P**.

Exemple de structure :

.
├── Dataset/ <br>
│   └── bronze/ <br>
│       ├── weather_C/ <br>
│       │   └── Aiglemont_hourly.csv <br>
│       └── weather_P/ <br>
└── py_ipynb_files/ <br>
    └── Data_Analysis.ipynb <br>