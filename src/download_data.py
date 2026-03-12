"""
Skript zum Herunterladen des Titanic-Datasets
"""

import os
import requests


def download_titanic_data():
    """
    Lädt das Titanic-Dataset herunter.
    
    Hinweis: Das Kaggle-Dataset erfordert eine API-Authentifizierung.
    Alternative: Verwende einen direkten Link oder lade manuell herunter.
    """
    
    # Beispiel-URLs (anpassen je nach Quelle)
    urls = {
        'train': 'URL_ZUM_TRAINING_SET',
        'test': 'URL_ZUM_TEST_SET'
    }
    
    output_dir = '../data/raw'
    os.makedirs(output_dir, exist_ok=True)
    
    for name, url in urls.items():
        output_path = os.path.join(output_dir, f'{name}.csv')
        print(f'Lade {name}.csv herunter...')
        
        # Hier Code zum Herunterladen einfügen
        # response = requests.get(url)
        # with open(output_path, 'wb') as f:
        #     f.write(response.content)
        
        print(f'{name}.csv gespeichert unter {output_path}')


if __name__ == '__main__':
    print("Titanic-Dataset Download")
    print("Bitte passe die URLs in diesem Skript an oder lade die Daten manuell herunter.")
    # download_titanic_data()
