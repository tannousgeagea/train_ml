import os
import requests

API_URL = "http://10.16.0.8:29085"
SAVE_DIR = "/media/models"

def get_model_weights(model_version_id):
    try:
        response = requests.get(f"{API_URL}/api/v1/model-versions/{model_version_id}", headers={"accept": "application/json"})
        response.raise_for_status()
        download_url = response.json().get("artifacts", {}).get("weights")

        if not download_url:
            raise Exception("No download URL returned by the API.")

        filename = os.path.basename(download_url.split("?")[0])
        filepath = os.path.join(SAVE_DIR, filename)

        print(f"Downloading from: {download_url}")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"File downloaded successfully: {filepath}")
        return filepath
    
    except Exception as e:
        raise Exception(f"Error: {e}")
    

if __name__ == "__main__":

    get_model_weights(
        4
    )