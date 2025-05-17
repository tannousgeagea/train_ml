import requests
import argparse
import os

def download_yolo_zip(api_url, save_dir):
    try:
        print(f"Requesting download URL from: {api_url}")
        response = requests.get(api_url, headers={"accept": "application/json"})
        response.raise_for_status()
        download_url = response.json().get("url")

        if not download_url:
            raise Exception("No download URL returned by the API.")

        filename = os.path.basename(download_url.split("?")[0])
        filepath = os.path.join(save_dir, filename)

        print(f"Downloading from: {download_url}")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"File downloaded successfully: {filepath}")
        return filepath

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YOLO model ZIP from API response")
    parser.add_argument('--api-url', required=True, help='URL of the version download API')
    parser.add_argument('--save-dir', default='.', help='Directory to save the downloaded file')

    args = parser.parse_args()
    download_yolo_zip(args.api_url, args.save_dir)
