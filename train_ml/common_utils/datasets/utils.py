import os
import yaml
import requests
import argparse
import zipfile
import django
django.setup()
from django.conf import settings
from django.core.files.base import ContentFile
from typing import Literal
from datasets.models import Dataset, get_version_file

API_URL = "http://cvisionops.want:29085"

def download_dataset(version_id:str, save_dir:str, annotation_format:Literal["yolo", "coco"], dataset:Dataset):
    try:
        print(f"Requesting download URL for: {version_id}")
        response = requests.get(
            f"{API_URL}/api/v1/versions/{version_id}/download", 
            params={
                "format": annotation_format
            },
            headers={"accept": "application/json"}
            )
        response.raise_for_status()
        download_url = response.json().get("url")

        if not download_url:
            raise Exception("No download URL returned by the API.")

        filename = os.path.basename(download_url.split("?")[0])
        filepath = os.path.join(save_dir, filename)
    
        if dataset:
            version_file = f"{settings.MEDIA_ROOT}/{get_version_file(dataset, filename)}"
            dataset.version_file = version_file
            dataset.save(update_fields=["version_file"])
            filepath = version_file

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"Downloading from: {download_url}")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"File downloaded successfully: {filepath}")
        return filepath

    except Exception as e:
        raise e

def unzip_file(file: str, extract_to: str = None):
    """
    Unzips a .zip archive to the specified directory (or current directory if not provided).
    
    Args:
        file (str): Path to the .zip file.
        extract_to (str, optional): Directory to extract contents into. Defaults to same directory as zip.
    """
    try:
        extract_to = extract_to or os.path.splitext(file)[0]
        os.makedirs(extract_to, exist_ok=True)

        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        print(f"Extracted to: {extract_to}")
        return extract_to

    except zipfile.BadZipFile:
        raise Exception(f"Invalid ZIP file: {file}")
    except Exception as e:
        raise Exception(f"Failed to extract {file}: {e}")
    

def create_data_yaml(data_dir:str, class_names: list, output_path: str = "data.yaml"):
    """
    Generates a data.yaml file for YOLO training.

    Args:
        train_dir (str): Path to training images directory.
        val_dir (str): Path to validation images directory.
        class_names (list): List of class names.
        output_path (str): Where to save the YAML file (default: "data.yaml").
    
    Returns:
        str: Path to the generated data.yaml file.
    """
    data = {
        "path": os.path.abspath(data_dir),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(class_names),
        "names": class_names
    }

    with open(output_path, "w") as f:
        yaml.dump(data, f)

    print(f"YOLO data.yaml created at: {output_path}")
    return output_path

def prepare_data_yaml(data_dir):
    if not os.path.exists(
        os.path.join(
            data_dir, "data.yaml"
        )
    ):
        raise FileExistsError(f"data.yaml Not Found")
    
    data_yaml = yaml.safe_load(open(f"{data_dir}/data.yaml", "r"))
    data_yaml.update(
        {
            "path": data_dir
        }
    )
    with open(f"{data_dir}/data.yaml", "w") as f:
        yaml.dump(data_yaml, f)
    
    return f"{data_dir}/data.yaml"


def prepare_dataset(version_id:str, save_dir:str, annotation_format:Literal["yolo", "coco"]):
    try:
        dataset = Dataset.objects.filter(dataset_id=version_id).first()
        if dataset and dataset.version_file:
            file_path = dataset.version_file.path
        else:
            file_path = download_dataset(
                version_id=version_id,
                annotation_format=annotation_format,
                save_dir=save_dir,
                dataset=dataset,
            )
                
        dataset_dir = unzip_file(file=file_path)
        data = prepare_data_yaml(dataset_dir)
        return data
    except Exception as e:
        raise e

if __name__ == "__main__":
    # unzip_file(file=file_path)
    MEDIA_ROOT = settings.MEDIA_ROOT
    data = prepare_dataset(
        version_id=102,
        annotation_format="yolo",
        save_dir=f"/media",
        
    )

    print(data)

