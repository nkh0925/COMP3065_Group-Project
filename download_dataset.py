import os
import zipfile
import logging
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_isot_fake_news_dataset():
    """
    Automated downloader for the ISOT Fake News dataset, including:
        - Environment check (Kaggle API keys)
        - Dataset download
        - Unzip with progress
        - File verification
    """
    # 1. Load environment variables (Kaggle credentials)
    load_dotenv()
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')
    
    # Check if credentials exist
    if not kaggle_username or not kaggle_key:
        logging.error("Kaggle credentials not found! Please set KAGGLE_USERNAME and KAGGLE_KEY in a .env file")
        return False
    
    # 2. Initialize Kaggle API
    api = KaggleApi()
    try:
        api.authenticate()
        logging.info("Kaggle API authentication successful")
    except Exception as e:
        logging.error(f"Kaggle API authentication failed: {str(e)}")
        return False
    
    # 3. Define dataset information
    dataset_owner = "emineyetm"
    dataset_name = "fake-news-detection-datasets"
    download_dir = "./data"  # dataset save directory
    zip_file_path = f"{download_dir}/{dataset_name}.zip"
    
    # 4. Create directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # 5. Download dataset
    try:
        logging.info(f"Starting download of ISOT Fake News dataset: {dataset_owner}/{dataset_name}")
        api.dataset_download_files(
            f"{dataset_owner}/{dataset_name}",
            path=download_dir,
            unzip=False  # download as zip first, manual unzip for verification
        )
        logging.info(f"Dataset zip downloaded to: {zip_file_path}")
    except Exception as e:
        logging.error(f"Dataset download failed: {str(e)}")
        return False
    
    # 6. Unzip dataset
    try:
        logging.info("Unzipping dataset...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Show extraction progress
            for file in tqdm(zip_ref.infolist(), desc="Extracting"):
                zip_ref.extract(file, download_dir)
        logging.info("Dataset extraction complete")
        
        # Remove zip file to save space (optional)
        os.remove(zip_file_path)
        logging.info("Zip file deleted to save storage space")
    except Exception as e:
        logging.error(f"Dataset extraction failed: {str(e)}")
        return False
    
    # 7. Verify files
    expected_files = ["True.csv", "Fake.csv"]
    missing_files = []
    for file in expected_files:
        file_path = os.path.join(download_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        logging.error(f"Missing dataset files: {missing_files}")
        return False
    else:
        logging.info("Dataset verification passed: True.csv and Fake.csv found")
        
        # Print basic dataset info
        import pandas as pd
        true_df = pd.read_csv(os.path.join(download_dir, "True.csv"))
        fake_df = pd.read_csv(os.path.join(download_dir, "Fake.csv"))
        logging.info(f"Number of true news samples: {len(true_df)}")
        logging.info(f"Number of fake news samples: {len(fake_df)}")
        logging.info("Dataset download and verification complete!")
        return True


if __name__ == "__main__":
    download_isot_fake_news_dataset()