import os
import zipfile
import logging
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_isot_fake_news_dataset():
    """
    自动化下载ISOT假新闻数据集，包含环境校验、解压、文件验证
    """
    # 1. 加载环境变量（Kaggle密钥）
    load_dotenv()
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')
    
    # 校验密钥是否存在
    if not kaggle_username or not kaggle_key:
        logging.error("未配置Kaggle密钥！请在.env文件中设置KAGGLE_USERNAME和KAGGLE_KEY")
        return False
    
    # 2. 初始化Kaggle API
    api = KaggleApi()
    try:
        api.authenticate()
        logging.info("Kaggle API认证成功")
    except Exception as e:
        logging.error(f"Kaggle API认证失败：{str(e)}")
        return False
    
    # 3. 定义数据集信息
    dataset_owner = "emineyetm"
    dataset_name = "fake-news-detection-datasets"
    download_dir = "./data"  # 数据集保存目录
    zip_file_path = f"{download_dir}/{dataset_name}.zip"
    
    # 4. 创建保存目录
    os.makedirs(download_dir, exist_ok=True)
    
    # 5. 下载数据集
    try:
        logging.info(f"开始下载ISOT假新闻数据集：{dataset_owner}/{dataset_name}")
        api.dataset_download_files(
            f"{dataset_owner}/{dataset_name}",
            path=download_dir,
            unzip=False  # 先下载压缩包，手动解压便于校验
        )
        logging.info(f"数据集压缩包已下载至：{zip_file_path}")
    except Exception as e:
        logging.error(f"数据集下载失败：{str(e)}")
        return False
    
    # 6. 解压数据集
    try:
        logging.info("开始解压数据集...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # 显示解压进度
            for file in tqdm(zip_ref.infolist(), desc="解压中"):
                zip_ref.extract(file, download_dir)
        logging.info("数据集解压完成")
        
        # 删除压缩包（可选）
        os.remove(zip_file_path)
        logging.info("已删除压缩包，节省存储空间")
    except Exception as e:
        logging.error(f"数据集解压失败：{str(e)}")
        return False
    
    # 7. 验证文件是否完整
    expected_files = ["True.csv", "Fake.csv"]
    missing_files = []
    for file in expected_files:
        file_path = os.path.join(download_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        logging.error(f"数据集文件缺失：{missing_files}")
        return False
    else:
        logging.info("数据集文件验证通过，包含True.csv和Fake.csv")
        
        # 打印基础信息
        import pandas as pd
        true_df = pd.read_csv(os.path.join(download_dir, "True.csv"))
        fake_df = pd.read_csv(os.path.join(download_dir, "Fake.csv"))
        logging.info(f"真实新闻样本数：{len(true_df)}")
        logging.info(f"假新闻样本数：{len(fake_df)}")
        logging.info("数据集下载与验证完成！")
        return True

if __name__ == "__main__":
    download_isot_fake_news_dataset()