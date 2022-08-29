import os
import sys
from app_logger.logger import logging
from app_utils.utils import read_yaml_file
from app_exception.exception_handler import AppException
from app_entity.config_entity import DataIngestionConfig, DataValidationConfig

ROOT_DIR = os.getcwd()

CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_FILE_NAME)


class AppConfiguration:
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH):
        try:
            self.configs_info = read_yaml_file(file_path=config_file_path)
        except Exception as e:
            raise AppException(e, sys) from e

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_ingestion_config = self.configs_info['data_ingestion_config']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']
            dataset_dir = data_ingestion_config['dataset_dir']

            ingested_data_dir = os.path.join(artifacts_dir, dataset_dir, data_ingestion_config['ingested_dir'])
            raw_data_dir = os.path.join(artifacts_dir, dataset_dir, data_ingestion_config['raw_data_dir'])

            response = DataIngestionConfig(
                dataset_download_url = data_ingestion_config['dataset_download_url'],
                raw_data_dir = raw_data_dir,
                ingested_dir = ingested_data_dir
            )

            logging.info(f"Data Ingestion Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e

    

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_validation_config = self.configs_info['data_validation_config']
            data_ingestion_config = self.configs_info['data_ingestion_config']
            dataset_dir = data_ingestion_config['dataset_dir']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']
            credit_csv_file = data_validation_config['credit_csv_file']
            movies_csv_file = data_validation_config['movies_csv_file']

            credit_csv_file_dir = os.path.join(artifacts_dir, dataset_dir, data_ingestion_config['ingested_dir'], credit_csv_file)
            movies_csv_file_dir = os.path.join(artifacts_dir, dataset_dir, data_ingestion_config['ingested_dir'], movies_csv_file)
            clean_data_path = os.path.join(artifacts_dir, dataset_dir, data_validation_config['clean_data_dir'])
            serialized_objects_dir = os.path.join(artifacts_dir, data_validation_config['serialized_objects_dir'])

            response = DataValidationConfig(
                clean_data_dir = clean_data_path,
                credit_csv_file = credit_csv_file_dir,
                movies_csv_file = movies_csv_file_dir,
                serialized_objects_dir = serialized_objects_dir
            )

            logging.info(f"Data Validation Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e