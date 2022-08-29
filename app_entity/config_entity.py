from collections import namedtuple

DataIngestionConfig = namedtuple("DatasetConfig", ["dataset_download_url",
                                                   "raw_data_dir",
                                                   "ingested_dir"])

DataValidationConfig = namedtuple("DataValidationConfig", ["clean_data_dir",
                                                         "credit_csv_file",
                                                         "movies_csv_file",
                                                         "serialized_objects_dir"])                            