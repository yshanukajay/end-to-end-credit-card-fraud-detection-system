import os
import pandas as pd
import logging
from abc import ABC, abstractmethod

# Retrieve module-level logger without configuring it globally
logger = logging.getLogger(__name__)


class DataIngestor(ABC):
    """
    Abstract Base Class for Data Ingestion. Handles common logging and error wrapper.
    """
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        # 1. Basic Local File Validation
        is_url = file_path_or_link.startswith(('http://', 'https://', 'ftp://'))
        if not is_url and not os.path.exists(file_path_or_link):
            error_msg = f"File not found at specified path: {file_path_or_link}"
            logger.error(f"✗ {error_msg}")
            raise FileNotFoundError(error_msg)

        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - {self._get_format_name()}")
        logger.info(f"{'='*60}")
        logger.info(f"Starting ingestion from: {file_path_or_link}")
        
        try:
            # Delegate format-specific loading to the child class
            df = self._read_data(file_path_or_link)
            
            logger.info(f"✓ Successfully loaded data - Shape: {df.shape}, Columns: {list(df.columns)}")
            logger.info(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            logger.info(f"{'='*60}\n")
            return df
        except Exception as e:
            logger.error(f"✗ Failed to load data: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise

    @abstractmethod
    def _read_data(self, file_path_or_link: str) -> pd.DataFrame:
        """Format-specific reader implementation."""
        pass

    @abstractmethod
    def _get_format_name(self) -> str:
        """Helper to print format name in logs."""
        pass


class DataIngestorCSV(DataIngestor):
    def _read_data(self, file_path_or_link: str) -> pd.DataFrame:
        return pd.read_csv(file_path_or_link)

    def _get_format_name(self) -> str:
        return "CSV"


class DataIngestorExcel(DataIngestor):
    def _read_data(self, file_path_or_link: str) -> pd.DataFrame:
        return pd.read_excel(file_path_or_link)

    def _get_format_name(self) -> str:
        return "EXCEL"


class DataIngestorFactory:
    """
    Factory class to return correct Ingestor based on file extension.
    """
    @staticmethod
    def get_ingestor(file_path_or_link: str) -> DataIngestor:
        ext = os.path.splitext(file_path_or_link)[1].lower()
        if ext == '.csv':
            return DataIngestorCSV()
        elif ext in ['.xlsx', '.xls']:
            return DataIngestorExcel()
        else:
            raise ValueError(f"Unsupported file format: {ext or 'No Extension'}")