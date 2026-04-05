import os
import logging
from logging.handlers import RotatingFileHandler
from src.config import settings

class ColorFormatter(logging.Formatter):
    """Añade colores ANSI a la consola para mayor observabilidad MLOps."""
    COLORS = {
        logging.DEBUG: "\x1b[38;20m",
        logging.INFO: "\x1b[36;20m", # Cyan
        logging.WARNING: "\x1b[33;20m", # Yellow
        logging.ERROR: "\x1b[31;20m", # Red
        logging.CRITICAL: "\x1b[31;1m" # Bold Red
    }
    RESET = "\x1b[0m"
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    def format(self, record):
        log_fmt = self.COLORS.get(record.levelno, self.RESET) + self.FORMAT + self.RESET
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name: str) -> logging.Logger:
    """Configura y retorna un logger estandarizado con RotatingFile y ConsoleColored."""
    os.makedirs(settings.LOGS_PATH, exist_ok=True)
    
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Logfile con rotación para evitar saturar el disco (Max 5MB x 3 backups)
        log_file_path = os.path.join(settings.LOGS_PATH, 'pipeline.log')
        fh = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
        fh.setFormatter(file_fmt)
        
        # Console con Colores
        ch = logging.StreamHandler()
        ch.setFormatter(ColorFormatter())
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.propagate = False
        
    return logger
