# ma_lab/core/logging_config.py
import logging
import logging.handlers
from pathlib import Path

def setup_logging(name: str = "ma_lab"):
    """Configura logging para produção"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Rotate logs
    handler = logging.handlers.RotatingFileHandler(
        f"logs/{name}.log",
        maxBytes=10_485_760,  # 10MB
        backupCount=5
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Também loga no console em desenvolvimento
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.addHandler(console)
    
    return logger