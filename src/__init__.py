# src/__init__.py
"""
Paquetes locales
"""

__version__ = '0.1.0'

# Importar funciones principales para acceso directo
from .data_management import get_categorical_number_columns, delete_columns
from .model_trainer import ModelTrainer
__all__ = [
    'get_categorical_number_columns',
    'delete_columns',
    'ModelTrainer'
]