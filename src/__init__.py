# src/__init__.py
"""
Paquetes locales
"""

__version__ = '0.1.0'

# Importar funciones principales para acceso directo
from .data_management import get_categorical_number_columns, delete_columns
__all__ = [
    'get_categorical_number_columnsdata',
    'delete_columns',
]