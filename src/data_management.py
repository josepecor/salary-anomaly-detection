
import pandas as pd
import numpy as np
from typing import List


def get_categorical_number_columns() -> List[str]:
    """
    Retorna las columnas categóricas que sin interpretadas como números del dataset IBM HR Analytics.
    
    Returns:
        List[str]: Lista con los nombres de las columnas categóricas.
    """
    categorical_columns = [
        'Education',
        'EnviromentSatisfaction', 
        'JobInvolvement',
        'JobSatisfaction',
        'PerformanceRating',
        'RelationshipSatisfaction',
        'WorkLifeBalance'
    ]
    
    return categorical_columns

def delete_columns() -> List[str]:
    """
    Retorna las columnas a eliminar del dataset IBM HR Analytics.

    Returns:
        List[str]: Lista con los nombres de las columnas a eliminar.
    """
    columns_to_delete = []

    # Columnas con valores constantes
    columns_to_delete.extend([
        'EmployeeCount',
        'Over18',
        'StandardHours'
    ])

    # Columnas con valores unicos
    columns_to_delete.extend([
        'EmployeeNumber'
    ])

    # Columnas baja correlación con la variable objetivo
    columns_to_delete.extend([  
        'MonthlyRate', 
        'PercentSalaryHike',
        'TrainingTimesLastYear',
        'DistanceFromHome',
        'HourlyRate',
        'DailyRate',
        'EnvironmentSatisfaction',  
        'StockOptionLevel'  
    ])

    # Columnas media correlación con la variable objetivo
    columns_to_delete.extend([  
        'NumCompaniesWorked',
    ])

    # Columnas redundantes
    columns_to_delete.extend([
        'TotalWorkingYears',
        'YearsInCurrentRole',
        'YearsWithCurrManager',
    ])

    # Columnas que provocan sesgo en el análisis
    columns_to_delete.extend([
        'Gender',
        'MaritalStatus',
        'RelationshipSatisfaction',
        'Age'    

    ])
    
    return columns_to_delete

__all__ = [
    'get_categorical_number_columns',
    'delete_columns'
]

