# Detección de Inconsistencias Salariales mediante Modelos Predictivos

## Contexto y Motivación

La entrada en vigor de la **Directiva Europea de Transparencia Salarial (UE 2023/970)** exige a las empresas justificar de forma objetiva cualquier diferencia salarial relevante entre empleados que desempeñan funciones similares. Sin herramientas adecuadas, esta tarea resulta difícil para departamentos de RRHH, especialmente en organizaciones con plantillas amplias y estructuras salariales complejas.

El riesgo de incurrir en desigualdades retributivas no justificadas no solo implica consecuencias legales, sino también un impacto reputacional significativo. Por ello, surge la necesidad de un sistema analítico capaz de identificar incoherencias salariales de forma automática, transparente y basada en datos.

---

## Objetivo del Proyecto

El propósito del proyecto es construir un **modelo predictivo de regresión** capaz de estimar el **salario mensual esperado (MonthlyIncome)** de un empleado a partir de sus características profesionales.  
Comparando el salario real con el estimado se pueden detectar:

- posibles inconsistencias salariales,
- casos de riesgo de inequidad interna,
- desviaciones que requieren revisión,
- costes de regularización.

El sistema actúa como soporte cuantitativo para auditorías internas y cumplimiento normativo.

---

## Análisis Exploratorio de Datos

### Contexto Inicial

El dataset objeto de estudio contiene **1,470 registros** distribuidos en **35 variables**. La primera inspección reveló la ausencia de valores nulos, lo que facilitó el proceso inicial de exploración.

Durante esta fase preliminar se identificaron variables categóricas incorrectamente clasificadas como numéricas, las cuales fueron convertidas a su tipo correspondiente. Aunque los valores numéricos se encontraban dentro de rangos esperados, se decidió realizar un análisis exhaustivo de todas las variables para optimizar el modelo predictivo.

### Estrategia de Selección de Variables

El objetivo principal fue identificar y eliminar aquellas variables que no aportaran capacidad diferenciadora o que pudieran introducir ruido en el modelo, comprometiendo así su rendimiento y consistencia.

#### Variables Eliminadas

El análisis resultó en la identificación de **15 variables candidatas** para eliminación, clasificadas en cuatro categorías:

**1. Variables Constantes (3)**

- `EmployeeCount`, `Over18`, `StandardHours`: Varianza nula, sin aporte informativo

**2. Identificadores Únicos (1)**

- `EmployeeNumber`: Sin poder predictivo al tratarse de un identificador

**3. Variables con Baja Correlación (8)**

- `MonthlyRate` (0.03), `PercentSalaryHike` (-0.03), `TrainingTimesLastYear` (-0.02)
- `DistanceFromHome` (-0.02), `HourlyRate` (-0.02), `DailyRate` (0.01)
- `EnvironmentSatisfaction` (-0.01), `StockOptionLevel` (0.01)

**4. Variables Redundantes por Multicolinealidad (3)**

- `TotalWorkingYears`: Correlación 0.78 con JobLevel
- `YearsInCurrentRole`: Correlación 0.76 con YearsAtCompany
- `YearsWithCurrManager`: Correlación 0.77 con YearsAtCompany

![Mapa de correlaciones](images/correlation_heatmap.png)

![Correlaciones con variable objetivo](images/correlaciones_con_objetivo.png)

### Consideraciones Éticas

En coordinación con el departamento de RRHH, se identificaron **variables con potencial sesgo discriminatorio**: `Gender`, `MaritalStatus`, `RelationshipSatisfaction`, `Age`. Previo a su eliminación, se procedió a validar la consistencia del dataset.

### Validación de Consistencia

#### Coherencia Temporal

El análisis univariado temporal confirmó la ausencia de inconsistencias cronológicas. Los valores atípicos detectados fueron validados por RRHH como casos documentados y justificados.

![Análisis de consistencia temporal](images/analisis_consistencia_temporal.png)

#### Análisis de Outliers Salariales

El análisis univariado inicial sugería salarios fuera de rango para ciertos roles. Sin embargo, el **análisis multivariado** (considerando rol, nivel y experiencia) demostró que estos valores estaban justificados y no representaban inconsistencias salariales.

![Análisis multivariado salario-rol-nivel](images/analisis_salario_rol_nivel.png)

---

## Inicialización y Uso

Proyecto realizado con la versión 3.13 de python, no se puede garantizar que con versiones inferiores funcione todas la librerias

```bash
# Crear entorno virtual con version especifica
python3.13 -m venv .venv

# Activar entono virtual en entorno Unix (Linux y Mac)
source ./.venv/bin/activate

# Instalar librerias necesarias
pip3 install -r requirements.txt
```
