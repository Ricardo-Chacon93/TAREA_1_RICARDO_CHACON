# TAREA_1_RICARDO_CHACON
 
# Clasificador de Propinas para Viajes en Taxi en NYC (2020)

Este proyecto utiliza datos de viajes de taxis amarillos de Nueva York para construir un modelo de machine learning que predice si la propina será alta (mayor al 20% del costo del viaje).

## Descripción del Dataset

El diccionario de los datos puede encontrarse [acá](https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf):

| Field Name      | Description |
| ----------- | ----------- |
| VendorID      | A code indicating the TPEP provider that provided the record. 1= Creative Mobile Technologies, LLC; 2= VeriFone Inc.       |
| tpep_pickup_datetime   | The date and time when the meter was engaged.        |
| tpep_dropoff_datetime   | The date and time when the meter was disengaged.        |
| Passenger_count   | The number of passengers in the vehicle. This is a driver-entered value.      |
| Trip_distance   | The elapsed trip distance in miles reported by the taximeter.      |
| PULocationID   | TLC Taxi Zone in which the taximeter was engaged.      |
| DOLocationID   | TLC Taxi Zone in which the taximeter was disengaged      |
| RateCodeID   | The final rate code in effect at the end of the trip. 1= Standard rate, 2=JFK, 3=Newark, 4=Nassau or Westchester, 5=Negotiated fare, 6=Group ride     |
| Store_and_fwd_flag | This flag indicates whether the trip record was held in vehicle memory before sending to the vendor, aka “store and forward,” because the vehicle did not have a connection to the server. Y= store and forward trip, N= not a store and forward trip |
| Payment_type | A numeric code signifying how the passenger paid for the trip. 1= Credit card, 2= Cash, 3= No charge, 4= Dispute, 5= Unknown, 6= Voided trip |
| Fare_amount | The time-and-distance fare calculated by the meter. |
| Extra | Miscellaneous extras and surcharges. Currently, this only includes the \$0.50 and \$1 rush hour and overnight charges. |
| MTA_tax | \$0.50 MTA tax that is automatically triggered based on the metered rate in use. |
| Improvement_surcharge | \$0.30 improvement surcharge assessed trips at the flag drop. The improvement surcharge began being levied in 2015. |
| Tip_amount | Tip amount – This field is automatically populated for credit card tips. Cash tips are not included. |
| Tolls_amount | Total amount of all tolls paid in trip. |
| Total_amount | The total amount charged to passengers. Does not include cash tips. |

## Estructura del Proyecto

- `data`: Contiene los datos en diferentes estados de procesamiento.
- `models`: Contiene el modelo entrenado.
- `notebooks`: Notebooks de Jupyter para análisis exploratorio y desarrollo.
  - `train_model.ipynb`: Notebook para entrenar el modelo.
  - `test_conclusion.ipynb`: Notebook para probar y concluir el modelo.
- `references`: Documentación y referencias adicionales.
- `reports`: Reportes generados y figuras.
- `src`: Código fuente dividido en módulos.
  - `data`: Carga y preprocesamiento de datos.
  - `features`: Ingeniería de características.
  - `models`: Entrenamiento y evaluación de modelos.
  - `visualization`: Visualización de datos.

## Ejecución 

Para poder ejecutar el entrenamiento, testeo y ver conclusiones debe ejecutar los archivos `train_model.ipynb` y `test_conclusion.ipynb` de la carpeta notebooks

## Instrucciones de Instalación

```bash
pip install -r requirements.txt

