import os
import pandas as pd
from src import f1_score, precision_score, recall_score, accuracy_score

# Crear una función para descargar y guardar los datos
def descargar_datos_mensuales(año, meses, ruta_guardado):
    url_base = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{}-{:02d}.parquet"
    
    if not os.path.exists(ruta_guardado):
        os.makedirs(ruta_guardado)
    
    for mes in meses:
        url = url_base.format(año, mes)
        archivo_guardado = os.path.join(ruta_guardado, f"yellow_tripdata_{año}-{mes:02d}.parquet")
        try:
            # Descargar y guardar el archivo Parquet
            df = pd.read_parquet(url)
            df.to_parquet(archivo_guardado)
            print(f"Datos de {año}-{mes:02d} descargados y guardados en {archivo_guardado}")
        except Exception as e:
            print(f"Error al descargar los datos de {año}-{mes:02d}: {e}")


def preprocess(df, target_col,features, EPS):

   # Basic cleaning
    df = df[df['fare_amount'] > 0].reset_index(drop=True)  # avoid divide-by-zero
    # add target
    df['tip_fraction'] = df['tip_amount'] / df['fare_amount']
    df[target_col] = df['tip_fraction'] > 0.2

    # add features
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_minute'] = df['tpep_pickup_datetime'].dt.minute
    df['work_hours'] = (df['pickup_weekday'] >= 0) & (df['pickup_weekday'] <= 4) & (df['pickup_hour'] >= 8) & (df['pickup_hour'] <= 18)
    df['trip_time'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.seconds
    df['trip_speed'] = df['trip_distance'] / (df['trip_time'] + EPS)

    # drop unused columns
    df = df[['tpep_dropoff_datetime'] + features + [target_col]]
    df[features + [target_col]] = df[features + [target_col]].astype("float32").fillna(-1.0)

    # convert target to int32 for efficiency (it's just 0s and 1s)
    df[target_col] = df[target_col].astype("int32")

    return df.reset_index(drop=True)

def evaluar_modelo_por_mes(año, meses, ruta_guardado, modelo, features, target_col,EPS):
    resultados = []

    for mes in meses:
        archivo_mes = f"{ruta_guardado}/yellow_tripdata_{año}-{mes:02d}.parquet"
        try:
            taxi_mes = pd.read_parquet(archivo_mes)
            taxi_test = preprocess(taxi_mes, target_col, features, EPS)
            
            preds_test = modelo.predict_proba(taxi_test[features])
            preds_test_labels = [p[1] for p in preds_test.round()]
            
            f1 = f1_score(taxi_test[target_col], preds_test_labels)
            precision = precision_score(taxi_test[target_col], preds_test_labels)
            recall = recall_score(taxi_test[target_col], preds_test_labels)
            accuracy = accuracy_score(taxi_test[target_col], preds_test_labels)
            
            resultados.append({
                'mes': mes,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy
            })
            print(f"{año}-{mes:02d} -> F1: {f1}, Precisión: {precision}, Recall: {recall}, Exactitud: {accuracy}")
        except Exception as e:
            print(f"Error al evaluar los datos de {año}-{mes:02d}: {e}")
    
    return resultados

# Función para preprocesar los datos combinados
def preprocess_2(df, target_col='high_tip'):
    EPS = 1e-9
    df['tip_fraction'] = df['tip_amount'] / df['total_amount']
    df[target_col] = df['tip_fraction'] > 0.2

    # add features
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_minute'] = df['tpep_pickup_datetime'].dt.minute
    df['work_hours'] = (df['pickup_weekday'] >= 0) & (df['pickup_weekday'] <= 4) & (df['pickup_hour'] >= 8) & (df['pickup_hour'] <= 18)
    df['trip_time'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.seconds
    df['trip_speed'] = df['trip_distance'] / (df['trip_time'] + EPS)

    features = ['trip_distance', 'pickup_weekday', 'pickup_hour', 'pickup_minute', 'work_hours', 'trip_time', 'trip_speed']
    df = df[features + [target_col, 'mes']]
    df[features + [target_col]] = df[features + [target_col]].astype("float32").fillna(-1.0)
    df[target_col] = df[target_col].astype("int32")

    return df.reset_index(drop=True)

# Función para cargar y combinar datos mensuales
def cargar_y_combinar_datos(año, meses, ruta_guardado, target_col):
    combined_data = pd.DataFrame()
    
    for mes in meses:
        archivo_mes = f"{ruta_guardado}/yellow_tripdata_{año}-{mes:02d}.parquet"
        try:
            taxi_mes = pd.read_parquet(archivo_mes)
            taxi_mes['mes'] = mes  # Agregar columna del mes
            taxi_mes = preprocess_2(taxi_mes, target_col=target_col)
            combined_data = pd.concat([combined_data, taxi_mes], ignore_index=True)
            print(f"Datos de {año}-{mes:02d} cargados y combinados.")
        except Exception as e:
            print(f"Error al cargar los datos de {año}-{mes:02d}: {e}")
    
    return combined_data