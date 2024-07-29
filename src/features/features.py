numeric_feat = [
    "pickup_weekday",
    "pickup_hour",
    'work_hours',
    "pickup_minute",
    "passenger_count",
    'trip_distance',
    'trip_time',
    'trip_speed'
]
categorical_feat = [
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
]
features = numeric_feat + categorical_feat
EPS = 1e-7

target_col = "high_tip"

año = 2020
meses = range(1, 13)  # Meses de enero a diciembre
ruta_guardado = "../data/raw/"
ruta_datos_comb = "datos_taxi_combinados_2020.parquet"
ruta_guarda_modelo = "../models/random_forest.joblib"