# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd 
import os 
import gzip 
import pickle 
import json 
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

def obtener_datos_desde_archivo_comprimido(ruta_archivo: str) -> pd.DataFrame:
    """Función que lee un archivo CSV comprimido en formato zip y retorna un DataFrame"""
    return pd.read_csv(ruta_archivo, index_col=False, compression="zip")

def obtener_datos_desde_archivo_comprimido(ruta_archivo: str) -> pd.DataFrame:
    """Función que lee un archivo CSV comprimido en formato zip y retorna un DataFrame"""
    return pd.read_csv(ruta_archivo, index_col=False, compression="zip")

def limpiar_conjunto_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesamiento inicial de los datos: renombra columnas, elimina registros inválidos"""
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.loc[df["MARRIAGE"] != 0] 
    df = df.loc[df["EDUCATION"] != 0] 
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
    return df

def construir_pipeline_modelo() -> Pipeline:
    """Creación de la pipeline de procesamiento y clasificación con Random Forest"""
    variables_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    transformador = ColumnTransformer(
        transformers=[("categoricas", OneHotEncoder(handle_unknown="ignore"), variables_categoricas)],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("transformacion", transformador),
            ("modelo", RandomForestClassifier(random_state=42)),
        ]
    )

def configurar_optimizador_parametros(pipeline: Pipeline) -> GridSearchCV:
    """Creación del estimador con GridSearchCV para optimizar hiperparámetros"""
    grilla_parametros = {
        "modelo__n_estimators": [50, 100, 200],
        "modelo__max_depth": [None, 5, 10, 20],
        "modelo__min_samples_split": [2, 5, 10],
        "modelo__min_samples_leaf": [1, 2, 4],
    }

    return GridSearchCV(
        pipeline,
        grilla_parametros,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

def persistir_modelo_entrenado(ruta: str, estimador: GridSearchCV):
    """Guardar el modelo entrenado en un archivo comprimido"""
    os.makedirs(os.path.dirname(ruta), exist_ok=True) 
    with gzip.open(ruta, "wb") as f:
        pickle.dump(estimador, f)

def calcular_metricas_rendimiento(nombre_dataset: str, y_real, y_predicho) -> dict:
    """Calcular métricas de precisión y otras métricas de evaluación"""
    return {
        "type": "metrics",
        "dataset": nombre_dataset,
        "precision": precision_score(y_real, y_predicho, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_predicho),
        "recall": recall_score(y_real, y_predicho, zero_division=0),
        "f1_score": f1_score(y_real, y_predicho, zero_division=0),
    }

def obtener_metricas_matriz_confusion(nombre_dataset: str, y_real, y_predicho) -> dict:
    """Calcular la matriz de confusión y sus métricas asociadas"""
    matriz_confusion = confusion_matrix(y_real, y_predicho)
    return {
        "type": "cm_matrix",
        "dataset": nombre_dataset,
        "true_0": {"predicted_0": int(matriz_confusion[0][0]), "predicted_1": int(matriz_confusion[0][1])},
        "true_1": {"predicted_0": int(matriz_confusion[1][0]), "predicted_1": int(matriz_confusion[1][1])},
    }

def ejecutar_proceso_completo():
    """Función principal que ejecuta todo el flujo de trabajo"""

    # Carga de datasets de entrenamiento y prueba
    datos_prueba = obtener_datos_desde_archivo_comprimido(os.path.join("files/input/", "test_data.csv.zip"))
    datos_entrenamiento = obtener_datos_desde_archivo_comprimido(os.path.join("files/input/", "train_data.csv.zip"))

    # Limpieza y preprocesamiento de los datos
    datos_prueba = limpiar_conjunto_datos(datos_prueba)
    datos_entrenamiento = limpiar_conjunto_datos(datos_entrenamiento)

    # Separación de variables predictoras y objetivo para conjunto de prueba
    caracteristicas_prueba = datos_prueba.drop(columns=["default"])
    objetivo_prueba = datos_prueba["default"]

    # Separación de variables predictoras y objetivo para conjunto de entrenamiento
    caracteristicas_entrenamiento = datos_entrenamiento.drop(columns=["default"])
    objetivo_entrenamiento = datos_entrenamiento["default"]

    # Construcción del pipeline de machine learning
    pipeline = construir_pipeline_modelo()

    # Configuración y entrenamiento del optimizador de hiperparámetros
    optimizador = configurar_optimizador_parametros(pipeline)
    optimizador.fit(caracteristicas_entrenamiento, objetivo_entrenamiento)

    # Persistencia del modelo entrenado
    persistir_modelo_entrenado(os.path.join("files/models/", "model.pkl.gz"), optimizador)

    # Generación de predicciones y cálculo de métricas
    predicciones_prueba = optimizador.predict(caracteristicas_prueba)
    metricas_conjunto_prueba = calcular_metricas_rendimiento("test", objetivo_prueba, predicciones_prueba)
    predicciones_entrenamiento = optimizador.predict(caracteristicas_entrenamiento)
    metricas_conjunto_entrenamiento = calcular_metricas_rendimiento("train", objetivo_entrenamiento, predicciones_entrenamiento)

    # Cálculo de matrices de confusión
    matriz_confusion_prueba = obtener_metricas_matriz_confusion("test", objetivo_prueba, predicciones_prueba)
    matriz_confusion_entrenamiento = obtener_metricas_matriz_confusion("train", objetivo_entrenamiento, predicciones_entrenamiento)

    # Persistencia de métricas en archivo JSON
    os.makedirs("files/output/", exist_ok=True)
    with open(os.path.join("files/output/", "metrics.json"), "w") as archivo:
        archivo.write(json.dumps(metricas_conjunto_entrenamiento) + "\n")
        archivo.write(json.dumps(metricas_conjunto_prueba) + "\n")
        archivo.write(json.dumps(matriz_confusion_entrenamiento) + "\n")
        archivo.write(json.dumps(matriz_confusion_prueba) + "\n")

if __name__ == "__main__":
    ejecutar_proceso_completo()