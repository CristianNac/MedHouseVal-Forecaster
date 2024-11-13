# California Housing Price Prediction

Este repositorio contiene un proyecto de predicción del valor de las casas en California utilizando varios algoritmos de Machine Learning. El objetivo es predecir la variable `MedHouseVal` (valor medio de las casas) a partir de un conjunto de características del mercado inmobiliario y demográficas en California.

## Descripción del Proyecto

El conjunto de datos utilizado proviene del conjunto de datos de `fetch_california_housing` de la biblioteca `scikit-learn`, y contiene las siguientes variables:

- **MedInc**: Ingreso medio por unidad de propiedad en el área.
- **HouseAge**: Edad media de las casas en el área.
- **AveRooms**: Promedio de habitaciones por vivienda.
- **AveBedrms**: Promedio de dormitorios por vivienda.
- **Population**: Población total en el área.
- **AveOccup**: Promedio de ocupación por vivienda.
- **Latitude**: Latitud de la ubicación.
- **Longitude**: Longitud de la ubicación.
- **MedHouseVal**: Valor medio de las casas (variable objetivo).

## Análisis Exploratorio de Datos (EDA)

Realicé un pequeño análisis exploratorio de datos (EDA) para comprender mejor las relaciones entre las variables. Durante este análisis, descubrí que las variables `AveRooms` y `AveBedrms` están altamente correlacionadas, con un coeficiente de Pearson alto, lo que sugiere multicolinealidad. Debido a esto, decidí descartar una de estas variables (en este caso, `AveBedrms`) para evitar problemas de multicolinealidad en los modelos.

Además, noté que las características numéricas tenían diferentes escalas, por lo que utilicé **StandardScaler** para estandarizar los datos y mejorar el rendimiento de los modelos.

## Train-Test Split

Para evaluar los modelos, utilicé un **train-test split** con un tamaño de prueba (test size) del **20%** (0.2). Esto significa que el 80% de los datos se utilizaron para entrenar los modelos, y el 20% restante se utilizó para probar su desempeño en datos no vistos.

## Modelos Utilizados

A continuación, se describen los modelos utilizados para la predicción de los valores de `MedHouseVal`:

1. **Regresión Lineal Múltiple**: Modelo de regresión simple para prever la relación lineal entre las variables predictoras y la variable objetivo.
2. **Ridge y Lasso**: Modelos de regresión regularizada (L2 y L1 respectivamente) para evitar el sobreajuste y mejorar la generalización.
3. **Árbol de Decisión**: Modelo basado en particiones recursivas de los datos, útil para capturar relaciones no lineales.
4. **Random Forest**: Un conjunto de árboles de decisión que ayuda a reducir la varianza y mejora la precisión del modelo.
5. **Gradient Boosting**: Un algoritmo de boosting que construye árboles de decisión de manera secuencial para corregir los errores de los modelos anteriores.
6. **XGBoost (Extreme Gradient Boosting)**: Una versión optimizada de Gradient Boosting que mejora la velocidad y la precisión de los modelos.

### Ajuste de Hiperparámetros

Para optimizar los hiperparámetros de los modelos, utilicé **RandomizedSearchCV**. Este método realiza una búsqueda aleatoria sobre un espacio de hiperparámetros definido para encontrar la mejor combinación posible de parámetros para cada modelo.

## Métrica de Evaluación

La métrica principal utilizada para evaluar los modelos fue el **RMSE** (Root Mean Squared Error o Raíz del Error Cuadrático Medio), que mide la diferencia promedio entre los valores predichos y los valores reales, penalizando más los errores grandes.

### Fórmula del RMSE:

La fórmula del RMSE es:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?RMSE%20=%20%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%28y_i%20-%20%5Chat%7By_i%7D%29%5E2%7D" />
</p>

Un **RMSE** bajo indica que el modelo está realizando buenas predicciones, mientras que un valor alto indica que hay un error significativo en las predicciones.

## Resultados

A continuación, se presentan los valores de RMSE obtenidos para cada modelo entrenado:

| **Modelo**                  | **RMSE** |
|-----------------------------|----------|
| Regresión Lineal Múltiple    | 0.81     |
| Ridge Regression             | 0.80     |
| Lasso Regression             | 0.81     |
| Árbol de Decisión           | 0.77     |
| Random Forest               | 0.80     |
| Gradient Boosting           | 0.65     |
| XGBoost                     | 0.64     |

## Análisis de Resultados

Al observar los valores de RMSE, podemos ver que los mejores modelos para este conjunto de datos son **XGBoost** y **Gradient Boosting**, con RMSE de **0.64** y **0.65**, respectivamente. Estos dos modelos de boosting son los que mejor han generalizado, logrando los errores más bajos en las predicciones. 

Por otro lado, los modelos de regresión regularizada (Ridge y Lasso) y la **Regresión Lineal Múltiple** obtuvieron valores de RMSE de **0.80** y **0.81**, lo que indica un desempeño moderado, pero inferior a los modelos de boosting. El **Árbol de Decisión** también mostró un buen desempeño con un RMSE de **0.77**, destacando entre los modelos no lineales.

Esto sugiere que los modelos basados en boosting (como **Gradient Boosting** y **XGBoost**) son los más adecuados para este tipo de predicción, debido a su capacidad para manejar relaciones no lineales en los datos y para mejorar la precisión mediante el enfoque iterativo.
