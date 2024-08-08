# Análisis Exploratorio de Datos del Titanic

## Introducción

El desastre del Titanic es uno de los naufragios más conocidos de la historia. En este proyecto, realizo un análisis exploratorio de datos (EDA) utilizando el famoso dataset del Titanic para descubrir patrones, tendencias y posibles factores que influyeron en la supervivencia de los pasajeros.

## Objetivos del Proyecto

1. **Explorar el dataset del Titanic** para entender mejor la estructura de los datos, las variables más relevantes, y las relaciones entre ellas.
2. **Crear nuevas variables** para enriquecer el análisis.
3. **Construir modelos predictivos** para predecir la supervivencia de los pasajeros y evaluar su desempeño.
4. **Visualizar los resultados** de manera clara y atractiva para identificar patrones clave y sacar conclusiones relevantes.

## Dataset

El dataset utilizado es de dominio público y está disponible en Kaggle. Contiene información sobre los pasajeros del Titanic, incluyendo detalles como clase de pasajero, edad, sexo, tarifa pagada, y si sobrevivieron o no.

## Resultados Destacados

- **Variables Relevantes**: Las variables `Pclass`, `Sex`, `Age`, y `Fare` mostraron una fuerte correlación con la probabilidad de supervivencia.
- **Análisis de Familias**: Pasajeros que viajaban solos tenían una menor probabilidad de supervivencia en comparación con aquellos que viajaban en familia.
- **Modelos Predictivos**: El modelo de regresión logística alcanzó una precisión del XX%, con un buen balance entre precisión y recall.

## Conclusiones

El análisis sugiere que la clase social (`Pclass`) y el sexo (`Sex`) fueron los factores más determinantes en la probabilidad de supervivencia. Este proyecto muestra cómo un análisis exploratorio de datos puede ofrecer insights valiosos a partir de datos históricos.

## Requisitos

- Python 3.x
- Pandas
- Seaborn
- Matplotlib
- Plotly
- Scikit-learn
- Jupyter Notebook (opcional)

## Cómo ejecutar el proyecto

1. Clona el repositorio: 
    ```sh
    git clone https://github.com/olcesefacundo97/Titanic-EDA.git
    ```
2. Navega al directorio del proyecto:
    ```sh
    cd Titanic-EDA
    ```
3. Crea y activa un entorno virtual:
    ```sh
    python -m venv myenv
    source myenv/bin/activate  # En Windows usa: myenv\Scripts\activate
    ```
4. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```
5. Ejecuta el análisis:
    ```sh
    python titanic_eda.py
    ```

## Créditos

Este análisis fue realizado por [Facundo Olcese](https://www.linkedin.com/in/olcesefacundo97/).
