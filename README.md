# Análisis Exploratorio de Datos (EDA) - Titanic Dataset

## Contexto Histórico
El RMS Titanic fue un transatlántico británico que se hundió en el Océano Atlántico Norte en las primeras horas del 15 de abril de 1912, después de chocar con un iceberg durante su viaje inaugural desde Southampton, Reino Unido, hasta la ciudad de Nueva York. Este desastre resultó en la muerte de más de 1.500 pasajeros y tripulantes, lo que lo convierte en uno de los desastres marítimos más mortales en tiempos de paz. El análisis de los datos relacionados con los pasajeros del Titanic ofrece una oportunidad para explorar cómo diferentes factores pudieron haber influido en la supervivencia.

## Objetivos del Proyecto
- Realizar un análisis exploratorio de datos completo del dataset de pasajeros del Titanic.
- Visualizar patrones y relaciones entre diferentes variables.
- Implementar y comparar varios modelos de machine learning para predecir la supervivencia.

## Limitaciones del Análisis
Es importante considerar algunas limitaciones en este análisis:
- **Sesgo de Muestreo**: Los datos disponibles representan solo a una parte de los pasajeros y pueden no ser completamente representativos de todos los que estuvieron a bordo.
- **Calidad de los Datos**: Existen valores faltantes y algunas variables categóricas que pueden no haber sido capturadas con precisión.
- **Suposiciones Simplificadas**: Algunas decisiones en el procesamiento de datos, como la imputación de valores faltantes, se basan en suposiciones que pueden no reflejar la realidad de manera precisa.

## Estructura del Proyecto
1. **Análisis Exploratorio de Datos (EDA)**:
    - Limpieza y procesamiento de datos.
    - Visualización de distribuciones y correlaciones.
    - Clustering de pasajeros y análisis de interacciones.
2. **Modelado Predictivo**:
    - Implementación de modelos de regresión logística, Random Forest, y Gradient Boosting.
    - Validación cruzada y optimización de hiperparámetros.
    - Comparación de modelos y evaluación del desempeño.
3. **Conclusiones y Recomendaciones**:
    - Discusión de los resultados y consideraciones finales.

## Requisitos
Este proyecto utiliza Python y las siguientes bibliotecas:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- plotly

## Ejecución del Proyecto
Clona el repositorio y ejecuta el archivo `titanic_eda.ipynb`. Asegúrate de instalar las dependencias utilizando:

```bash
pip install -r requirements.txt
```

## Resultados del Análisis
- Se identificaron patrones significativos entre variables como la clase del pasajero, el sexo y la edad, con la tasa de supervivencia.
- El modelo de **Random Forest** optimizado obtuvo la mejor precisión con validación cruzada.
- Un análisis contrafactual sugiere que, si todos los pasajeros hubieran estado en primera clase, la tasa de supervivencia podría haber sido mayor.

## Publicación
Este proyecto está disponible en Kaggle y GitHub. Siéntete libre de contribuir y realizar mejoras.

## Contribuciones
Si deseas contribuir al proyecto, por favor sigue los siguientes pasos:
1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature-nueva-caracteristica`).
3. Realiza los cambios y haz commit (`git commit -m 'Añadir nueva característica'`).
4. Haz push a la rama (`git push origin feature-nueva-caracteristica`).
5. Abre un Pull Request.

## Licencia
Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo [LICENSE](LICENSE) para más detalles.