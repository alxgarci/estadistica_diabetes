import matplotlib.pyplot as plt
import pandas as p
import matplotlib
import matplotlib.pyplot as plot
import pandas as pd
import seaborn as sb
import numpy as np

from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

matplotlib.use("TkAgg")


def main():
    diabetes = p.read_csv("diabetes.csv")
    print("#" * 84)
    print(">Filas totales de datos: " + str(len(diabetes.index)))
    print(diabetes.to_string())

    # Importamos nuestros datos desde el csv
    diabetes_original = p.read_csv("diabetes.csv")

    # Quitamos la columna "Outcome" ya que no nos servirá para nuestro estudio estadístico
    diabetes = diabetes_original.drop("Outcome", axis="columns")
    print(diabetes.to_string())

    # Calculamos la Media, Mediana, Moda, Rango, Desviacion Tipica, Varianza de cada variable y la metemos en un
    # diccionario que mas tarde representaremos en una tabla
    datos_dict = dict()
    for x in diabetes.columns:
        datos_dict.update({
            x: [diabetes[x].mean(), diabetes[x].median(),
                diabetes[x].mode()[0],
                diabetes[x].max() - diabetes[x].min(), diabetes[x].std(),
                diabetes[x].var()]
        })

    columnas = ["Media", "Mediana", "Moda", "Rango", "Desviacion Tipica", "Varianza"]
    media_moda_etc = p.DataFrame.from_dict(datos_dict, orient="index", columns=columnas)
    print(media_moda_etc.to_string())

    # Graficas de barras de la Media, Mediana, Moda, Rango, Desviacion Tipica de cada variable
    varianza = media_moda_etc["Varianza"]
    media_moda_etc2 = media_moda_etc.drop(["Varianza"], axis="columns")
    media_moda_etc2.loc[media_moda_etc2.index.drop(["Insulin", "Glucose", "BloodPressure"])].plot.bar(rot=0)
    plot.xticks(rotation=45)
    plot.show()
    media_moda_etc2.loc[["Insulin", "Glucose", "BloodPressure"]].plot.bar(rot=0)
    plot.xticks(rotation=45)
    plot.show()

    # Varianza de cada variable
    plot.figure()
    varianza.plot.bar(rot=0)
    # Etiquetas rotadas para mejor lectura
    plot.xticks(rotation=45)
    plot.show()

    # Histogramas de cada variable
    for x in diabetes.columns:
        plot.figure()
        sb.histplot(diabetes[x].T)
        plot.title("Histograma de " + x)
        plot.show()

    # Diagrama de cajas de cada variable
    sb.boxplot(diabetes[diabetes.columns.drop(["Insulin", "Glucose", "BloodPressure"])])
    plot.title("Diagrama de cajas (1/2)")
    plot.xticks(rotation=45)
    plot.show()

    plot.figure()
    sb.boxplot(diabetes[["Insulin", "Glucose", "BloodPressure"]])
    plot.title("Diagrama de cajas (2/2)")
    plot.xticks(rotation=45)
    plot.show()

    # Creamos una matriz de correlacion para poder ver cuales son
    # las variables más relacionadas de forma clara por medio de su coeficiente de correlacion
    sb.heatmap(diabetes.corr())
    plot.title("Matriz de coeficiente de correlación entre variables")
    plot.show()

    # Diagrama de correlacion por pares, para entender la anterior matriz de correlacion
    sb.pairplot(diabetes)
    plot.show()

    # TODO:
    # ✓Un análisis de regresión para las dos variables continuas, así como su gráfico de dispersión y el
    # coeficiente de correlación.
    # ✓Gráficos y tablas que pueden explicar las variables de una manera más completa (diagramas de
    # caja según variable categórica, por ejemplo)


if __name__ == '__main__':
    main()
