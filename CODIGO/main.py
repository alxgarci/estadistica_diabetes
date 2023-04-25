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

    #######################################
    #          PREVISUALIZACION           #
    #######################################

    # Media, Mediana, Moda, Rango, Desviacion Tipica, Varianza
    # p.Series(diabetes["Pregnancies"].values.flatten()).mode()[0]
    datos_dict = {
        "Pregnancies": [diabetes["Pregnancies"].mean(), diabetes["Pregnancies"].median(),
                        diabetes["Pregnancies"].mode()[0],
                        diabetes["Pregnancies"].max() - diabetes["Pregnancies"].min(), diabetes["Pregnancies"].std(),
                        diabetes["Pregnancies"].var()],
        "Glucose": [diabetes["Glucose"].mean(), diabetes["Glucose"].median(),
                    diabetes["Glucose"].mode()[0],
                    diabetes["Glucose"].max() - diabetes["Glucose"].min(), diabetes["Glucose"].std(),
                    diabetes["Glucose"].var()],
        "BloodPressure": [diabetes["BloodPressure"].mean(), diabetes["BloodPressure"].median(),
                          diabetes["BloodPressure"].mode()[0],
                          diabetes["BloodPressure"].max() - diabetes["BloodPressure"].min(),
                          diabetes["BloodPressure"].std(),
                          diabetes["BloodPressure"].var()],
        "SkinThickness": [diabetes["SkinThickness"].mean(), diabetes["SkinThickness"].median(),
                          diabetes["SkinThickness"].mode()[0],
                          diabetes["SkinThickness"].max() - diabetes["SkinThickness"].min(),
                          diabetes["SkinThickness"].std(),
                          diabetes["SkinThickness"].var()],
        "Insulin": [diabetes["Insulin"].mean(), diabetes["Insulin"].median(),
                    diabetes["Insulin"].mode()[0],
                    diabetes["Insulin"].max() - diabetes["Insulin"].min(), diabetes["Insulin"].std(),
                    diabetes["Insulin"].var()],
        "BMI": [diabetes["BMI"].mean(), diabetes["BMI"].median(),
                diabetes["BMI"].mode()[0],
                diabetes["BMI"].max() - diabetes["BMI"].min(), diabetes["BMI"].std(),
                diabetes["BMI"].var()],
        "DiabetesPedigreeFunction": [diabetes["DiabetesPedigreeFunction"].mean(),
                                     diabetes["DiabetesPedigreeFunction"].median(),
                                     diabetes["DiabetesPedigreeFunction"].mode()[0],
                                     diabetes["DiabetesPedigreeFunction"].max() - diabetes[
                                         "DiabetesPedigreeFunction"].min(), diabetes["DiabetesPedigreeFunction"].std(),
                                     diabetes["DiabetesPedigreeFunction"].var()],
        "Age": [diabetes["Age"].mean(), diabetes["Age"].median(),
                diabetes["Age"].mode()[0],
                diabetes["Age"].max() - diabetes["Age"].min(), diabetes["Age"].std(),
                diabetes["Age"].var()],
        "Outcome": [diabetes["Outcome"].mean(), diabetes["Outcome"].median(),
                    diabetes["Outcome"].mode()[0],
                    diabetes["Outcome"].max() - diabetes["Outcome"].min(), diabetes["Outcome"].std(),
                    diabetes["Outcome"].var()]

    }

    p.options.display.float_format = '{:,.2f}'.format
    columnas = ["Media", "Mediana", "Moda", "Rango", "Desviacion Tipica", "Varianza"]
    media_moda_etc = p.DataFrame.from_dict(datos_dict, orient="index",
                                           columns=columnas)

    print(media_moda_etc.to_string())

    media_moda_etc.loc[media_moda_etc.index.drop(["Insulin", "Glucose"])].plot.bar(rot=0)

    plot.title("Media, moda, etc de todas las variables")
    plot.show()

    media_moda_etc.loc[["Insulin", "Glucose"]].plot.bar(rot=0)

    plot.title("Media, moda, etc de Insulin y Glucose")
    plot.show()

    # Histograma
    for x in diabetes.columns:
        sb.histplot(diabetes[x].T)
        plot.title("Histograma de " + x)
        plot.show()

    # Diagrama de cajas
    columnas_diabetes = list(diabetes)
    print(columnas_diabetes)
    sb.boxplot(diabetes[diabetes.columns.drop(["Insulin", "Glucose"])])
    plot.title("Diagrama de cajas (1/2)")
    plot.show()
    #
    # sb.boxplot(diabetes[["Insulin", "Glucose"]])
    # plot.title("Diagrama de cajas (2/2)")
    # plot.show()


    # Pruebas
    # for x in media_moda_etc.index:
    #     # sb.barplot(x=media_moda_etc.columns, data=media_moda_etc.loc[x])
    #     media_moda_etc.loc[x].plot.bar(rot=0)
    #     plot.title("Media, moda, etc de " + x)
    #     plot.show()


if __name__ == '__main__':
    main()
