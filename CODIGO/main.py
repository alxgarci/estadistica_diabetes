import warnings

import matplotlib
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sb
import numpy as np
from scipy.stats import stats
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
matplotlib.use("TkAgg")


def main():
    # Importamos nuestros datos desde el csv
    stroke_original = pd.read_csv("stroke.csv", na_values=[np.nan])
    stroke_original.drop("id", axis="columns", inplace=True)
    stroke_original.gender = [1 if each == "Male" else 0 for each in stroke_original.gender]
    stroke_original.ever_married = [1 if each == "Yes" else 0 for each in stroke_original.ever_married]
    stroke_original.work_type = [1 if each == "Private" else 0 for each in stroke_original.work_type]
    stroke_original.Residence_type = [1 if each == "Urban" else 0 for each in stroke_original.Residence_type]
    stroke_original.smoking_status = [1 if each == "formerly smoked" else 0 for each in stroke_original.smoking_status]
    # Quitamos la columna "Outcome" ya que no nos servirá para nuestro estudio estadístico
    # stroke = stroke_original.drop("stroke", axis="columns")
    stroke = stroke_original.dropna(axis="rows")
    print(stroke.to_string())

    # In[9]:

    # Calculamos la Media, Mediana, Moda, Rango, Desviacion Tipica, Varianza de cada variable y la metemos en un
    # diccionario que mas tarde representaremos en una tabla
    datos_dict = dict()
    for x in stroke.columns:
        stroke[x] = stroke[x].astype(str)
        stroke[x] = pd.to_numeric(stroke[x])
        datos_dict.update({
            x: [stroke[x].mean(), stroke[x].median(),
                stroke[x].mode()[0],
                stroke[x].max() - stroke[x].min(), stroke[x].std(),
                stroke[x].var()]
        })

    columnas = ["Media", "Mediana", "Moda", "Rango", "Desviacion Tipica", "Varianza"]
    media_moda_etc = pd.DataFrame.from_dict(datos_dict, orient="index", columns=columnas)
    print(media_moda_etc.to_string())

    # In[10]:

    # Graficas de barras de la Media, Mediana, Moda, Rango, Desviacion Tipica de cada variable
    varianza = media_moda_etc["Varianza"]
    # media_moda_etc2.loc[media_moda_etc2.index.drop(["Insulin", "Glucose", "BloodPressure"])].plot.bar(rot=0)
    # plot.xticks(rotation=45)
    # media_moda_etc2.loc[["Insulin", "Glucose", "BloodPressure"]].plot.bar(rot=0)
    # media_moda_etc.drop(["Varianza"], axis="columns").plot.bar(rot=0)

    for x in media_moda_etc.index:
        plot.figure()
        plot.title("Grafica de barras de análisis descriptivo de " + x)
        media_moda_etc.loc[x].plot.bar(rot=0)
        plot.xticks(rotation=45)
        plot.show()

    # Varianza de cada variable
    plot.figure()

    plot.title("Grafica de barras de varianza de las variables")
    varianza.plot.bar(rot=0)
    plot.xticks(rotation=45)
    plot.show()
    # Etiquetas rotadas para mejor lectura

    # In[11]:

    # Histogramas de cada variable stroke[["hypertension", "heart_disease", "ever_married", "work_type",
    # "Residence_type", "smoking_status", "stroke"]].hist(figsize = (20,20))

    stroke[stroke.columns.drop(
        ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status",
         "stroke"])].hist(figsize=(6, 6))
    plot.show()

    plot.figure()
    count_an = sb.countplot(x="gender", data=stroke)
    count_an.set_xticklabels(labels=["Female", "Male"])
    count_an.set(xlabel="gender", ylabel="Personas")
    plot.show()

    for x in stroke.columns.drop(["age", "avg_glucose_level", "gender", "bmi"]):
        plot.figure()
        count_an = sb.countplot(x=x, data=stroke)
        count_an.set_xticklabels(labels=["No", "Yes"])
        count_an.set(xlabel=x, ylabel="Personas")
        plot.show()

    # In[12]:

    # Diagrama de cajas de cada variable
    stroke[["work_type", "ever_married", "Residence_type", "gender", "heart_disease", "hypertension"]].boxplot(
        figsize=(6, 6))
    plot.title("Diagrama de cajas de variable categorica gender")
    plot.xticks(rotation=45)

    # plot.figure()
    # sb.boxplot(stroke[stroke.columns[5:]])
    # plot.title("Diagrama de cajas (2/2)")
    # plot.xticks(rotation=45)

    # In[13]:

    # Creamos una matriz de correlacion para poder ver cuales son las variables más relacionadas de forma clara por
    # medio de su coeficiente de correlacion
    plot.figure()
    sb.heatmap(stroke.corr(), annot=True)
    plot.title("Matriz de coeficiente de correlación entre variables")
    plot.figure()
    # In[14]:

    # Diagrama de correlacion por pares, para entender la anterior matriz de correlacion
    sb.pairplot(stroke)

    # In[15]:

    # Encontramos que age y ever_married tienen una alta correlacion, imprimimos su correlacion en grafica
    # sb.scatterplot(stroke, x="age", y="avg_glucose_level")

    plot.scatter(x=stroke["bmi"], y=stroke["avg_glucose_level"])
    plot.figure()

    # In[18]:

    slope, intercept, r, p, std_err = stats.linregress(stroke["bmi"], stroke["avg_glucose_level"])

    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, stroke["bmi"]))
    plot.scatter(stroke["bmi"], stroke["avg_glucose_level"])
    plot.plot(stroke["bmi"], mymodel, "r")
    plot.show()


if __name__ == '__main__':
    main()
