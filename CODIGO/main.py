import math
import warnings

import matplotlib
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sb
import numpy as np
from scipy import stats
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
    # In[14]:

    # Diagrama de correlacion por pares, para entender la anterior matriz de correlacion
    plot.figure()
    sb.pairplot(stroke)

    # In[15]:

    # Encontramos que age y ever_married tienen una alta correlacion, imprimimos su correlacion en grafica
    # sb.scatterplot(stroke, x="age", y="avg_glucose_level")

    plot.figure()
    plot.scatter(x=stroke["bmi"], y=stroke["avg_glucose_level"])

    # In[18]:

    slope, intercept, r, p, std_err = stats.stats.linregress(stroke["bmi"], stroke["avg_glucose_level"])

    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, stroke["bmi"]))
    plot.figure()
    plot.scatter(stroke["bmi"], stroke["avg_glucose_level"])
    plot.plot(stroke["bmi"], mymodel, "r")
    plot.show()

    # Intervalos de confianza para variables
    # Medias
    media_bmi = stroke["bmi"].mean()
    media_glucose = stroke["avg_glucose_level"].mean()

    # Desviacion estandar
    std_bmi = stroke["bmi"].std()
    std_glucose = stroke["avg_glucose_level"].std()

    # Error estandar
    err_estandar_bmi = std_bmi / math.sqrt(len(stroke["bmi"]))
    err_estandar_glucose = std_bmi / math.sqrt(len(stroke["avg_glucose_level"]))

    # Encontramos nuestro t* value
    t_bmi = stats.t.ppf(0.95, df=len(stroke["bmi"]) - 1)
    t_glucose = stats.t.ppf(0.95, df=len(stroke["avg_glucose_level"]) - 1)

    # Sacamos los intervalos de confianza
    ic_bmi_suma = media_bmi + t_bmi * err_estandar_bmi
    ic_bmi_resta = media_bmi - t_bmi * err_estandar_bmi

    ic_glucose_suma = media_glucose + t_glucose * err_estandar_glucose
    ic_glucose_resta = media_glucose + t_glucose * err_estandar_glucose

    df_res = pd.DataFrame()
    df_res["bmi"] = [media_bmi, std_bmi, err_estandar_bmi, t_bmi, ic_bmi_suma, ic_bmi_resta]
    df_res["avg_glucose_level"] = [media_glucose, std_glucose, err_estandar_glucose, t_glucose, ic_glucose_suma,
                                   ic_glucose_resta]
    df_res = df_res.set_index(pd.Index(["Media", "Desviacion Estandar", "Error estandar", "t",
                                        "Intervalo de confianza (sumado)", "Intervalo de confianza (restado)"]))
    print(df_res.to_string())

    # Contrastes de hipótesis de cada una de las dos variables

    # # # BMI
    # Pondremos como hipotesis nula que la media es superior a 40 (H0)
    # y como hipotesis alternativa que es diferente a 40
    # Establecemos el nivel de significancia por ejemplo a 0.05

    alfa = 0.05
    H0 = 40
    st_hipotesis = f"Que la media sea igual a {H0}"
    st_alt_hipotesis = f"Que la media sea diferente a {H0}"

    # Calculamos t-value y p-value
    t_value_bmi, p_value_bmi = stats.ttest_1samp(stroke["bmi"], H0)

    df_res = pd.DataFrame()

    # Comparamos los resultados con el nivel de significancia (alfa)
    if p_value_bmi < alfa:
        st_cond = "Se rechaza la Hipotesis Nula (H0)"
    else:
        st_cond = "No se puede rechazar la Hipotesis Nula (H0)"
    df_res["bmi"] = [alfa, t_value_bmi, p_value_bmi, st_hipotesis, st_alt_hipotesis, st_cond]

    # # # avg_glucose_level
    # Pondremos como hipotesis nula que la media es superior a 109 (H0)
    # y como hipotesis alternativa que es diferente a 109
    # Establecemos el nivel de significancia por ejemplo a 0.05
    alfa = 0.05
    H0 = 109
    st_hipotesis = f"Que la media sea igual a {H0}"
    st_alt_hipotesis = f"Que la media sea diferente a {H0}"

    t_value_glucose, p_value_glucose = stats.ttest_1samp(stroke["avg_glucose_level"], H0)

    # Comparamos los resultados con el nivel de significancia (alfa)
    if p_value_glucose < alfa:
        st_cond = "Se rechaza la Hipotesis Nula (H0)"
    else:
        st_cond = "No se puede rechazar la Hipotesis Nula (H0)"

    df_res["avg_glucose_level"] = [alfa, t_value_glucose, p_value_glucose, st_hipotesis, st_alt_hipotesis, st_cond]
    df_res = df_res.set_index(
        pd.Index(["Nivel de significancia", "Valor 't'", "Valor 'p'", "Hipotesis H0", "Hipotesis Ha", "Resultado"]))
    print(df_res.to_string())


if __name__ == '__main__':
    main()
