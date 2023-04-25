import pandas as p
import matplotlib
import matplotlib.pyplot as plot
import seaborn as sb
import numpy as np

from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

matplotlib.use("TkAgg")

precisiones_array = []
kfold_validation_array = []

kfold = KFold(n_splits=10)


def add_acc_to_array(Y_test, prediction):
    accurancy = metrics.accuracy_score(y_true=Y_test, y_pred=prediction)
    precision = metrics.precision_score(y_true=Y_test, y_pred=prediction)
    recall = metrics.recall_score(y_true=Y_test, y_pred=prediction)
    f1 = metrics.f1_score(y_true=Y_test, y_pred=prediction)
    data = [accurancy, precision, recall, f1]
    precisiones_array.append(data)


def add_kfold_to_array(modelo, data, outcome):
    kfold_res = cross_val_score(modelo, data, outcome, cv=kfold)
    kfold_validation_array.append(kfold_res.mean())


def main():
    diabetes = p.read_csv("diabetes.csv")
    print("#" * 84)
    print(">Filas antes de preprocesado: " + str(len(diabetes.index)))

    #######################################
    #          PREPROCESAMIENTO           #
    #######################################

    # Se sustituyen los 0 por NaN y mas tarde se eliminan las filas que contienen NaN
    # Excluyendo la columna Outcome (Son datos imposibles)
    diabetes[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = diabetes[
        ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan)
    diabetes = diabetes.dropna()
    print(">Filas después de preprocesado: " + str(len(diabetes.index)))
    print(diabetes.head())
    print("#" * 84)

    #######################################
    #        ANALISIS EXPLORATORIO        #
    #######################################
    count_an = sb.countplot(x="Outcome", data=diabetes)
    count_an.set_xticklabels(labels=["No", "Yes"])
    count_an.set(xlabel="Outcome", ylabel="Personas")

    plot.title("OUTCOME")
    plot.show()

    sb.pairplot(data=diabetes, hue="Outcome", diag_kind="kde")
    plot.show()

    outcome = diabetes["Outcome"]
    data = diabetes[diabetes.columns[:8]]

    # Dividimos los datos en train y test
    train, test = train_test_split(diabetes, test_size=0.33, random_state=42, stratify=diabetes["Outcome"])

    # Dividimos los train y test en datos numericos y categoricos
    X_train = train[train.columns[:8]]
    X_test = test[test.columns[:8]]
    Y_train = train["Outcome"]
    Y_test = test["Outcome"]

    # Ya tenemos los conjuntos train con X (categorias) y con Y (datos numericos),
    # ya podemos introducirlos en los distintos modelos de clasificacion

    #######################################
    #      MODELOS DE CLASIFICACION       #
    #######################################
    # SVM Lineal
    modelo = svm.SVC(kernel="linear", random_state=122)
    modelo.fit(X_train, Y_train)
    prediction = modelo.predict(X_test)
    add_acc_to_array(Y_test, prediction)
    add_kfold_to_array(modelo, data, outcome)

    # Regresion Logistica
    modelo = LogisticRegression(max_iter=600, random_state=122)
    modelo.fit(X_train, Y_train)
    prediction = modelo.predict(X_test)
    add_acc_to_array(Y_test, prediction)
    add_kfold_to_array(modelo, data, outcome)

    # Decision Tree
    modelo = DecisionTreeClassifier(random_state=122)
    modelo.fit(X_train, Y_train)
    prediction = modelo.predict(X_test)
    add_acc_to_array(Y_test, prediction)
    add_kfold_to_array(modelo, data, outcome)

    # K Neighbors
    modelo = KNeighborsClassifier(n_neighbors=10)
    modelo.fit(X_train, Y_train)
    prediction = modelo.predict(X_test)
    add_acc_to_array(Y_test, prediction)
    add_kfold_to_array(modelo, data, outcome)

    # Naive Bayes
    modelo = BernoulliNB(binarize=True)
    modelo.fit(X_train, Y_train)
    prediction = modelo.predict(X_test)
    add_acc_to_array(Y_test, prediction)
    add_kfold_to_array(modelo, data, outcome)

    #######################################
    #       MATRIZ DE CORRELACION         #
    #######################################

    color = sb.color_palette("coolwarm", as_cmap=True)
    sb.heatmap(diabetes[diabetes.columns[:8]].corr(), annot=True, cmap=color)
    fig = plot.gcf()
    fig.set_size_inches(20, 14)
    plot.title("MATRIZ DE CORRELACIÓN")
    plot.show()

    #######################################
    #       KFOLD CROSS VALIDATION        #
    #######################################
    modelos = ["SVM", "Regresion Logistica", "Decision Tree", "K-Nearest", "Naive Bayes"]
    tabla_kfold = p.DataFrame(index=modelos, columns=["KFold (K = 10)"], data=kfold_validation_array)
    p.options.display.max_columns = 100
    p.options.display.float_format = "{:,.7f}".format
    print("#" * 84)
    print(tabla_kfold)
    print("#" * 84)
    sb.barplot(tabla_kfold.T)
    plot.title("KFold Cross Validation (K = 10)")
    plot.show()

    #######################################
    #       TABLA DATOS PRECISIONES       #
    #######################################
    nombres_precisiones_calculadas = ["Accurancy", "Precision", "Recall", "F1"]
    tabla_precisiones = p.DataFrame(index=nombres_precisiones_calculadas)
    p.options.display.float_format = "{:,.7f}".format
    for data, nombre_col in zip(precisiones_array, modelos):
        tabla_precisiones[nombre_col] = p.array(data=data)
    print("#" * 84)
    print(tabla_precisiones)
    print("#" * 84)

    #######################################
    #      GRAFICO DATOS PRECISIONES      #
    #######################################
    box = p.DataFrame(precisiones_array, index=modelos)

    # print("\n> Precisiones de los metodos")
    # for pre, mod in zip(precisiones_array, modelos):
    #     print(mod)
    #     for fl, npc in zip(pre, nombres_precisiones_calculadas):
    #         print(" " + npc + ": " + "{:.8}".format(fl))
    sb.boxenplot(box.T)
    plot.title("Precisiones métodos de clasificación")
    plot.show()


if __name__ == "__main__":
    main()
