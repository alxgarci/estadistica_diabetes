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


def main():
    diabetes = p.read_csv("diabetes.csv")
    print("#" * 84)
    print(">Filas totales de datos: " + str(len(diabetes.index)))
    print(diabetes.head())

    #######################################
    #          PREVISUALIZACION           #
    #######################################

    # diabetes[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = diabetes[
    #     ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan)


if __name__ == '__main__':
    main()
