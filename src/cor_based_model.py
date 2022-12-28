"""
One 2 One Sep
"""
import numpy as np

import config

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
import pandas as pd
import typing as typ
import json
from matplotlib import pyplot as plt
from pathlib import Path

df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)


def get_combined_X_y(colInd1: str, colInd2: str, toPredict: int, x: int):
    seq1: np.ndarray = df[colInd1].copy().values.reshape(-1)
    seq2: np.ndarray = df[colInd2].copy().values.reshape(-1)
    listX1 = list()
    listX2 = list()
    listY1 = list()
    listY2 = list()
    indToPredict = list()
    i = toPredict
    while i < len(seq1):
        if i - toPredict - x < 0:
            i = i + 12
            continue
        listX1.append(seq1[i - toPredict - x:i - toPredict:1])
        listX2.append(seq2[i - toPredict - x:i - toPredict:1])
        listY1.append(seq1[i])
        listY2.append(seq2[i])
        indToPredict.append(i)
        i = i + 12
    X = np.concatenate((np.array(listX1), np.array(listX2)), axis=1)
    X = scale(X)
    y = {
        colInd1: np.array(listY1),
        colInd2: np.array(listY2),
    }
    return X, y


def main_strong_linear_correlation(
        colInd1: str = "Residual Fuel Oil Electric Power Sector CO2 Emissions",
        colInd2: str = "Petroleum Electric Power Sector CO2 Emissions"
):
    res: str = "colInd,toPredict,modelName,args,x,mae\n"
    modelInfo = {
        "LR": (LinearRegression, {}),
        "RFR": (RandomForestRegressor, {
            "n_estimators": [75, 100, 125],
            "max_depth": [None, 5],
            "min_samples_split": [2, 4],
            "random_state": [42],
        }),
        "SVM": (
            LinearSVR, {
                "tol": [1e-3, 1e-4],
                "C": [0.5, 1.0, 2.0, 4.0],
                "max_iter": [100000],
                "random_state": [42],
            }
        )
    }
    for toPredict in range(6):
        #   for model of colInd, toPredict
        bestModelName: typ.Dict[str, str] = {colInd1: "", colInd2: ""}
        bestParam = {colInd1: dict(), colInd2: dict()}
        bestScore = {colInd1: -np.inf, colInd2: -np.inf}
        bestX = {colInd1: -1, colInd2: -1}
        for x in range(1, 7):
            (X, y) = get_combined_X_y(colInd1, colInd2, toPredict, x)
            for modelName, (Model_t, gridPara) in modelInfo.items():
                gs = GridSearchCV(Model_t(), gridPara, scoring="neg_mean_absolute_error", cv=5)
                gs.fit(X, y[colInd1])
                if bestScore[colInd1] < gs.best_score_:
                    bestModelName[colInd1] = modelName
                    bestParam[colInd1] = gs.best_params_
                    bestScore[colInd1] = gs.best_score_
                    bestX[colInd1] = x
            for modelName, (Model_t, gridPara) in modelInfo.items():
                gs = GridSearchCV(Model_t(), gridPara, scoring="neg_mean_absolute_error", cv=5)
                gs.fit(X, y[colInd2])
                if bestScore[colInd2] < gs.best_score_:
                    bestModelName[colInd2] = modelName
                    bestParam[colInd2] = gs.best_params_
                    bestScore[colInd2] = gs.best_score_
                    bestX[colInd2] = x
        res = res + f"\"{colInd1}\",{toPredict},{bestModelName[colInd1]},\"{bestParam[colInd1]}\",{bestX[colInd1]},{-bestScore[colInd1]:.2f}\n"
        res = res + f"\"{colInd2}\",{toPredict},{bestModelName[colInd2]},\"{bestParam[colInd2]}\",{bestX[colInd2]},{-bestScore[colInd2]:.2f}\n"
    with open(config.RES_FOLDER_PATH.COR_BASED_MODEL.MAIN / "result.csv", "w") as fp:
        fp.write(res)


if __name__ == '__main__':
    #   Resid-Natur
    # main_strong_linear_correlation(
    #     "Residual Fuel Oil Electric Power Sector CO2 Emissions",
    #     "Natural Gas Electric Power Sector CO2 Emissions",
    # )
    #   Natur-Coal
    main_strong_linear_correlation(
        "Natural Gas Electric Power Sector CO2 Emissions",
        "Coal Electric Power Sector CO2 Emissions",
    )
