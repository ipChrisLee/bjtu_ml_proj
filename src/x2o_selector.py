"""
One 2 One Sep
"""
import numpy as np

import config
from common import *

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
import typing as typ
import json
from matplotlib import pyplot as plt
from pathlib import Path

df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)


def get_X_y(colInd: str, toPredict: int, x: int):
    colInd: str
    col: pd.Series = df[colInd].copy()
    col = col.drop(col[col == "Not Available"].index)
    seq = col.values.reshape(-1)
    listX = list()
    listY = list()
    indToPredict = list()
    i = toPredict
    while i < len(seq):
        if i - toPredict - x < 0:
            i = i + 12
            continue
        listX.append(seq[i - toPredict - x:i - toPredict:1])
        listY.append(seq[i])
        indToPredict.append(i)
        i = i + 12
    X = np.array(listX)
    y = np.array(listY)
    return X, y


def main():
    res: str = "colInd,toPredict,modelName,args,x,mae\n"
    modelInfo = {
        "LR": (LinearRegression, {}),
        "RFR": (RandomForestRegressor, {
            "n_estimators": [50, 75, 100, 125, 150],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 4, 8],
            "random_state": [42],
        }),
        "SVM": (
            LinearSVR, {
                "tol": [1e-3, 1e-4, 1e-5],
                "C": [0.5, 1.0, 2.0, 4.0],
                "max_iter": [100000],
                "random_state": [42],
            }
        )
    }
    for colInd in df.columns[2:]:
        for toPredict in range(6):
            #   for model of colInd, toPredict
            bestModelName: str = ""
            bestParam = dict()
            bestScore = -np.inf
            bestX = -1
            for x in range(1, 7):
                (X, y) = get_X_y(colInd, toPredict, x)
                for modelName, (Model_t, gridPara) in modelInfo.items():
                    gs = GridSearchCV(Model_t(), gridPara, scoring="neg_mean_absolute_error", cv=5)
                    gs.fit(X, y)
                    if bestScore < gs.best_score_:
                        bestModelName = modelName
                        bestParam = gs.best_params_
                        bestScore = gs.best_score_
                        bestX = x
            res = res + f"\"{colInd}\",{toPredict},{bestModelName},\"{bestParam}\",{bestX},{-bestScore:.2f}\n"
    with open(config.RES_FOLDER_PATH.X2M_SEL.MAIN / "result.csv", "w") as fp:
        fp.write(res)


if __name__ == '__main__':
    main()
