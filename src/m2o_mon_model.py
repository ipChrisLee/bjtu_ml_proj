"""
Multiple(6) 2 One Monthly Linear Regression
"""

import config
from common import *

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import typing as typ
import json
from matplotlib import pyplot as plt
from pathlib import Path

modelPathPair = {
    "linear regression": (LinearRegression, config.RES_FOLDER_PATH.M2O_M_LR.MAIN, dict()),
    "random forest regressor": (RandomForestRegressor, config.RES_FOLDER_PATH.M2O_M_RFR.MAIN, dict()),
    "svm": (LinearSVR, config.RES_FOLDER_PATH.M2O_M_SVM.MAIN,
            {"random_state": config.RANDOM_SEED, "max_iter": 100000, "tol": 1e-3}),
}
for name, (Model_t, MAIN_PATH, vargs) in modelPathPair.items():
    result: typ.Dict[str, typ.Dict[str, typ.Dict[str, float]]] = dict()

    for colInd, seqTotal, trainRatioAll, xTrainTotal, yTrainTotal, xTestTotal, yTestTotal, data \
            in get_and_split_mon_m2o_data(0.5):
        #   train
        result[colInd] = dict()
        models: typ.Dict[int, Model_t] = {i: Model_t(**vargs) for i in range(12)}
        # models: typ.Dict[int, RandomForestRegressor] = {i: RandomForestRegressor() for i in range(12)}
        for mon, monData in data.items():
            trainRatio, xTrain, yTrain, xTest, yTest = monData
            models[mon].fit(xTrain, yTrain)
        #   metric
        for d in range(1, 7):
            yPred = np.zeros(yTestTotal.shape)
            for i in range(len(yPred)):
                mon = trainRatioAll + 2 + i - d - 5
                xx = np.array([seqTotal[mon + j][0] for j in range(6)])
                yy = np.array([[0]])
                for t in range(0, d):
                    yy = models[(mon + t) % 12].predict(xx.reshape(1, -1))
                    # print(yy)
                    xx = [xx[j - 1] for j in range(1, 6)] + [yy.reshape(-1)[0]]
                    xx = np.array(xx)
                yPred[i] = yy[0]
            # conclude result
            if True:
                resThis: typ.Dict[str, float] = dict()
                resThis["mse"] = mean_squared_error(yTestTotal, yPred)
                resThis["mae"] = mean_absolute_error(yTestTotal, yPred)
                result[colInd][f"{d} day"] = resThis
            #   print img
            imgFilePath = MAIN_PATH / f"vi_img_for_{d}_day_pred"
            imgFilePath.mkdir(exist_ok=True)
            if True:
                idx = [i for i in range(len(seqTotal))]
                # plt.plot(idx, seq)
                plt.subplot(211)
                plt.plot(idx[trainRatioAll + 2:], yTestTotal)
                plt.plot(idx[trainRatioAll + 2:], yPred)
                plt.title(f"{colInd[:20]}... (mse={resThis['mse']:.2f}, mae={resThis['mae']:.2f})")
                plt.subplot(212)
                plt.plot(idx, seqTotal)
                plt.plot(idx[trainRatioAll + 2:], yPred)
                plt.savefig(imgFilePath / f"{colInd}.png")
                plt.clf()

    add_mae_score_sum(result)
    with open(MAIN_PATH / "res.json", "w") as fp:
        fp.write(json.dumps(result, indent=4))
