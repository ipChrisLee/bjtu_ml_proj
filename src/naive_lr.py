"""
Naive linear regression
只是非常简单地从当月数据预测下个月数据。
"""

import config
from common import *

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import typing as typ
import json
from matplotlib import pyplot as plt
from pathlib import Path

df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)
result: typ.Dict[str, typ.Dict[str, typ.Dict[str, float]]] = dict()

for (colInd, seq, trainRatio, xTrain, yTrain, xTest, yTest) in get_and_split_data(0.7):
    #   TODO:这里部分内容可以参考后续的代码修改。
    lr = LinearRegression()
    lr.fit(xTrain, yTrain)
    result[colInd] = dict()
    for d in range(1, 7):
        yPred = np.zeros(yTest.shape)
        for i in range(len(yPred)):
            yy = lr.predict([seq[trainRatio + 2 + i - d]])
            for t in range(d - 1):
                yy = lr.predict(yy)
            yPred[i] = yy[0]
        # conclude result
        if True:
            resThis: typ.Dict[str, float] = dict()
            resThis["mse"] = mean_squared_error(yTest, yPred)
            resThis["mae"] = mean_absolute_error(yTest, yPred)
            result[colInd][f"{d} day"] = resThis
        #   print img
        imgFilePath = config.RES_FOLDER_PATH.NAIVE_LR.MAIN / f"vi_img_for_{d}_day_pred"
        imgFilePath.mkdir(exist_ok=True)
        if True:
            idx = [i for i in range(len(seq))]
            # plt.plot(idx, seq)
            plt.subplot(211)
            plt.plot(idx[trainRatio + 2:], yTest)
            plt.plot(idx[trainRatio + 2:], yPred)
            plt.title(f"{colInd[:20]}... (mse={resThis['mse']:.2f}, mae={resThis['mae']:.2f})")
            plt.subplot(212)
            plt.plot(idx, seq)
            plt.plot(idx[trainRatio + 2:], yPred)
            plt.savefig(imgFilePath / f"{colInd}.png")
            plt.clf()

add_mae_score_sum(result)
with open(config.RES_FOLDER_PATH.NAIVE_LR.MAIN / "res.json", "w") as fp:
    fp.write(json.dumps(result, indent=4))
