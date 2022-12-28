"""
One 2 One Sep
"""
import numpy as np

from common import add_mae_score_sum
import config

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import typing as typ
import json
from matplotlib import pyplot as plt

modelPathPair = {
	"linear regression": (LinearRegression, config.RES_FOLDER_PATH.O2O_S_LR.MAIN, dict()),
	"random forest regressor": (RandomForestRegressor, config.RES_FOLDER_PATH.O2O_S_RFR.MAIN, dict()),
	"svm": (LinearSVR, config.RES_FOLDER_PATH.O2O_S_SVM.MAIN,
	        {"random_state": config.RANDOM_SEED, "max_iter": 100000, "tol": 1e-4}),
	"lasso": (Lasso, config.RES_FOLDER_PATH.O2O_S_LASSO.MAIN, dict()),
}

d = 1
df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)
for name, (Model_t, MAIN_PATH, vargs) in modelPathPair.items():
	result: typ.Dict[str, typ.Dict[str, typ.Dict[str, float]]] = dict()
	for colInd in df.columns[2:]:
		result[colInd] = dict()

		colInd: str
		col: pd.Series = df[colInd].copy()
		col = col.drop(col[col == "Not Available"].index)
		seq = col.values.reshape(-1)
		indAll = [_ for _ in range(len(seq))]
		#   train
		models: typ.Dict[int, Model_t] = {i: Model_t(**vargs) for i in range(12)}
		for toPredict in range(6):
			listX = list()
			listY = list()
			indToPredict = list()
			i = toPredict
			while i < len(seq):
				if i - toPredict - d < 0:
					i = i + 12
					continue
				listX.append(seq[i - toPredict - d:i - toPredict:1])
				listY.append(seq[i])
				indToPredict.append(i)
				i = i + 12
			X = np.array(listX)
			y = np.array(listY)
			#   split to train and test
			fracTrain = 1
			fracTest = 0.5
			ratioTrain = int(len(X) * fracTrain)
			ratioTest = int(len(X) * fracTest)
			xTrain = X[0:ratioTrain]
			yTrain = y[0:ratioTrain]
			xTest = X[ratioTest:]
			yTest = y[ratioTest:]
			indTest = indToPredict[ratioTest:]
			#   create model
			model = Model_t(**vargs)
			model.fit(xTrain, yTrain)
			yPred = model.predict(xTest)
			# conclude result
			if True:
				resThis: typ.Dict[str, float] = dict()
			resThis["mse"] = mean_squared_error(yTest, yPred)
			resThis["mae"] = mean_absolute_error(yTest, yPred)
			result[colInd][f"{toPredict + 1} day"] = resThis
			#   print img
			imgFilePath = MAIN_PATH / f"vi_img_for_{toPredict + 1}_day_pred"
			imgFilePath.mkdir(exist_ok=True)
			if True:
				plt.subplot(211)
			plt.plot(indTest, yTest)
			plt.plot(indTest, yPred, "o")
			plt.title(f"{colInd[:20]}... (mse={resThis['mse']:.2f}, mae={resThis['mae']:.2f})")
			plt.subplot(212)
			plt.plot(indAll, seq)
			plt.plot(indTest, yPred, "o")
			plt.savefig(imgFilePath / f"{colInd}.png")
			plt.clf()

	add_mae_score_sum(result)
	with open(MAIN_PATH / "res.json", "w") as fp:
		fp.write(json.dumps(result, indent=4))
