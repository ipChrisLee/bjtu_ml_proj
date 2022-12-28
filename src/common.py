import config
import pandas as pd
import typing as typ
import numpy as np


def get_and_split_data(trainFraction: float = 0.7) \
		-> typ.List[typ.Tuple[str, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
	"""
	Usage:
	for colInd, seq, trainRatio, xTrain, yTrain, xTest, yTest in get_and_split_data(0.8):
		pass
	[0:trainRatio] is x-axis of train data, [1:trainRatio + 1] is y-axis of train data.
	[trainRatio + 1:-1] is x-axis of test data, [trainRatio + 2:] is y-axis of test data.
	"""
	res = list()
	df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)
	for colInd in df.columns[2:]:
		#   fetch data and train
		colInd: str
		col: pd.Series = df[colInd].copy()
		col = col.drop(col[col == "Not Available"].index)
		seq = col.values.reshape(-1, 1)
		trainRatio = int(len(seq) * trainFraction)
		xTrain = seq[0:trainRatio]
		yTrain = seq[1:trainRatio + 1]
		xTest = seq[trainRatio + 1:-1]
		yTest = seq[trainRatio + 2:]
		res.append((colInd, seq, trainRatio, xTrain, yTrain, xTest, yTest))
	return res


def get_and_split_mon_data(trainFractionOuter: float = 0.7, trainFractionInner: float = 0.98) -> \
		typ.List[
			typ.Tuple[str, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, typ.Dict[
				int, typ.Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
			]
		]:
	"""
	Usage:
		See below.
	"""
	res = list()
	df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)
	for colInd in df.columns[2:]:
		#   fetch data and train
		colInd: str
		dic = dict()
		col: pd.Series = df[colInd].copy()
		col = col.drop(col[col == "Not Available"].index)
		seq = col.values.reshape(-1, 1)
		for mon in range(12):
			mXSeq = seq[mon:-1:12]
			mYSeq = seq[mon + 1::12]
			assert len(mXSeq) == len(mYSeq)
			trainRatio = int(len(mXSeq) * trainFractionInner)
			xTrain = mXSeq[0:trainRatio]
			yTrain = mYSeq[1:trainRatio + 1]
			xTest = mXSeq[trainRatio + 1:-1]
			yTest = mYSeq[trainRatio + 2:]
			dic[mon] = (trainRatio, xTrain, yTrain, xTest, yTest)
		trainRatio = int(len(seq) * trainFractionOuter)
		xTrain = seq[0:trainRatio]
		yTrain = seq[1:trainRatio + 1]
		xTest = seq[trainRatio + 1:-1]
		yTest = seq[trainRatio + 2:]
		res.append((colInd, seq, trainRatio, xTrain, yTrain, xTest, yTest, dic))
	return res


def get_and_split_mon_m2o_data(trainFractionOuter: float = 0.7, trainFractionInner: float = 0.95) -> \
		typ.List[
			typ.Tuple[str, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, typ.Dict[
				int, typ.Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
			]
		]:
	"""
	Usage:
		See below.
		6 to one
	"""
	res = list()
	df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)
	for colInd in df.columns[2:]:
		#   fetch data and train
		colInd: str
		dic = dict()
		col: pd.Series = df[colInd].copy()
		col = col.drop(col[col == "Not Available"].index)
		seq = col.values.reshape(-1, 1)
		for mon in range(12):
			mXSeq = \
				np.concatenate(
					(
						np.array(seq[mon + 0:-6:12]).reshape(-1, 1),
						np.array(seq[mon + 1:-5:12]).reshape(-1, 1),
						np.array(seq[mon + 2:-4:12]).reshape(-1, 1),
						np.array(seq[mon + 3:-3:12]).reshape(-1, 1),
						np.array(seq[mon + 4:-2:12]).reshape(-1, 1),
						np.array(seq[mon + 5:-1:12]).reshape(-1, 1),
					), axis=1
				)
			mYSeq = seq[mon + 6::12]
			assert len(mXSeq) == len(mYSeq)
			trainRatio = int(len(mXSeq) * trainFractionInner)
			xTrain = mXSeq[0:trainRatio]
			yTrain = mYSeq[1:trainRatio + 1]
			xTest = mXSeq[trainRatio + 1:-1]
			yTest = mYSeq[trainRatio + 2:]
			dic[mon] = (trainRatio, xTrain, yTrain, xTest, yTest)
		trainRatio = int(len(seq) * trainFractionOuter)
		xTrain = seq[0:trainRatio]
		yTrain = seq[1:trainRatio + 1]
		xTest = seq[trainRatio + 1:-1]
		yTest = seq[trainRatio + 2:]
		res.append((colInd, seq, trainRatio, xTrain, yTrain, xTest, yTest, dic))
	return res


def add_mae_score_sum(result: typ.Dict[str, typ.Dict[str, typ.Dict[str, float]]]):
	score = 0
	for colInd in config.COL_NAMES:
		colScore = 0
		for i in range(1, 7):
			colScore = colScore + result[colInd][f"{i} day"]["mae"]
		colScore = colScore / 6
		score = score + colScore * config.LAST_YEAR_AVE[colInd]
	score = score / sum([v for (_, v) in config.LAST_YEAR_AVE.items()])
	result["Score"] = dict()
	result["Score"]["mae"] = score
	return score


def main():
	# for colInd, seqTotal, trainRatioAll, xTrainTotal, yTrainTotal, xTestTotal, yTestTotal, data \
	#         in get_and_split_mon_data(0.7, 0.7):
	#     print(colInd)
	#     print("\t", seqTotal.shape, trainRatioAll, xTrainTotal.shape, yTrainTotal.shape, xTestTotal.shape,
	#           yTestTotal.shape)
	#     for mon, monData in data.items():
	#         trainRatio, xTrain, yTrain, xTest, yTest = monData
	#         print("\t\t", mon)
	#         print("\t\t\t", trainRatio, xTrain.shape, yTrain.shape, xTest.shape, yTest.shape)

	for colInd, seqTotal, trainRatioAll, xTrainTotal, yTrainTotal, xTestTotal, yTestTotal, data \
			in get_and_split_mon_m2o_data(0.7):
		print(colInd)
		print("\t", seqTotal.shape, trainRatioAll, xTrainTotal.shape, yTrainTotal.shape, xTestTotal.shape,
		      yTestTotal.shape)
		for mon, monData in data.items():
			trainRatio, xTrain, yTrain, xTest, yTest = monData
			print("\t\t", mon)
			print("\t\t\t", trainRatio, xTrain.shape, yTrain.shape, xTest.shape, yTest.shape)


if __name__ == '__main__':
	main()
