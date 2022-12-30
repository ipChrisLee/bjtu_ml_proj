import config

import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)


def print_cor_img():
	for colIndX in df.columns[2:]:
		for colIndY in df.columns[2:]:
			if colIndX in config.MISSING_DATA_COL_NAMES or colIndY in config.MISSING_DATA_COL_NAMES or colIndX == colIndY:
				continue
			plt.title(f"{colIndX[0:5]}-{colIndY[0:5]}")
			plt.plot(df[colIndX].values, df[colIndY].values, "b+")
			plt.savefig(config.VIS_CORRELATION_FOLDER_PATH / f"{colIndX[0:5]}-{colIndY[0:5]}")
			plt.clf()


def cor_mat():
	with open(config.COR_MAT_FILE_PATH, "w") as fp:
		res = "X"
		for colIndX in df.columns[2:]:
			if colIndX in config.MISSING_DATA_COL_NAMES:
				continue
			res = res + f",{colIndX[0:5]}"
		res = res + "\n"
		for colIndY in df.columns[2:]:
			if colIndY in config.MISSING_DATA_COL_NAMES:
				continue
			res = res + f"{colIndY[0:5]}"
			for colIndX in df.columns[2:]:
				if colIndX in config.MISSING_DATA_COL_NAMES:
					continue
				res = res + f",{pearsonr(df[colIndY].values, df[colIndX].values)[0]:.4f}"
			res = res + "\n"
		fp.write(res)


if __name__ == '__main__':
	print_cor_img()
	cor_mat()
