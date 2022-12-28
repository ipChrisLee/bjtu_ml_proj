import config

import pandas as pd
from matplotlib import pyplot as plt

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
    pass


if __name__ == '__main__':
    print_cor_img()
