import config
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def print_to_multi_files():
    df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)
    print(df)
    print(df.columns)
    for colInd in df.columns[2:]:
        colInd: str
        col: pd.Series = df[colInd].copy()
        col = col.drop(col[col == "Not Available"].index)
        #   origin data vis
        plt.title(colInd)
        plt.plot(col)
        plt.savefig(config.VIS_ORG_DATA_FOLDER_PATH / f"{colInd}.png")
        plt.clf()
        #   pre-nxt data vis
        plt.title(colInd)
        plt.plot(col.values[0:-1], col.values[1:], "bo")
        plt.savefig(config.VIS_PRE_NXT_DATA_FOLDER_PATH / f"{colInd}.png")
        plt.clf()
        #   pre-nxt data for every month vis
        for d in range(0, 12):
            plt.title(colInd)
            plt.plot(col.values[d:-1:12], col.values[d + 1::12], "bo")
            monFolderPath = config.VIS_PRE_NXT_PER_MON_FOLDER_PATH / f"{d + 1}"
            monFolderPath.mkdir(exist_ok=True)
            plt.savefig(monFolderPath / f"{colInd}.png")
            plt.clf()
        #   pre-nxt data for total
        if True:
            plt.title(colInd)
            for d in range(0, 12):
                plt.plot(col.values[d:-1:12], col.values[d + 1::12], "+")
            totalFolderPath = config.VIS_PRE_NXT_PER_MON_FOLDER_PATH / "total"
            totalFolderPath.mkdir(exist_ok=True)
            plt.savefig(totalFolderPath / f"{colInd}.png")
            plt.clf()


def print_to_single_img():
    df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)
    idx = 1
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for colInd in df.columns[2:]:
        colInd: str
        col: pd.Series = df[colInd].copy()
        col = col.drop(col[col == "Not Available"].index)
        #   origin data vis
        plt.subplot(330 + idx)
        plt.title(colInd[0:5])
        plt.plot(col)
        idx = idx + 1
    plt.savefig(config.VIS_ORG_DATA_FOLDER_PATH / f"all_in_one.png")
    plt.clf()
    idx = 1
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for colInd in df.columns[2:]:
        colInd: str
        col: pd.Series = df[colInd].copy()
        col = col.drop(col[col == "Not Available"].index)
        #   origin data vis
        plt.subplot(330 + idx)
        plt.title(colInd[0:5])
        plt.plot(col.iloc[0:-1], col.iloc[1:], "b*")
        idx = idx + 1
    plt.savefig(config.VIS_PRE_NXT_DATA_FOLDER_PATH / f"pre_nxt_all_in_one.png")


if __name__ == '__main__':
    print_to_single_img()
