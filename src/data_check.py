"""
验证数据是否满足一些性质，包括：
1. 所有的月份都是连着的。
    这玩意之间看下数量就行。。。有两个数据是1989年开始全的，另外的都是1973年开始全的。
2. 数据的"Total ... "项目是其他项目的和。
    不是。
    英语不好的锅，那玩意翻译过来也不是"总和"的意思。

"""

import config

import pandas as pd
import numpy as np

df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)


def check_total_is_sum():
    sumOfAll = df[config.COL_NAMES[-1]].values.copy()
    for colInd in df.columns[2:]:
        if colInd == config.COL_NAMES[-1]:
            continue
        #   fetch data and train
        colInd: str
        col: pd.Series = df[colInd].copy()
        col[col == "Not Available"] = 0.0
        seq = col.values
        sumOfAll = sumOfAll - seq
    print(sumOfAll)
    print(sum(sumOfAll))


check_total_is_sum()
