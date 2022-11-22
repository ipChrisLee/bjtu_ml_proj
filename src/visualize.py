import config
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)
print(df)
print(df.columns)
for colInd in df.columns[2:-1]:
    colInd: str
    col: pd.Series = df[colInd].copy()
    col[col == "Not Available"] = 0
    plt.title(colInd)
    plt.plot(col)
    plt.savefig(config.VISUALIZE_IMG_FOLDER_PATH / Path(f"{colInd}.png"))
    plt.clf()
