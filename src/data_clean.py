import config
import pandas as pd

df: pd.DataFrame = pd.read_excel(config.ORG_XLSX_DATA_PATH)
df.drop(df.tail(6).index, inplace=True)

with pd.ExcelWriter(config.XLSX_DATA_PATH) as writer:
    df.to_excel(writer)
