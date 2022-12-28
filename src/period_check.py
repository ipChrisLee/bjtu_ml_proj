#   https://dreamhomes.top/posts/202107271550/
import config

import pandas as pd
from scipy.fftpack import fft, fftfreq
import numpy as np
from statsmodels.tsa.stattools import acf

df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)

for colInd in df.columns[2:]:
    colInd: str
    col: pd.Series = df[colInd].copy()
    col = col.drop(col[col == "Not Available"].index)
    seq: np.ndarray = col.values.reshape(-1)
    nSeq: np.ndarray = np.zeros(seq.shape)
    for i in range(len(seq)):
        nSeq[i] = np.sum(seq[(i // 12 * 12):((i + 12) // 12 * 12)])
    seq = nSeq

    # fft_series = fft(seq)
    # power = np.abs(fft_series)
    # sample_freq = fftfreq(fft_series.size)
    #
    # pos_mask = np.where(sample_freq > 0)
    # freqs = sample_freq[pos_mask]
    # powers = power[pos_mask]
    #
    # top_k_seasons = 8
    # # top K=3 index
    # top_k_idxs = np.argpartition(powers, -top_k_seasons)[-top_k_seasons:]
    # top_k_power = powers[top_k_idxs]
    # fft_periods = (1 / freqs[top_k_idxs]).astype(int)
    #
    # print(f"{colInd} : {fft_periods}")

    # Expected time period
    # for lag in fft_periods:
    #     lag = fft_periods[np.abs(fft_periods - time_lag).argmin()]
    #     acf_score = acf(seq, nlags=lag)[-1]
    #     print(f"lag: {lag} fft acf: {acf_score}")
    acf_score = acf(seq, nlags=12, fft=False)[-1]
    print(f"{colInd[0:5]} & {acf_score:.4f} \\\\")
