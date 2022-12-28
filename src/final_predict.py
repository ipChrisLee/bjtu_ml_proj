import config

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
import numpy as np
import pandas as pd
import csv
import json
from io import StringIO

resCSV = """"Coal Electric Power Sector CO2 Emissions",0,SVM,"{'C': 1.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",6,4.34
"Coal Electric Power Sector CO2 Emissions",1,LR,"{}",5,5.82
"Coal Electric Power Sector CO2 Emissions",2,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",4,5.80
"Coal Electric Power Sector CO2 Emissions",3,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",2,6.03
"Coal Electric Power Sector CO2 Emissions",4,SVM,"{'C': 1.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.0001}",4,4.07
"Coal Electric Power Sector CO2 Emissions",5,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",6,4.94
"Natural Gas Electric Power Sector CO2 Emissions",0,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 1e-05}",3,1.00
"Natural Gas Electric Power Sector CO2 Emissions",1,LR,"{}",2,1.45
"Natural Gas Electric Power Sector CO2 Emissions",2,SVM,"{'C': 1.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",2,1.60
"Natural Gas Electric Power Sector CO2 Emissions",3,SVM,"{'C': 2.0, 'max_iter': 100000, 'random_state': 42, 'tol': 1e-05}",5,1.62
"Natural Gas Electric Power Sector CO2 Emissions",4,SVM,"{'C': 2.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.0001}",4,2.20
"Natural Gas Electric Power Sector CO2 Emissions",5,SVM,"{'C': 1.0, 'max_iter': 100000, 'random_state': 42, 'tol': 1e-05}",5,2.13
"Distillate Fuel, Including Kerosene-Type Jet Fuel, Oil Electric Power Sector CO2 Emissions",0,SVM,"{'C': 2.0, 'max_iter': 100000, 'random_state': 42, 'tol': 1e-05}",2,0.47
"Distillate Fuel, Including Kerosene-Type Jet Fuel, Oil Electric Power Sector CO2 Emissions",1,RFR,"{'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 150, 'random_state': 42}",3,0.27
"Distillate Fuel, Including Kerosene-Type Jet Fuel, Oil Electric Power Sector CO2 Emissions",2,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",5,0.17
"Distillate Fuel, Including Kerosene-Type Jet Fuel, Oil Electric Power Sector CO2 Emissions",3,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 1e-05}",5,0.10
"Distillate Fuel, Including Kerosene-Type Jet Fuel, Oil Electric Power Sector CO2 Emissions",4,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.0001}",5,0.10
"Distillate Fuel, Including Kerosene-Type Jet Fuel, Oil Electric Power Sector CO2 Emissions",5,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",6,0.17
"Petroleum Coke Electric Power Sector CO2 Emissions",0,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",5,0.10
"Petroleum Coke Electric Power Sector CO2 Emissions",1,SVM,"{'C': 4.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",2,0.09
"Petroleum Coke Electric Power Sector CO2 Emissions",2,SVM,"{'C': 1.0, 'max_iter': 100000, 'random_state': 42, 'tol': 1e-05}",3,0.15
"Petroleum Coke Electric Power Sector CO2 Emissions",3,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.0001}",3,0.12
"Petroleum Coke Electric Power Sector CO2 Emissions",4,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",4,0.13
"Petroleum Coke Electric Power Sector CO2 Emissions",5,SVM,"{'C': 4.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",4,0.13
"Residual Fuel Oil Electric Power Sector CO2 Emissions",0,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",1,1.31
"Residual Fuel Oil Electric Power Sector CO2 Emissions",1,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",1,1.18
"Residual Fuel Oil Electric Power Sector CO2 Emissions",2,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",1,1.08
"Residual Fuel Oil Electric Power Sector CO2 Emissions",3,SVM,"{'C': 2.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",5,0.88
"Residual Fuel Oil Electric Power Sector CO2 Emissions",4,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 1e-05}",5,1.02
"Residual Fuel Oil Electric Power Sector CO2 Emissions",5,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.0001}",5,1.21
"Petroleum Electric Power Sector CO2 Emissions",0,SVM,"{'C': 1.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",1,1.73
"Petroleum Electric Power Sector CO2 Emissions",1,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 1e-05}",2,1.45
"Petroleum Electric Power Sector CO2 Emissions",2,SVM,"{'C': 1.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",1,1.34
"Petroleum Electric Power Sector CO2 Emissions",3,LR,"{}",5,1.02
"Petroleum Electric Power Sector CO2 Emissions",4,SVM,"{'C': 1.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",3,1.09
"Petroleum Electric Power Sector CO2 Emissions",5,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 1e-05}",6,1.46
"Geothermal Energy Electric Power Sector CO2 Emissions",0,LR,"{}",3,0.00
"Geothermal Energy Electric Power Sector CO2 Emissions",1,LR,"{}",1,0.00
"Geothermal Energy Electric Power Sector CO2 Emissions",2,LR,"{}",3,0.00
"Geothermal Energy Electric Power Sector CO2 Emissions",3,LR,"{}",1,0.00
"Geothermal Energy Electric Power Sector CO2 Emissions",4,LR,"{}",3,0.00
"Geothermal Energy Electric Power Sector CO2 Emissions",5,LR,"{}",1,0.00
"Non-Biomass Waste Electric Power Sector CO2 Emissions",0,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",4,0.02
"Non-Biomass Waste Electric Power Sector CO2 Emissions",1,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",6,0.02
"Non-Biomass Waste Electric Power Sector CO2 Emissions",2,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",4,0.02
"Non-Biomass Waste Electric Power Sector CO2 Emissions",3,SVM,"{'C': 2.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",5,0.02
"Non-Biomass Waste Electric Power Sector CO2 Emissions",4,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",4,0.02
"Non-Biomass Waste Electric Power Sector CO2 Emissions",5,SVM,"{'C': 2.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",5,0.02
"Total Energy Electric Power Sector CO2 Emissions",0,SVM,"{'C': 4.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",2,5.62
"Total Energy Electric Power Sector CO2 Emissions",1,SVM,"{'C': 1.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",5,6.52
"Total Energy Electric Power Sector CO2 Emissions",2,SVM,"{'C': 1.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",2,5.51
"Total Energy Electric Power Sector CO2 Emissions",3,SVM,"{'C': 4.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",2,5.64
"Total Energy Electric Power Sector CO2 Emissions",4,SVM,"{'C': 2.0, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",5,5.70
"Total Energy Electric Power Sector CO2 Emissions",5,SVM,"{'C': 0.5, 'max_iter': 100000, 'random_state': 42, 'tol': 0.001}",6,6.74"""

df: pd.DataFrame = pd.read_excel(config.XLSX_DATA_PATH)
res = dict()
for colInd in df.columns[2:]:
    res[colInd] = [0 for _ in range(6)]
reader = csv.reader(resCSV.split('\n'), delimiter=',')
for row in reader:
    colInd, toPredict, modelName, vargs, x, mae = row
    colInd: str
    toPredict = int(toPredict)
    modelName: str
    vargs = eval(vargs)
    x = int(x)
    mae = float(mae)
    if modelName == "LR":
        model = LinearRegression(**vargs)
    elif modelName == "SVM":
        model = LinearSVR(**vargs)
    elif modelName == "RFR":
        model = RandomForestRegressor(**vargs)
    else:
        exit(-1)

    col: pd.Series = df[colInd].copy()
    col = col.drop(col[col == "Not Available"].index)
    seq = col.values.reshape(-1)
    indAll = [_ for _ in range(len(seq))]
    #   train
    listX = list()
    listY = list()
    i = toPredict
    while i < len(seq):
        if i - toPredict - x < 0:
            i = i + 12
            continue
        listX.append(seq[i - toPredict - x:i - toPredict:1])
        listY.append(seq[i])
        i = i + 12
    X = np.array(listX)
    y = np.array(listY)
    model.fit(X, y)
    predX = np.array([seq[-x - 1:-1]])
    predY = model.predict(predX)
    res[colInd][toPredict] = predY[0]

print(res)
with open(config.TOTAL_RESULT, "w") as fp:
    json.dump(res, fp, indent=4)
