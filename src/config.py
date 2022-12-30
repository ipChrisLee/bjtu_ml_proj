import sys
from pathlib import Path


def verified_failed(reason: str = ""):
	print(f"Config init failed.\nReason : [{reason}]\nMake sure you are running on project root!", file=sys.stderr)
	exit(1)


RANDOM_SEED = 42

#   file paths
CONFIG_FILE_PATH = Path("src/config.py")
ORG_XLSX_DATA_PATH = Path("data/origin_data.xlsx")
XLSX_DATA_PATH = Path("data/Carbon_Dioxide_Emissions_From_Energy_Consumption__Electric_Power_Sector.xlsx")
#   Visualize original data
VIS_ORG_DATA_FOLDER_PATH = Path("vi_img/visual_data")
#   Visualize pre-nxt data
VIS_PRE_NXT_DATA_FOLDER_PATH = Path("vi_img/pre_nxt")
#   Visualize pre-nxt data for every month
VIS_PRE_NXT_PER_MON_FOLDER_PATH = Path("vi_img/pre_nxt_pm")
#
VIS_CORRELATION_FOLDER_PATH = Path("vi_img/correlation")
#
TOTAL_RESULT = Path("result/total_res.json")
#
COR_MAT_FILE_PATH = Path("result/cor_mat.csv")


class RES_FOLDER_PATH:
	class NAIVE_LR:
		MAIN = Path("result/naive_lr")

	class NAIVE_RFR:
		MAIN = Path("result/naive_rfr")

	class O2O_M_LR:
		MAIN = Path("result/o2o_mon_lr")

	class O2O_M_RFR:
		MAIN = Path("result/o2o_mon_rfr")

	class O2O_M_SVM:
		MAIN = Path("result/o2o_mon_svm")

	class M2O_M_LR:
		MAIN = Path("result/m2o_mon_lr")

	class M2O_M_RFR:
		MAIN = Path("result/m2o_mon_rfr")

	class M2O_M_SVM:
		MAIN = Path("result/m2o_mon_svm")

	class M2O_S_LR:
		MAIN = Path("result/m2o_sep_lr")

	class M2O_S_RFR:
		MAIN = Path("result/m2o_sep_rfr")

	class M2O_S_SVM:
		MAIN = Path("result/m2o_sep_svm")

	class M2O_S_LASSO:
		MAIN = Path("result/m2o_sep_lasso")

	class O2O_S_LR:
		MAIN = Path("result/o2o_sep_lr")

	class O2O_S_RFR:
		MAIN = Path("result/o2o_sep_rfr")

	class O2O_S_SVM:
		MAIN = Path("result/o2o_sep_svm")

	class O2O_S_LASSO:
		MAIN = Path("result/o2o_sep_lasso")

	class X2M_SEL:
		MAIN = Path("result/x2o_sel")

	class COR_BASED_MODEL:
		MAIN = Path("result/cor_based_model")


#   index name
COL_NAMES = [
	"Coal Electric Power Sector CO2 Emissions",
	"Natural Gas Electric Power Sector CO2 Emissions",
	"Distillate Fuel, Including Kerosene-Type Jet Fuel, Oil Electric Power Sector CO2 Emissions",
	"Petroleum Coke Electric Power Sector CO2 Emissions",
	"Residual Fuel Oil Electric Power Sector CO2 Emissions",
	"Petroleum Electric Power Sector CO2 Emissions",
	"Geothermal Energy Electric Power Sector CO2 Emissions",
	"Non-Biomass Waste Electric Power Sector CO2 Emissions",
	"Total Energy Electric Power Sector CO2 Emissions",
]
MISSING_DATA_COL_NAMES = [
	"Geothermal Energy Electric Power Sector CO2 Emissions",
	"Non-Biomass Waste Electric Power Sector CO2 Emissions",
]

LAST_YEAR_AVE = {
	"Coal Electric Power Sector CO2 Emissions": 75.77,
	"Natural Gas Electric Power Sector CO2 Emissions": 51.27,
	"Distillate Fuel, Including Kerosene-Type Jet Fuel, Oil Electric Power Sector CO2 Emissions": 0.3283,
	"Petroleum Coke Electric Power Sector CO2 Emissions": 0.7323,
	"Residual Fuel Oil Electric Power Sector CO2 Emissions": 0.3615,
	"Petroleum Electric Power Sector CO2 Emissions": 1.422,
	"Geothermal Energy Electric Power Sector CO2 Emissions": 0.03933,
	"Non-Biomass Waste Electric Power Sector CO2 Emissions": 0.8752,
	"Total Energy Electric Power Sector CO2 Emissions": 129.4,
}

if not CONFIG_FILE_PATH.exists():
	verified_failed(f"{CONFIG_FILE_PATH}")
