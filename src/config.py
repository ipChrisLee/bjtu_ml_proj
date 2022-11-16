from os.path import isfile
import sys

CONFIG_FILE_PATH = "src/config.py"
if not isfile(CONFIG_FILE_PATH):
    print("You should run on project root!", file=sys.stderr)
    exit(1)

XLSX_DATA_PATH = "Carbon_Dioxide_Emissions_From_Energy_Consumption__Electric_Power_Sector.xlsx"
