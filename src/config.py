from os.path import exists
import sys
import typing as typ
from pathlib import Path


def verified_failed(reason: str = ""):
    print(f"{reason}\nMake sure you are running on project root!", file=sys.stderr)
    exit(1)


CONFIG_FILE_PATH = Path("src/config.py")
ORG_XLSX_DATA_PATH = Path("data/origin_data.xlsx")
XLSX_DATA_PATH = Path("data/Carbon_Dioxide_Emissions_From_Energy_Consumption__Electric_Power_Sector.xlsx")
VISUALIZE_IMG_FOLDER_PATH = Path("vi_img")

if not CONFIG_FILE_PATH.exists():
    verified_failed(f"{CONFIG_FILE_PATH}")
if not (ORG_XLSX_DATA_PATH.exists() or XLSX_DATA_PATH):
    verified_failed(f"{ORG_XLSX_DATA_PATH} or {XLSX_DATA_PATH}")
if not VISUALIZE_IMG_FOLDER_PATH.exists():
    verified_failed(f"{VISUALIZE_IMG_FOLDER_PATH}")
