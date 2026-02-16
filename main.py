# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP conflict on macOS

import sys
from pathlib import Path
from src.gui import VectorLabelerApp

def main():
    tiff_path = Path("/Volumes/Adi-SSD/EfratData/IncuCyte cortex 010823 uptake/A11.tif")
    app = VectorLabelerApp(tiff_path)
    app.launch()

if __name__ == "__main__":
    main()


    