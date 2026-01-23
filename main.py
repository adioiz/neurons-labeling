# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP conflict on macOS

import sys
from pathlib import Path
from src.gui import VectorLabelerApp

def main():
    tiff_path = Path("/Volumes/Adi-SSD/EfratData/דאטה מסודר לאלון/IncuCyte cortex 180723/left/distance in pixel 300, known distance 200,pixel aspect ratio 1, unit of length micron, 1.5 pixel to micron/צפיפות/tif/A91 C1.tif")
    app = VectorLabelerApp(tiff_path)
    app.launch()

if __name__ == "__main__":
    main()


    