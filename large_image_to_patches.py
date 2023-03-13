import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


raster_path = "data/T36UXV_20200406T083559_TCI_10m.jp2"

mask_path ="data/train.jp2"

patch_size = 256

with rasterio.open(raster_path, "r", driver="JP2OpenJPEG") as src:
    raster_img = src.read()
    raster_meta = src.meta



with rasterio.open(raster_path) as src:
    width = src.width
    height = src.height
    meta = src.meta

    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            # Define the window to extract the patch
            window = Window(i, j, patch_size, patch_size)

            patch = src.read(window=window)

            patch_meta = meta.copy()
            patch_meta.update({
                'width': patch.shape[-1],
                'height': patch.shape[-2],
                'transform': rasterio.windows.transform(window, src.transform)
            })

            # Save the patch image to a new file
            with rasterio.open(f'data/patches/images/image_{i}_{j}.jp2', 'w', **patch_meta) as dst:
                dst.write(patch)



with rasterio.open(mask_path) as src:
    width = src.width
    height = src.height
    meta = src.meta

    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            # Define the window to extract the patch
            window = Window(i, j, patch_size, patch_size)

            patch = src.read(window=window)

            patch_meta = meta.copy()
            patch_meta.update({
                'width': patch.shape[-1],
                'height': patch.shape[-2],
                'transform': rasterio.windows.transform(window, src.transform)
            })

            # Save the patch image to a file
            with rasterio.open(f'data/patches/masks/mask_{i}_{j}.jp2', 'w', **patch_meta) as dst:
                dst.write(patch)