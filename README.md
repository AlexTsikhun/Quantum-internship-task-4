# Quantum-internship-task-4
## Soil erosion detection

Here is one of the open problems in Quantum. Soil erosion is a really unwanted process that spoils huge areas of fertile land. Your task is to train model for erosion detection.

### Solution report
Soil erosion is a serious problem common in many regions of the world. Erosion occurs by itself, through the Earth (lithospheric plates in motion), but human activity add impact. Soil erosion depends on the quality and condition of the original soil (where it was located, etc.), on human activity and the slope of the land. The process of soil or rock destruction by water flow, wind, ice, etc. Studies have shown that soil erosion in the forest usually occurred in areas with highly exposed topsoil due to the lack of understory grass cover [2]. Model SEUFM used in research.

From a field experiment conducted in sloping agricultural land, hilly regions of South China, the results showed that compared with slope tillage methods, ridge tillage reduces average annual runoff by 6.11%–64.2%. Used slope tillage and ridge contour tillage in a field study in South China to reduce soil erosion on agricultural land. In both tilled plots, runoff during heavy rainfall was reduced by 10% for the slope tilled plot and 49% for the contour ridge tilled plot [2, 3].

Approaches that prevent erosion:
- Cover Crops: Approach, involves the use of cover crops to prevent soil erosion. Cover crops can protect the soil from erosion by reducing the speed and volume of runoff and providing ground cover [4].
- Terracing: effective method for preventing soil erosion on steep slopes. Terraces can reduce the speed and volume of runoff, which can help prevent erosion [5].

#### References:
[1]A Remote Sensing Based Method to Detect Soil Erosion in Forests.

[2]A systematic review of soil erosion control practices on the agricultural land in Asia.

[3]Analysis of soil erosion characteristics in small watershed of the loess tableland Plateau of China.

[4]Comparative analysis of water budgets across the U.S. Long-Term Agroecosystem Research network.

[5]https://www.youtube.com/watch?v=uEbe2t5iWyQ&ab_channel=AgPhD

### Project Structure:
```
├───README.md                           <- The top-level README for developers using this project
|
├───data                                
|   |───masks                            <- Given me masks.
|   |───patches                   
|   |   |───images                       <- Generated smaller images.
|   |   |───masks                        <- Masks from smaller images.
│   |───T36UXV_20200406T083559_TCI_10m.  <- Oririnal, large image
|   └───train.jp2  
|
├───notebooks                            <- EDA
│   └───patches-images-split.ipynb.ipynb <- Make smaller images (patches)
│   └───soil-erosion-detect-eda.ipynb.ipynb  <- EDA notebook
|
├───models                              
|   └───unet-sm.h5
|
├───large_image_to_patches.py
├───train_model.py                      <- Training
├───predict_model.py                    <- Inference
|
├───.gitignore                          <- Ignore files
|
└───requirements.txt                    <- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`
```
### Solution
