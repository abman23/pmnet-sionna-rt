# PMNet with SionnaRT: Pathloss Map Prediction

## Overview

- This repository provides detailed instructions on the training of **PMNet**—an NN tailored for path loss map prediction. 
- The PMNet are trained with site-specific channel measurement data obtained through **Sionna RT**. 
- The method includes two main steps: 
    1. Generating path loss map dataset by utilizing **Sionna RT**; 
    2. Training **PMNet** with the pathloss map dataset to predict path loss map.

- Example: Ground Truth (**SionnaRT** measurement) vs. Predicted (**PMNet** prediction)

    <img src="figures/Prediction_scene1.png" alt="prediction1" width="600"/> </br>
    <!-- <img src="figures/Prediction_scene2.png" alt="prediction2" width="400"/> -->



## Dataset: SionnaRT-based Pathloss Map (USC Campus Area)


- A 3D map of the USC campus, created with *Blender OSM*, was utilized. These models were then exported to create scenes in **Sionna RT**.

    <img src="figures/View_OpenStreetMap.png" alt="map_USC" height="230"/> &nbsp; 
    <img src="figures/View_Blender.png" alt="blender_3D_USC" height="230"/>
<!-- <img src="figures/CityMap_USC.png" alt="city_map" width="200"/> -->

### Data Pre-Processing

1. **Configuration (TX/RX/Channel/etc.)**:
    - *Details will be updated...*
2. **Map Generation**:
    - For each scene at a specific TX location, three types of maps are generated:
        1. **Pathloss Maps**: These are grayscale images that visualize pathloss (or pathgain) across regions of interest (RoI).
            - Gray conversion: $-200 \sim 0$ [dBm] pathgain $\rightarrow$ $55 \sim 255$ grayscale
        2. **City Maps**: These are grayscale images showing RoI and buildings.
            - Grayscale mapping: $0$ (Black) and $255$ (White) gray value represent building and ROI area, respectively.
        3. **TX Maps**: These are grayscale images indicating the TX locations, which is highlited with $255$ (White) gray value.
3. **Cropping**:
    - Images cropped into 256x256 pixels, ensuring inclusion of TX point and are further augmented.
    - A total of $6455$ cropped images are produced for the USC campus map dataset.

> ***"How to Pre-Process Data?"***
- To pre-process the pathloss map data, simply run the following script. 
Please replace `[START]` and `[END]` with the TX points you want to start and end data mining with. A bigger range will require a lot of memory. A good estimate to have is a range of 5. In order to mine data for all 104 TX, you can run the file updating the `[START]` and `[END]` arguments.

    ```
    python data/preprocess.py [START] [END]
    ```


## Model: PMNet
- To train the PMNet model, we use stacked cropped City and TX maps from the data/cropped folder as input to predict the Pathloss map as the output.

> ***"How to Train?"***
- To train PMNet, simply run the `train.py`.

    ```
    python train.py
    ```
 
> ***"How to Evaluate?"***

- To evaluate a PMNet, refer to the following commands. Please update the path to model for evaluation. Similarly make sure the data is already present in the `data/cropped` folder else follow the above section to prepare the data.

    ```
    python eval.py \
        --model_to_eval '[PATH_TO_MODEL]' 
    ```


### Download: Dataset and Checkpoint

- **Dataset**:
    - **Full Dataset**: [.zip](https://drive.google.com/file/d/1_39J6FnhmVIxsyBDQdCkIbN3cF09h9pz/view?usp=sharing)
    - **Uncropped**: [images](https://drive.google.com/drive/folders/1AHCQtniNpr1DjGMYrWgwxddmQ3IXCgav?usp=drive_link)
    - **Cropped**: [images](https://drive.google.com/drive/folders/1E49AIF7q7LsQWHR68tGV_XJC7ubgplEs?usp=drive_link)
    

- **Checkpoint (Pre-trained PMNet)**:
    - **ckpt (RMSE: 0.00158)**: [pmnet](https://drive.google.com/file/d/1nymEoDKlKGk1aOzm5pNgeTcSE9MG3YGV/view?usp=sharing)
