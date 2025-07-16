# TerrainFloodSense: Improving Seamless Flood Mapping with Cloudy Satellite Imagery via Water Occurrence and Terrain Data Fusion
Reference:

Zhiwei Li, Shaofen Xu, Qihao Weng. TerrainFloodSense: Improving Seamless Flood Mapping with Cloudy Satellite Imagery via Water Occurrence and Terrain Data Fusion. 2025. (In Revision)



### 1. Generation of Enhanced Water Occurrence Data via Multi-Source Data Fusion

**_BayesOccEnhencement_Main.py**

#### 1.1 Data Preparation

**Water Occurrence:** https://global-surface-water.appspot.com/download
**DSM:** https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V2_2
**HAND:** https://gee-community-catalog.org/projects/hand/#resolutions 

Note: Make sure all data are in a uniform data range, projection, and resolution.

#### 1.2 Parameter Description

**Water_Occur_path:** Path to the Water Occurrence data;
**HAND_Path:** Path to the HAND data;
**DEM_Path:** Path to the DEM data;
**wt:** The weight ratio of HAND, the weight ratio of DEM is (1-wt), with the sum of the two weights being 1;
**thr:** Threshold for modifying Water Occurrence values; Water Occurrence values below this threshold will be modified;
**classes:** Classes for resampling the histogram of the original Water Occurrence when performing histogram matching;
**Occ_Bayes:** Whether to save the Water Occurrence calculated using the Bayesian method based on geographic data. If set to None, it will not be saved; if set to a path, it will be saved;
**Occ_Bayes_Matching:** <u>Path to save the enhanced Water Occurrence</u>, which integrates the Water Occurrence calculated based on geographic data with histogram matching applied to the original Water Occurrence.



### 2. Cloud Reconstruction Based on Enhanced Water Occurrence

_Flood_Mapping_HLS_Main.py

#### 2.1 Data Preparation

Preparation of initial multiple single-band data or single multi-band data is consistent with “*Beyond clouds: Seamless flood mapping using Harmonized Landsat and Sentinel-2 time series imagery and water occurrence data*”.https://github.com/dr-lizhiwei/SeamlessFloodMapper

#### 2.2 Parameter Description

**DataPath:** Folder for storing processed data and result files;
**Bands_Folder:** Folder containing multiple single-band data;
**Bandfusion:** Whether multi-band fusion is needed for single-band data;
**RenderHLS:** Whether pseudo-color visualization is needed for multi-band data;
**Fmask2Cloud:** Whether cloud mask data needs to be binary decoded;
**config:** Path to the model parameter settings file for the large flood semantic segmentation model;
**ckpt:** Model weights for the large flood semantic segmentation model;
**bands:** Bands used by the semantic segmentation model;
**SemanticSegment:** Path to save semantic segmentation results;
**Water_Occur_path:** <u>Path to save enhanced Water Occurrence</u>;
**CloudRemoval:** Whether cloud removal operation is needed;
**InitialWaterMaps:** Whether to visualize Initial Water Maps;
**WaterReconstruction:** Method for reconstructing water maps using global or local thresholds;
**ReconstructedWaterMaps:** Whether to visualize Reconstructed Water Maps.

