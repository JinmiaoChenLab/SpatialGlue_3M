# Deciphering spatial domains from spatial multi-omics with SpatialGlue 
This repository contains the script of SpatialGlue_3M, an extended version of SpatialGlue, tailored for integration of **spatial triplet modality data**. 

For the script of SpatialGlue, please refer to: [https://github.com/JinmiaoChenLab/SpatialGlue/tree/main](https://github.com/JinmiaoChenLab/SpatialGlue/tree/main)

For the step-to-step tutorial of SpatialGlue, please refer to: [https://spatialglue-tutorials.readthedocs.io/en/latest/](https://spatialglue-tutorials.readthedocs.io/en/latest/)
 

![](https://github.com/JinmiaoChenLab/SpatialGlue_3M/blob/main/Workflow.jpg)

## Overview
Integration of multiple data modalities in a spatially informed manner remains an unmet need for exploiting spatial multi-omics data. Here, we introduce SpatialGlue, a novel graph neural network with dual-attention mechanism, to decipher spatial domains by intra-omics integration of spatial location and omics measurement followed by cross-omics integration. We demonstrate that SpatialGlue can more accurately resolve spatial domains at a higher resolution across different tissue types and technology platforms, to enable biological insights into cross-modality spatial correlations. SpatialGlue is computation resource efficient and can be applied for data from various spatial multi-omics technological platforms, including Spatial-epigenome-transcriptome, Stereo-CITE-seq, SPOTS, and 10x Visium. Next, we will extend SpatialGlue to more platforms, such as 10x Genomics Xenium and Nanostring CosMx. 

## Requirements
You'll need to install the following packages in order to run the codes.
* python==3.8
* torch>=1.8.0
* cudnn>=10.2
* numpy==1.22.3
* scanpy==1.9.1
* anndata==0.8.0
* rpy2==3.4.1
* pandas==1.4.2
* scipy==1.8.1
* scikit-learn==1.1.1
* scikit-misc==0.2.0
* tqdm==4.64.0
* matplotlib==3.4.2
* R==4.0.3

## Tutorial
For the step-by-step tutorial, please refer to:
[https://spatialglue-tutorials.readthedocs.io/en/latest/](https://spatialglue-tutorials.readthedocs.io/en/latest/)

## Citation
Yahui Long, Kok Siong Ang, Raman Sethi, Sha Liao, Yang Heng, Lynn van Olst, Shuchen Ye, Chengwei Zhong, Hang Xu, Di Zhang, Immanuel Kwok, Nazihah Husna, Min Jian, Lai Guan Ng, Ao Chen, Nicholas R. J. Gascoigne, David Gate, Rong Fan, Xun Xu & Jinmiao Chen. Deciphering spatial domains from spatial multi-omics with SpatialGlue. **Nature Methods**. 2024.
