# Alignment and Projection of scRNA-seq and Spatial Transcriptomics in Whole Transcriptome Space
FAT is proposed as a conformable alignment and projection high throughput transcript gene expression prediction method. It deciphers both shared and whole space single-cell spatial data alignment and projection across species, paired/unpaired datasets, and sample conditions. 

![FAT workflow](https://github.com/qingli7/FAT/blob/main/FAT_workflow.jpg?raw=true)


# Installation
The requirements of FAT can be installed by:  
`pip install -r requirements.txt`

# Useage
FAT provides some tasks about spatial imputation and projection. 
Run 'spatial_projection.py' for the spatial projection demo on mouse datasets. You can replace the datasets with other data divided into train and test datasets.

Spatial imputation with foundation model adopt CellPLM in FAT can run 'spatial_imputation.py' for human datasets liver and lung. If you want to experiment with other data, you can choose other pre-trained foundation models to replace CellPLM.

