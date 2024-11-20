# Alignment and Projection of scRNA-seq and Spatial Transcriptomics in Whole Transcriptome Space
FAT is proposed as a conformable alignment and projection high throughput transcript gene expression prediction method. It deciphers both shared and whole space single-cell spatial data alignment and projection across species, paired/unpaired datasets, and sample conditions. 

![FAT workflow](https://github.com/qingli7/FAT/blob/main/FAT_workflow.jpg?raw=true)


# Installation
The requirements of FAT can be installed by:  
`pip install -r requirements.txt`

# Useage
FAT provides some tasks related to spatial imputation and projection. 
Run `spatial_projection.py` for the spatial projection demo on [mouse datasets](https://www.nature.com/articles/s41592-022-01480-9). You can replace the datasets with other data divided into train and test datasets.

Spatial imputation with foundation model adopt CellPLM in FAT can run `spatial_imputation.py` for human datasets [Liver](https://info.vizgen.com/ffpe-showcase?submissionGuid=88ba0a44-26e2-47a2-8ee4-9118b9811fbf) and [Lung](https://info.vizgen.com/ffpe-showcase?submissionGuid=88ba0a44-26e2-47a2-8ee4-9118b9811fbf). If you want to experiment with other data, you can choose other pre-trained foundation models to replace CellPLM.

Batch effect demo uses [human brain cell dataset Zhuang](https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-1.html) and can be run by `spatial_batch_effect.py`.
