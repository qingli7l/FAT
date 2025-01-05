# Alignment and Projection of scRNA-seq and Spatial Transcriptomics in Whole Transcriptome Space
FAT (Foundation Model Large Space Augmented Whole Transcriptome Spatial Gene Imputation) is a conformable alignment and projection high throughput transcript gene expression prediction method. It deciphers both shared and whole space single-cell spatial data alignment and projection across species, paired/unpaired datasets, and sample conditions. 

![FAT workflow](https://github.com/qingli7/FAT/blob/main/FAT_workflow.jpg?raw=true)


# Installation
The requirements of FAT can be installed by:  
```
conda create -n fat python=3.9 -y
pip install -r requirements.txt
```

# Useage
FAT provides some tasks related to spatial imputation and projection. 

1. Run `spatial_projection.py` for the spatial projection demo on [mouse datasets](https://www.nature.com/articles/s41592-022-01480-9). You can replace the datasets with other data divided into train and test datasets.

2. Spatial imputation can run `spatial_imputation.py` for human datasets [Liver](https://info.vizgen.com/ffpe-showcase?submissionGuid=88ba0a44-26e2-47a2-8ee4-9118b9811fbf) and [Lung](https://info.vizgen.com/ffpe-showcase?submissionGuid=88ba0a44-26e2-47a2-8ee4-9118b9811fbf). If you want to experiment with other data, you can choose other pre-trained foundation models on the data to replace CellPLM adopted in FAT. The environment installation of spatial imputation is consistent with the environment of the selected foundation model.

3. Batch effect demo trains on [human brain cell dataset Zhuang](https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-1.html) with `spatial_batch_effect.py`.

# Acknowledgement
The code is partially adapted from [CellPLM](https://github.com/OmicsML/CellPLM).
