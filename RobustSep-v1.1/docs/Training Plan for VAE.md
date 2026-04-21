# Training Plan for VAE (RobustSep)

### *Primary Datasets:*

1\) SVGDataSet (vector-origin graphics): [https://share.google/xBtAOfzq6wcuoSHAo](https://share.google/xBtAOfzq6wcuoSHAo)

The dataset contains vector-origin graphics with solid colors and clean edges, which closely match how the pipeline processes images in small patches. These graphics are visually similar to packaging artwork.  
**For the training configuration** \- SVG images are rasterized and split into small RGB patches.The VAE is trained to generate multiple valid CMYK separations per patch, conditioned on structure and printing parameters.

2\) Typography/layout dataset (DocLayNet): [https://github.com/DS4SD/DocLayNet](https://github.com/DS4SD/DocLayNet)

3\) SKU-110K (shelf images): [https://github.com/eg4000/SKU110K\_CVPR19](https://github.com/eg4000/SKU110K_CVPR19)

4\) RGB cube dataset (16,777,216 colors): [https://www.kaggle.com/datasets/shuvokumarbasak4004/rgb-color-dataset-16777216-colors](https://www.kaggle.com/datasets/shuvokumarbasak4004/rgb-color-dataset-16777216-colors)

To be used as a small supplement to ensure coverage of difficult colors like neons, deep darks, near-neutral ramps, while the main RGB corpus still comes from SVG/template typography generation plus real photos

