# TransTagger
The source code used for A Transformer-based Framework for POI-level Social Post Geolocation, published in ECIR 2023.

The dataset can be downloaded here: [TransTagger](https://sutdapac-my.sharepoint.com/:f:/g/personal/menglin_li_mymail_sutd_edu_sg/ElNN6uGzFDVFuJHHcBwR5VkBdLu4gSOnN_aGPq78_q5Fhw?e=nhGY5W).

Our Arxiv Paper can be found here: [A Transformer-based Framework for POI-level Social Post Geolocation](https://arxiv.org/abs/2211.01336).
## Requirements
tensorflow=2.6.0

keras_bert=0.89.0

keras_pos_embed 0.13.0
## Train model
We provide scripts to train our model, with all variants presented in the paper. You can change the model type, dataset, and representation methods accordingly. 
Before running the code, please download our dataset and [BERT checkpoint files](https://github.com/google-research/bert), then set the file paths correctly.
```shell
python run.py
```

## Citations
Please cite the following paper if you find the code and dataset helpful for your research.
```bib
@misc{https://doi.org/10.48550/arxiv.2211.01336,
  doi = {10.48550/ARXIV.2211.01336},
  url = {https://arxiv.org/abs/2211.01336},
  author = {Li, Menglin and Lim, Kwan Hui and Guo, Teng and Liu, Junhua},
  keywords = {Information Retrieval (cs.IR), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {A Transformer-based Framework for POI-level Social Post Geolocation},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
