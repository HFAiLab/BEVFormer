<div align="center">   
  
# BEVFormer: a Cutting-edge Baseline for Camera-based Detection
</div>


https://user-images.githubusercontent.com/27915819/161392594-fc0082f7-5c37-4919-830a-2dd423c1d025.mp4

> **BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers**
> - [Paper](http://arxiv.org/abs/2203.17270) | [Blog](https://www.zhihu.com/question/521842610/answer/2431585901) (in Chinese) | Presentation Slides at CVPR 2022 Workshop (soon) | Live-streaming video on BEV Perception (soon)



# Abstract
In this work, the authors present a new framework termed BEVFormer, which learns unified BEV representations with spatiotemporal transformers to support multiple autonomous driving perception tasks. In a nutshell, BEVFormer exploits both spatial and temporal information by interacting with spatial and temporal space through predefined grid-shaped BEV queries. To aggregate spatial information, the authors design a spatial cross-attention that each BEV query extracts the spatial features from the regions of interest across camera views. For temporal information, the authors propose a temporal self-attention to recurrently fuse the history BEV information.
The proposed approach achieves the new state-of-the-art **56.9\%** in terms of NDS metric on the nuScenes test set, which is **9.0** points higher than previous best arts and on par with the performance of LiDAR-based baselines.


# Methods
![method](figs/arch.png "model arch")


# Getting Started
- [Installation](docs/install.md) 
- [Prepare Dataset](docs/prepare_dataset.md)
- [Run and Eval](docs/getting_started.md)


# HFai Adaptation

Follow [hf_guide](./hf_guide.md) to adapt to Fire-Flyer II.

Train BEVFormer with 10 Nodes
```
hfai python tools/train.py projects/configs/bevformer/bevformer_base.py --work-dir out/node10_train --cfg-options optimizer.lr=0.0008 -- --nodes 10 --priority 40 --name node10_train
```

Eval BEVFormer with 10 Nodes
```
hfai python tools/test.py projects/configs/bevformer/bevformer_base.py out/node10_train/epoch_24.pth --launcher pytorch --eval bbox -- --nodes 10 --priority 40 --name node10_test
```

# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{li2022bevformer,
  title={BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng}
  journal={arXiv preprint arXiv:2203.17270},
  year={2022}
}
```
