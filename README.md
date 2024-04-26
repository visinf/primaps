# Boosting Unsupervised Semantic Segmentation with Principal Mask Proposals

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This is the official repository of our paper:

**Boosting Unsupervised Semantic Segmentation with Principal Mask Proposals**<br>
[Oliver Hahn](https://olvrhhn.github.io),
[Nikita Araslanov](https://arnike.github.io),
[Simone Schaub-Meyer](https://schaubsi.github.io),
and [Stefan Roth](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp)<br>
[[arXiv](https://arxiv.org/abs/2404.16818)]

**Abstract:** Unsupervised semantic segmentation aims to automatically partition images into semantically meaningful regions by identifying global categories within an image corpus without any form of annotation. Building upon recent advances in self-supervised representation learning, we focus on how to leverage these large pre-trained models for the downstream task of unsupervised segmentation. We present <i>PriMaPs</i> - Principal Mask Proposals - decomposing images into semantically meaningful masks based on their feature representation. This allows us to realize unsupervised semantic segmentation by fitting class prototypes to <i>PriMaPs</i> with a stochastic expectation-maximization algorithm, <i>PriMaPs-EM</i>. Despite its conceptual simplicity, <i>PriMaPs-EM</i> leads to competitive results across various pre-trained backbone models, including DINO and DINOv2, and across datasets, such as Cityscapes, COCO-Stuff, and Potsdam-3. Importantly, <i>PriMaPs-EM</i> is able to boost results when applied orthogonally to current state-of-the-art unsupervised semantic segmentation pipelines. 

<img src="assets/primaps_examples.png" width="512" />

Figure 1: Principal mask proposals (PriMaPs) are iteratively extracted
from an image (dashed arrows). Each mask is assigned a semantic class resulting in a pseudo label.


## Code and checkpoints coming soon!

## Results

PriMaPs-EM provides modest but consistent benefits over a wide range of baselines and datasets and reaches competitive segmentation accuracy w.r.t. the state-of-the-art constituting a straightforward, entirely orthogonal tool for boosting unsupervised semantic segmentation.

| Method       | Backbone      | Cityscapes (Acc / mIoU) | COCO-Stuff ( Acc / mIoU) | Potsdam-3 (Acc / mIoU) |
|--------------|:--------------:|:------------:|:------------:|:-----------:|
| Baseline     | DINO ViT-S/8 | 61.4 / 15.8 | 34.2 /  9.5  | 56.6 / 33.6 |
| +PriMaPs     | DINO ViT-S/8 | 81.2 / 19.3 | 46.5 / 16.4 | 62.5 / 39.0 |
| +SotA+PriMaPs|DINO ViT-S/8 | 76.3 / 19.2 | 57.8 / 25.1 | 78.4 / 64.2 |
| Baseline     | DINO ViT-B/8 | 49.2 / 15.5 | 38.8 / 15.7 | 66.1 / 49.4 |   
| +PriMaPs     |DINO ViT-B/8  | 59.6 / 17.6 | 48.4 / 21.9 | 80.5 / 66.9 |
| +SotA+PriMaPs|DINO ViT-B/8 | 78.6 / 21.6 | 57.9 / 29.7 | 83.3 / 71.0 |
| Baseline     | DINOv2 ViT-S/14 | 49.5 / 15.3 | 44.5 / 22.9 | 75.9 / 61.0 |
| +PriMaPs     | DINOv2 ViT-S/14 | 71.6 / 19.0 | 46.4 / 23.8 | 78.4 / 64.2 |
| Baseline     | DINOv2 ViT-B/14 | 36.1 / 14.9 | 35.0 / 17.9 | 82.4 / 69.9 |
| +PriMaPs     | DINOv2 ViT-B/14 | 82.8 / 21.2 | 52.6 / 23.6 | 83.1 / 71.0 |


## Citation
```
@article{Hahn:2024:BUS,
  title={Boosting Unsupervised Semantic Segmentation with Principal Mask Proposals},
  author={Oliver Hahn and Nikita Araslanov and Simone Schaub-Meyer and Stefan Roth},
  journal={ArXiv},
  year={2024}
}
```
