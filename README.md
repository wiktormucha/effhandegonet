# Repository in preparation...





<img src="./images/method_overview.png" width="1000">


# In My Perspective, In My Hands: Accurate Egocentric 2D Hand Pose and Action Recognition



[![View on arXiv](https://img.shields.io/badge/arXiv-2404.09308-b31b1b.svg)](https://arxiv.org/abs/2404.09308)

>Action recognition is essential for egocentric video understanding, allowing automatic and continuous monitoring of Activities of Daily Living (ADLs) without user effort. Existing literature focuses on 3D hand pose input, which requires computationally intensive depth estimation networks or wearing an uncomfortable depth sensor. In contrast,  there has been insufficient research in understanding 2D hand pose for egocentric action recognition, despite the availability of user-friendly smart glasses in the market capable of capturing a single RGB image. Our study aims to fill this research gap by exploring the field of 2D hand pose estimation for egocentric action recognition, making two contributions. Firstly, we introduce two novel approaches for 2D hand pose estimation, namely EffHandNet for single-hand estimation and EffHandEgoNet, tailored for an egocentric perspective, capturing interactions between hands and objects. Both methods outperform state-of-the-art models on H2O and FPHA public benchmarks. Secondly, we present a robust action recognition architecture from 2D hand and object poses. This method incorporates EffHandEgoNet, and a transformer-based action recognition method. Evaluated on H2O and FPHA datasets, our architecture has a faster inference time and achieves an accuracy of 91.32\% and 94.43\%, respectively, surpassing state of the art, including 3D-based methods. Our work demonstrates that using 2D skeletal data is a robust approach for egocentric action understanding. Extensive evaluation and ablation studies show the impact of the hand pose estimation approach, and how each input affects the overall performance.




# Bibtex

If you find this work useful or the models in your research or applications, please cite the paper using this BibTeX

```BibTeX
@article{mucha2024my,
  title={In My Perspective, In My Hands: Accurate Egocentric 2D Hand Pose and Action Recognition},
  author={Mucha, Wiktor and Kampel, Martin},
  journal={arXiv preprint arXiv:2404.09308},
  year={2024}
}
```