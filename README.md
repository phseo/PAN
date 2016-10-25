# Progressive Attention Networks for Visual Attribute Prediction

This implements a network introduced in the arXiv paper:

[Hierarchical Attention Networks](https://arxiv.org/abs/1606.02393)

You first need to place MNIST dataset in ./data/mnist/

- generate_mdist.py generates the MDIST synthetic dataset.
- pack.py packs the generated images and labels into a .npz file.
- train.py trains a progressive attention network with local context of size 3x3 and tests it at every epoch.

If you're using this code in a publication, please cite our paper.

    @article{seo2016hierarchical,
      title={Hierarchical Attention Networks},
      author={Seo, Paul Hongsuck and Lin, Zhe and Cohen, Scott and Shen, Xiaohui and Han, Bohyung},
      journal={arXiv preprint arXiv:1606.02393},
      year={2016}
    }

