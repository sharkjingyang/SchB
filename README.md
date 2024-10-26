# SchB-Flow
Pytorch implementation of schrödinger bridge flow。

## Associated Publication

A Machine Learning Framework for Geodesics Under Spherical Wasserstein–Fisher–Rao Metric and Its Application for Weighted Sample Generation

Paper: https://link.springer.com/article/10.1007/s10915-023-02396-y

Please cite as
    
    @article{jing2024machine,
      title={A Machine Learning Framework for Geodesics Under Spherical Wasserstein--Fisher--Rao Metric and Its Application for Weighted Sample Generation},
      author={Jing, Yang and Chen, Jiaheng and Li, Lei and Lu, Jianfeng},
      journal={Journal of Scientific Computing},
      volume={98},
      number={1},
      pages={5},
      year={2024},
      publisher={Springer}
    }



for 1d toy experiment, see 

```
trainToyOTflow_1d_inverse.py
```

for 2d and high-dimension experiment, see

```
trainToyOTflow_high_Bayes.py (train) & high_dim_Bayes folder (evaluation)
```

for primal-dual method to solve OT/sWFR, see

```
primal_dual_1d/OT-main.m
primal_dual_1d/SWFR-main.m
```

