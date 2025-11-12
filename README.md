This is the official implementation of the AAAI'26 paper titled [**"Revitalizing Canonical Pre‑Alignment for Irregular Multivariate Time Series Forecasting"**](https://www.arxiv.org/abs/2508.01971)

## 🔧 Reproducing Experiments

Under the path ./KAFNet/KAFNet, reviewers can reproduce experiment on Human Activity dataset by simply running:

```bash
bash ./test.sh
```

## 📂 Model Implementation

The core model implementation can be found in the file:

```
./model/KAFNet.py
```

This file contains the complete architecture of our proposed KAFNet model.
 

## Citation
If you find any of the code useful, feel free to cite this paper.
```
@article{zhou2025kafnet,
      title={Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting}, 
      author={Ziyu Zhou and Yiming Huang and Yanyun Wang and Yuankai Wu and James Kwok and Yuxuan Liang},
      year={2025},
      journal={arXiv preprint arXiv:2508.01971}
}

@inproceedings{zhou2025kafnet,
    title={Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting}, 
    author={Ziyu Zhou and Yiming Huang and Yanyun Wang and Yuankai Wu and James Kwok and Yuxuan Liang},
    year={2026},
    booktitle = {AAAI Conference on Artificial Intelligence}
}
```


## Acknowledgement
We use the code from the repository [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN)