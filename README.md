# DCT-Net: Domain-Calibrated Translation for Portrait Stylization

### [Project page](https://menyifang.github.io/projects/DCTNet/DCTNet.html) |  [Video](https://www.youtube.com/watch?v=Y8BrfOjXYQM) | [Paper](https://arxiv.org/abs/2207.02426)

Official implementation of DCT-Net for Portrait Stylization.


> [**DCT-Net: Domain-Calibrated Translation for Portrait Stylization**](arxiv_url_coming_soon),             
> [Yifang Men](https://menyifang.github.io/)<sup>1</sup>, Yuan Yao<sup>1</sup>, Miaomiao Cui<sup>1</sup>, [Zhouhui Lian](https://www.icst.pku.edu.cn/zlian/)<sup>2</sup>, Xuansong Xie<sup>1</sup>,        
> _<sup>1</sup>[DAMO Academy, Alibaba Group](https://damo.alibaba.com), Beijing, China_  
> _<sup>2</sup>[Wangxuan Institute of Computer Technology, Peking University](https://www.icst.pku.edu.cn/), China_     
> In: SIGGRAPH 2022 (**TOG**) 
> *[arXiv preprint](https://arxiv.org/abs/2207.02426)* 


## Demo
![demo_vid](assets/demo.gif)

# Web Demo

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SIGGRAPH2022/DCT-Net)

## News
(2022-07-07) The paper is available now at arxiv(https://arxiv.org/abs/2207.02426).

(2022-08-08) cartoon function can be directly call from pythonSDK of [modelscope](https://modelscope.cn/#/models).

(2022-08-08) The pertained model and infer code of 'anime' style is available now. More styles coming soon.



## Requirements
* python 3
* tensorflow (>=1.14)
* easydict
* numpy
* both CPU/GPU are supported


## Quick Start

```bash
git clone https://github.com/menyifang/DCT-Net.git
cd DCT-Net

```

### From python SDK
A quick use with python SDK

- Installation:
```bash
conda create -n dctnet python=3.8
conda activate dctnet
pip install tensorflow
pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

- Downloads:
```bash
python download.py
```

- Inference:
```bash
python run_sdk.py
```


### From source code
```bash
python run.py
```


## Acknowledgments

Face detector and aligner are adapted from [Peppa_Pig_Face_Engine](https://github.com/610265158/Peppa_Pig_Face_Engine
) and [InsightFace](https://github.com/TreB1eN/InsightFace_Pytorch).



## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{men2022dct,
  title={DCT-Net: Domain-Calibrated Translation for Portrait Stylization},
  author={Men, Yifang and Yao, Yuan and Cui, Miaomiao and Lian, Zhouhui and Xie, Xuansong},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={4},
  pages={1--9},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```







