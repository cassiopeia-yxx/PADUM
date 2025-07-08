# Pixel Adaptive Deep Unfolding Network with State Space Model for Image Deraining 

Yao Xiao, and Youshen Xia

<hr />

> **Abstract:** *Rain streaks   affects the visual quality and interfere with high-level vision tasks on rainy days.  Removing raindrops from  captured rainy images becomes  improtant in  computer vision applications. Recently, deep unfolding networks (DUNs) are shown their effectiveness on image deraining.  Yet,   there are two  issues that need to be further addressed : 1) Deep unfolding networks typically use convolutional neural networks (CNNs), which lack the ability to perceive global structures, thereby limiting the applicability of the network model; 2) Their gradient descent modules usually rely on a scalar step size, which limits the adaptability of the method to different input images. To address the two  issues,  we proposes a new image rain removal method based on a pixel adaptive deep unfolding network with state space models. The proposed network mainly consists of  both  the adaptive pixel-wise gradient descent (APGD) module and the stage fusion proximal mapping (SFPM) module. APGD module overcomes scalar step size inflexibility by adaptively adjusting the gradient step size for each pixel based on the previous stage features. SFPM module adopts a dual-branch architecture combining  CNNs  with state space models (SSMs) to  enhance the perception of both local and global structures.  Compared to Transformer-based models, SSM enables efficient long-range dependency modeling with linear complexity. In addition, we introduce a stage feature fusion with the Fourier transform mechanism to reduce information loss during the unfolding process, ensuring key features are effectively propagated.  Extensive experiments on multiple public datasets demonstrate that our method consistently outperforms state-of-the-art deraining methods in terms of  quantitative metrics and visual quality.*
<hr />



## Network Architecture

<img src = "./figs/network.png"> 

## Datasets
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>DID-Data</th>
    <th>DDN-Data</th>
    <th>SPA-Data</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="https://pan.baidu.com/s/1RV677SOIBgWB_3u9rInX4w">Download (nkyp)</a> </td>
    <td> <a href="https://pan.baidu.com/s/1AjR_gGMwadnaZRU-U_FJhQ">Download (ajck)</a> </td>
    <td> <a href="https://pan.baidu.com/s/1sUhI5xz9XGu0gTnNcYQ3xw">Download (hgg6)</a> </td>
    <td> <a href="https://pan.baidu.com/s/11cZKW0eGBunzgYatT82SPA">Download (62iw)</a> </td>
    <td> <a href="https://pan.baidu.com/s/18RxAjfJABRZJ8XculdaXoA">Download (vz83)</a> </td>
  </tr>
</tbody>
</table>

## Environment Setup
Please follow the steps below to create and configure the environment:
```
# Create a new conda environment with Python 3.10.13
conda create -n PADMamba python=3.10.13
conda activate PADMamba

# Install PyTorch and CUDA
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install common packages
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
pip install tb-nightly -i https://mirrors.aliyun.com/pypi/simple

# Download specific versions of causal_conv1d and mamba_ssm
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.3.post1/causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install downloaded wheel packages
pip install causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Replace the mamba_ssm implementation in your environment with the optimized version from [Vim](https://github.com/cassiopeia-yxx/Vim)
conda env list
cd Vim
cp -rf mamba-1p1p1/mamba_ssm /home/user/anaconda3/envs/PADM/lib/python3.10/site-packages

# install basicsr
python setup.py develop --no_cuda_ext
```




## Training
1. Please download the corresponding training datasets and put them in the folder `Datasets/train`. Download the testing datasets and put them in the folder `Datasets/test`. 
2. Follow the instructions below to begin training our model.
```
cd PADmamba
bash train.sh
```
Run the script then you can find the generated experimental logs in the folder `experiments`.

## Testing
1. Please download the corresponding testing datasets and put them in the folder `test/input`. Download the corresponding pre-trained models and put them in the folder `pretrained_models`.
2. Please modify the corresponding paths in the file `test.py` to the corresponding parsers `input_dir` and `result_dir`.
3. Follow the instructions below to begin testing our model.
```
python test.py'
```
Run the script then you can find the output visual results in the folder `test/output/Deraining`.

## Pre-trained Models
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>DID-Data</th>
    <th>DDN-Data</th>
    <th>SPA-Data</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="https://pan.baidu.com/s/1csOhCV8BxahzANnR9JMUPw">Download (1p8d)</a>  </td>
    <td> <a href="https://pan.baidu.com/s/1GdC_n_SJ1xutkG1hido9Ng">Download (xggm)</a>  </td>
    <td> <a href="https://pan.baidu.com/s/1JUotJ1SrpfOELF41_3VoKg">Download (qf2k)</a>  </td>
    <td> <a href="https://pan.baidu.com/s/1mZg87fyCxZq_gFTNKqUcmg">Download (fewt)</a>  </td>
    <td> <a href="https://pan.baidu.com/s/1oy9Aa_LDTbGlfz27iYlNzA">Download (85u4)</a>  </td>
  </tr>
</tbody>

</table>

## Visual Deraining Results
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>DID-Data</th>
    <th>DDN-Data</th>
    <th>SPA-Data</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu NetDisk	</td>
    <td> <a href="https://pan.baidu.com/s/1oIU-Bmm261G8EBkdRbZeAA">DWL (qmj8)</a>  </td>
    <td> <a href="https://pan.baidu.com/s/1mzN_HMX18bf7WB0FFR1IXg">DWL (2byh)</a>  </td>
    <td> <a href="https://pan.baidu.com/s/1QBCKTAW0Add1oM1vLuFgog">DWL (9prg)</a>  </td>
    <td> <a href="https://pan.baidu.com/s/1f0npqDmWSzM3DvoSkiFS2Q">DWL (u6ii)</a>  </td>
    <td> <a href="https://pan.baidu.com/s/1n8bXvys9gpQkCO5xByoiXA">DWL (fwt7)</a>  </td>

  </tr>
</tbody>
</table>


## Performance Evaluation
See folder "evaluations/Evaluation_DID-Data_DDN-Data or Evalution_Rain200L_Rain200H_SPA-Data". 

<img src = "./figs/table.png">



Some of the experimental results are based on the results collected by [DRSformer](https://github.com/cschenxiang/DRSformer), which is very comprehensive. Thanks for their awesome work.


## Citation
If you are interested in this work, please consider citing:

    @article{PADMamba,
        author={Yao, Xiao and Youshen, Xia}, 
        title={Pixel Adaptive Deep Unfolding Network with State Space Model for Image Deraining},
        journal={Neural Networks},
        year={2025},
    }

## Acknowledgment
This code is based on the [Vim](https://github.com/hustvl/Vim)、[DRSformer](https://github.com/cschenxiang/DRSformer). Thanks for their awesome work.

## Contact
If your submitted issue has not been noticed or there are further questions, please contact xiaoyao227192@163.com.

