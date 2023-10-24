FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN sed -i "s@http://\(deb\|security\).debian.org@https://mirrors.aliyun.com@g" /etc/apt/sources.list
RUN sed -i s/cn.archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
RUN sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt-get clean
#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update -y

# opencv

RUN apt-get install -y libglib2.0-0
RUN apt-get install libsm6 -y
RUN apt-get install libxrender1 -y
RUN apt install libxext6 -y
RUN apt-get install libgl1-mesa-glx -y

# python 3.7

RUN apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN rm -rf /opt/conda
RUN rm -rf /root/.conda

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n python-app python=3.7 && \
    conda activate python-app && \
    echo 'print("Hello World!")' > python-app.py

RUN echo 'conda activate python-app \n\
alias python-app="python python-app.py"' >> /root/.bashrc

SHELL ["conda", "run", "-n", "python-app", "/bin/bash", "-c"]

# pip basic

RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple --upgrade pip
RUN conda install -c conda-forge mamba
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple -U openmim

RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple ipython
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple matplotlib
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple tqdm
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple wget
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple click
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple easydict
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple future
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple requests
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple tqdm
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple ninja
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple psutil

RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple scipy
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple numpy
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple pandas
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple scikit-learn

# pip image
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple pyspng
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple Pillow==9.2.0
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple scikit-image==0.17.2
#RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple opencv-python==4.1.1.26
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple opencv-python==4.5.5.64
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple imageio-ffmpeg==0.4.3

# torch

#RUN pip3 install -i https://mirrors.ustc.edu.cn/pypi/web/simple torch==1.7.1 torchvision torchaudio
#RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple timm

# tensorflow
#RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple keras-applications>=1.0.6
#RUN pip install keras-applications>=1.0.6
#RUN pip install keras-applications>=1.0.8
#RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple keras-applications>=1.0.8
RUN pip install -i https://mirrors.aliyun.com/pypi/simple keras-applications>=1.0.8
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple tensorflow==1.15
#RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple tensorboard==2.6.0
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple tensorboardX==2.5.1

# mmcv
#RUN mim install mmcv-full -i https://pypi.tuna.tsinghua.edu.cn/simple

# pip other
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple easydict
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple --upgrade "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
#RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple opencv-python==4.5.5.64


ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "python-app"]
