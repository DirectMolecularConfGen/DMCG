FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV LANG=C.UTF-8
RUN rm -rf /etc/apt/sources.list.d/cuda.list && rm -rf /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    openssh-server  unzip curl \
    cmake gcc g++ ibverbs-providers \
    iputils-ping net-tools  iproute2  htop xauth \
    libxcursor1 libxdamage1 libxcomposite-dev libxrandr2 libxinerama1  \
    tmux wget vim git bzip2 ca-certificates  libxrender1  && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean

EXPOSE 22
# CMD ["/usr/sbin/sshd", "-D"]
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ay && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/profile && \
    echo "conda activate base" >> /etc/profile

WORKDIR /workspace
ENV envname py38
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -y -n $envname python=3.8 && \
    conda activate $envname && \
    conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia && \
    conda install pytorch-geometric=1.7.2 -c rusty1s -c conda-forge && \
    conda install -y -c conda-forge rdkit=2020.09.5 && \
    conda install -y tensorboard tqdm scipy scikit-learn black ipykernel && \
    conda install -y -c conda-forge graph-tool && \
    conda clean -ay &&  \
    sed -i 's/conda activate base/conda activate '"$envname"'/g' /etc/profile

ENV MKL_THREADING_LAYER GNU
ENV PATH /opt/conda/envs/${envname}/bin:$PATH
RUN echo "export LANG=C.UTF-8" >> /etc/profile && \
    echo "export MKL_THREADING_LAYER=GNU" >> /etc/profile

