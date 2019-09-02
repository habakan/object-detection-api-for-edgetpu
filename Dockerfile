FROM nvidia/cuda:10.0-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive


# python installation
ENV PYTHON_VERSION 3.7.1
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv
RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    gdebi \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    unzip \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
 && git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
 && $PYENV_ROOT/plugins/python-build/install.sh \
 && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
 && rm -rf $PYENV_ROOT

# cudnn reinstall
RUN apt remove --allow-change-held-packages -y libcudnn7 libcudnn7-dev
RUN wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.5.0.56-1+cuda10.0_amd64.deb 
RUN wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.5.0.56-1+cuda10.0_amd64.deb 
RUN dpkg -i libcudnn7_7.5.0.56-1+cuda10.0_amd64.deb
RUN dpkg -i libcudnn7-dev_7.5.0.56-1+cuda10.0_amd64.deb

# tensorflow and object detection api installation
RUN pip install tensorflow-gpu==1.13.1

RUN mkdir /protoc_3.3 && \
    cd protoc_3.3 && \
    wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip && \
    chmod 775 protoc-3.3.0-linux-x86_64.zip && \
    unzip protoc-3.3.0-linux-x86_64.zip

RUN apt-get install -y protobuf-compiler python-pil python-lxml python-tk
RUN pip install --user Cython contextlib2 pillow lxml matplotlib

RUN git clone https://github.com/tensorflow/models.git

RUN git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    make && \
    cp -r pycocotools /models/research/

RUN cd /models/research && \
    /protoc_3.3/bin/protoc object_detection/protos/*.proto --python_out=. 

ENV PYTHONPATH $PYTHONPATH:/models/research:/models/research/slim
#CMD export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

COPY utils/create_tf_record.py /models/research/object_detection/dataset_tools/ 
COPY utils/shell_script /models/reserach/.


# edge tpu compiler installation
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get update
RUN apt-get install -y edgetpu
