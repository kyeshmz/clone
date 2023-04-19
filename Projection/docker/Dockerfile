FROM nvcr.io/nvidia/tensorflow:21.03-tf1-py3 as tf
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install wget unzip git ca-certificates ninja-build openssl sqlite vim nano screen libgl1-mesa-dev bzip2 iputils-ping net-tools -y
RUN pip install --user numpy dlib pynng python-osc pytz trio imutils tqdm matplotlib scipy opencv-python pillow torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN curl -s https://install.zerotier.com | sudo bash
SHELL ["tail", "-f", "/dev/null"]
