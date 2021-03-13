FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install wget unzip git ca-certificates openssl sqlite -y
RUN git clone https://github.com/eladrich/pixel2style2pixel pixel2style2pixel
RUN cd pixel2style2pixel
RUN mkdir ./pretrained_models
RUN pip install numpy==1.18.4 tqdm==4.46.0 matplotlib==3.2.1 scipy==1.4.1 opencv-python==4.2.0.34 pillow==7.1.2
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip &&\
    unzip ninja-linux.zip -d /usr/local/bin/ &&\
    update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0" -O pretrained_models/psp_ffhq_encode.pt && rm -rf /tmp/cookies.txt
SHELL ["/bin/bash", "-c"]

