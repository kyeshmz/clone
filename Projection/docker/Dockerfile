FROM nvcr.io/nvidia/tensorflow:21.03-tf1-py3 as tf
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install wget unzip git ca-certificates ninja-build openssl sqlite vim nano screen libgl1-mesa-dev bzip2 -y
RUN pip install --user numpy dlib pynng python-osc pytz trio imutils tqdm matplotlib scipy opencv-python pillow torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.33.0/install.sh | bash
RUN export NVM_DIR="/usr/local/nvm" &&\
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" &&\
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion" &&\
nvm install stable && nvm use stable && npm install --global yarn &&\
yarn global add pm2 &&\
pm2 install pm2-slack &&\
pm2 set pm2-slack:slack_url https://hooks.slack.com/services/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 &&\
pm2 set pm2-slack:restart true &&\
pm2 set pm2-slack:start true &&\
pm2 set pm2-slack:exit true

SHELL ["/bin/bash", "-c"]
