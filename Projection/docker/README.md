docker memo
sudo docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --ipc=host -p 5000:5000/udp -p 5000:5000/tcp --rm -v /home/shks-ubuntu/stylegan2:/home/ kyeshmz /bin/bash

docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --ipc=host -p 5000:5000/udp -p 5000:5000/tcp --rm -v /home/shks-ubuntu/:/home/ morphing /bin/bash
