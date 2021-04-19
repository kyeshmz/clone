docker memo
sudo docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --ipc=host -p 5001:5001 --rm -v /home/shks-ubuntu/stylegan2:/home/ kyeshmz /bin/bash
