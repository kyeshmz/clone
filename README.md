# Docker startup commands

docker memo
sudo docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -d --ipc=host -p 5001:5001 --rm -v /home/shks-ubuntu/:/home/ kyeshmz tail -f /dev/null

sudo docker exec -it containername /bin/bash
