# Docker startup commands

docker memo
sudo docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -d --ipc=host -p 5001:5001 --rm -v /home/shks-ubuntu/:/home/ kyeshmz tail -f /dev/null

sudo docker exec -it containername /bin/bash




# Avatarify environment startup commands

- install Anaconda
- create python3.7 environment with name = py37
- launch terminal from py37
- `cd MorphingIdentity/Main/.venv-ava` (this is not included in this github)
- modify pyenv.cgf
```
home = C:\Users\*****\.conda\envs\py37
include-system-site-packages = false
version = 3.7.***

```
- then `launch_avaterify-hogehoge.bat`
