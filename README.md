### 1A.Starting of docker

We first check if there is docker container
`docker container ls`
If there is already a container, go to step 2A.

If there is no container, we start by using the container below

```
sudo docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -d --ipc=host -p 5001:5001 --rm -v /home/ubuntu/MorphingIdentity/:/home/ morphing /bin/bash

```

### 2A.Use after docker container starts

To use after docker container is setup
` docker exec -it containername /bin/bash`

If the processs is running from before, we need to kill it
`ps -a | grep python`

then kill using pid
`kill PIDOFPROCESS`

### 3A.Inside of docker

```

cd /home/MorphingIdentity/Projection/src
python main.py
# exit out of CTRL-P + CTRL-Q

```

# Avatarify environment startup commands

- install Anaconda
- create python3.7 environment with name = py37
- launch terminal from py37
- `cd MorphingIdentity/Main/.venv-ava` (this is not included in this github)
- modify pyenv.cgf

```

home = C:\Users\*\***_\.conda\envs\py37
include-system-site-packages = false
version = 3.7._**

```

- then `launch_avaterify-hogehoge.bat`

```

```
