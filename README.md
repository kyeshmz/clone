### 1A.Starting of docker

We first check if there is docker container
`docker container ls`
If there is already a container, go to step 2A.

If there is no container, we start by using the container below

```
docker run --shm-size=1g --ulimit memlock=-1  --ulimit stack=67108864 -d  --gpus all  --ipc=host -p 5001:5001   -v /home/ubuntu/:/home/ubuntu/ morphing /bin/bash -c "cd /home/ubuntu/MorphingIdentity/MorphingIdentity/Projection/src/; python /home/ubuntu/MorphingIdentity/MorphingIdentity/Projection/src/main.py"

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
