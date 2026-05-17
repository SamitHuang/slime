```
docker pull slimerl/slime:latest
```

```
docker run -itd --gpus all --ipc=host --shm-size=128g --net=host --privileged=true  --restart=always \
--ulimit memlock=-1 --ulimit stack=67108864 \
--ulimit nofile=65536:65536 \
--name DNAME  \
-it slimerl/slime:latest /bin/bash \

```
docker exec -it --user root DNAME bash
```

```
pip install vllm=0.16 

# for compatibility
pip install numpy==1.26.4
```
