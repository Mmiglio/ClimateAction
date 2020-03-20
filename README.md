# ClimateAction


## Docker

Build image with 
```bash
docker build --file Dockerfile --tag container-jupyter .
```
this will create a new image and install packages in `requirements.txt`.

Run with:
```bash
docker run --name jupyter-server -p 8888:8888 -v $PWD:/work container-jupyter
```
Now jupyter can be accessed by pasting one of the URLs provided, e.g
```bash
http://127.0.0.1:8888/?token=fe978e3ff88080bd7d7790750e955b0071cf5b8849462b74 
```

To stop the container use `docker stop jupyter-server`. It can be restarted usign `docker start -ai jupyter-server`. The container can be removed usign `docker rm jupyter-server`. 