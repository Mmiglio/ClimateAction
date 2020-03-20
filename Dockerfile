FROM python:3.7-slim

# install Java
USER root
RUN echo "deb http://security.debian.org/debian-security stretch/updates main" >> /etc/apt/sources.list                                                   
RUN mkdir -p /usr/share/man/man1 && \
    apt-get update -y && \
    apt-get install -y openjdk-8-jdk

RUN apt-get install unzip -y && \
    apt-get autoremove -y

# Install jupyterlab
RUN pip install jupyterlab

WORKDIR /work
# install requirements
ADD requirements.txt /work
RUN pip install -r requirements.txt

CMD jupyter-notebook --no-browser --ip 0.0.0.0 --port 8888 --allow-root /work
