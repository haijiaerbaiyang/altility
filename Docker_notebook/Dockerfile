FROM ubuntu:latest

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3 python3-pip python3-dev
RUN pip3 -q install pip --upgrade
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

WORKDIR /altility
CMD jupyter notebook --ip 0.0.0.0 --port 1111 --no-browser --allow-root

