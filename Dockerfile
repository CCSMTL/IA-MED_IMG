FROM pytorch/pytorch:latest

COPY ./CheXpert2/requirements.txt /install/requirements.txt
RUN pip install -r /install/requirements.txt
ENV img_dir="/mnt/e"
WORKDIR /code
