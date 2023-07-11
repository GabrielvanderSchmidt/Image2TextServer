FROM huggingface/transformers-pytorch-gpu

RUN apt-get update && \
    apt-get install python3-flask -y

RUN git clone https://github.com/GabrielvanderSchmidt/Image2TextServer.git

WORKDIR Image2TextServer

RUN mkdir images

ADD main.py main.py

CMD ["python3", "main.py"]
