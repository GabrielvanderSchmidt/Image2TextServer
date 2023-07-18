FROM huggingface/transformers-pytorch-gpu

ENV TRANSLATOR_SERVER=192.168.128.242:1880

RUN apt-get update && \
    apt-get install python3-flask -y

#RUN git clone https://github.com/GabrielvanderSchmidt/Image2TextServer.git

#WORKDIR Image2TextServer

RUN mkdir images

ADD main.py main.py

EXPOSE 1881

CMD ["python3", "-u", "main.py"]
