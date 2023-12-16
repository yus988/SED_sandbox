FROM python:3.8-buster
USER root

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

COPY req.txt /root/

RUN apt-get install -y vim less
# pqg_melspec 用
RUN apt-get install -y portaudio19-dev
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install libxkbcommon-x11-0
RUN apt-get install libdbus-1-3

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r /root/req.txt
