#!/bin/bash
sudo apt-get update
# install docker
sudo apt install docker.io
# start docker service
sudo systemctl start docker
sudo systemctl enable docker
# install docker compose
sudo curl -SL https://github.com/docker/compose/releases/download/v2.23.3/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
sudo chmod +x /user/local/bin/docker-compose
# build and run docker containers
cd ~/server
sudo docker-compose up --build