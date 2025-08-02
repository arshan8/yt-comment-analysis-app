#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 653028031424.dkr.ecr.eu-north-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 653028031424.dkr.ecr.eu-north-1.amazonaws.com/yt-chrome-plugin:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=arshan-app)" ]; then
    docker stop arshan-app
fi

if [ "$(docker ps -aq -f name=arshan-app)" ]; then
    docker rm arshan-app
fi


echo "Docker starting successfully."
docker run -d -p 80:5000 --name arshan-app 653028031424.dkr.ecr.eu-north-1.amazonaws.com/yt-chrome-plugin:latest
echo "Docker container started successfully."
