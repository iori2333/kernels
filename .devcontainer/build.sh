#!/bin/bash

docker build . -t devcontainers/cuda:13.1-devel -f .devcontainer/Dockerfile
