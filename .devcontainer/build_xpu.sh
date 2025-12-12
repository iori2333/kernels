#!/bin/bash

docker build . -t devcontainers/oneapi:2025.2.2 -f .devcontainer/Dockerfile.xpu
