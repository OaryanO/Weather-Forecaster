#!/bin/bash
echo "Installing Python 3.10.12..."
sudo apt-get update -y
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

echo "Setting Python 3.10 as default..."
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --config python3

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
