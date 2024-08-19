### Github repository of the PhD project of Alexander Hunt at the University of Edinburgh

This repository contains the machine code for the PhD project of Alexander Hunt at the University of Edinburgh. The project is supervised by Prof. Till Bachmann and Prof. Bob Fisher.

### Overview

machine-code contains the data processing pipeline, the AI interface and automatic results analysis for the PhD project. The pipeline is designed to process data imported from a microscope capturing device. The pipeline is designed to be modular and flexible, allowing for easy importing of data wether it be images or videos. The data will be converted to jpeg still images then cut into 416x416 pixel images. These images will be fed into a YOLOvX object detection model to detect present cells. The results will be analysed to determine the accuracy of the model or provide a report upon the sample. The pipeline is designed to be run on a server with a GPU to allow for fast processing of the data.

### Installation

First make sure you have the following installed:
- Python3 
- pip3
- CUDA 12.0
- cuDNN 8.0.5
- OpenCV 4.5.3 (compiled with CUDA for python and installed in PATH)
- darknet compiled with CUDA, cuDNN and OpenCV and installed in PATH (https://github.com/AlexeyAB/darknet)

Then run the following commands to install the required python packages:
```bash
pip3 install -r requirements.txt
```

### Usage

To run the program, run the following command:

```bash
python3 main.py
```

or on linux/mac:

```bash
./main.py
```
Then follow the instructions on the screen. The program will guide you through the process of importing data, processing it and analysing the results. The output path will be defined by you during the process.


