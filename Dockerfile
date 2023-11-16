FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
LABEL maintainer="Alexander Hunt <alexander.hunt@ed.ac.uk>"
LABEL description="Docker image setup to use Darknet with Cuda 11.7.1, Cudnn 8 and OpenCV 4.6.0 on Ubuntu 20.04"
RUN apt update && apt upgrade -y
ENV TZ=Europe/London \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get -yq install build-essential git pkg-config \
    libcudnn8* wget unzip python3 python-is-python3 python3-pip \
    libssl-dev zip unzip libeigen3-dev libgflags-dev libgoogle-glog-dev -y
RUN apt-get install libjpeg-dev libpng-dev libtiff-dev libopenjp2-7-dev -y
RUN apt-get install libavcodec-dev libavformat-dev libswscale-dev -y
RUN apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev -y
RUN apt-get install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev -y
RUN apt-get install libfaac-dev libvorbis-dev -y
RUN apt-get install libopencore-amrnb-dev libopencore-amrwb-dev -y
RUN apt-get install libgtk-3-dev -y 
RUN apt-get install libtbb-dev -y 
RUN apt-get install libprotobuf-dev protobuf-compiler -y
RUN apt-get install libgoogle-glog-dev libgflags-dev -y
RUN apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen -y
RUN apt-get install libgtkglext1 libgtkglext1-dev -y
RUN apt-get install libopenblas-dev liblapacke-dev libva-dev libopenjp2-tools libopenjpip-dec-server libopenjpip-server libqt5opengl5-dev libtesseract-dev -y 
WORKDIR /root/
RUN cd 
RUN pip3 install --upgrade pip
RUN pip3 install numpy imutils tqdm inquirer pandas seaborn matplotlib scikit-learn
RUN apt install *libopencv* python3-opencv -y
WORKDIR /root/
RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.0-rc1/cmake-3.28.0-rc1.tar.gz
RUN tar -xvzf cmake-3.28.0-rc1.tar.gz
WORKDIR /root/cmake-3.28.0-rc1
RUN ./bootstrap
RUN make -j$(nproc)
RUN make install
RUN ldconfig
WORKDIR /root/
RUN git clone https://github.com/AlexeyAB/darknet.git
WORKDIR /root/darknet 
COPY Makefile /root/darknet/Makefile
RUN make
RUN ln -s /root/darknet/darknet /usr/bin
WORKDIR /root/
COPY main.py /root/main.py
COPY yolov4.conv.137 /root/yolov4.conv.137
COPY code/ /root/code
COPY config.py.docker /root/code/data/config.py
ENTRYPOINT ["./main.py"]



