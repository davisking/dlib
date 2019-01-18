FROM nvidia/cuda:8.0-cudnn5-devel

RUN apt update -y; apt install -y \
git \
cmake \
libsm6 \
libxext6 \
libxrender-dev \
gcc-4.8 \
g++-4.8 \
python3 \
python3-pip

RUN pip3 install \
scikit-build

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 50
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 50

RUN git clone https://github.com/davisking/dlib.git
RUN mkdir -p /dlib/build

RUN cmake -H/dlib -B/dlib/build -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
RUN cmake --build /dlib/build

RUN cd /dlib; python3 /dlib/setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
