#!/bin/zsh

#source:https://github.com/OpenKinect/libfreenect2

echo "Starting installation libfreenect2"
git clone https://github.com/OpenKinect/libfreenect2.git &&
cd libfreenect2 &&
sudo apt-get install -y build-essential cmake pkg-config &&
sudo apt-get install -y libusb-1.0-0-dev &&

sudo apt-get install -y libturbojpeg0-dev &&
sudo apt-get install -y libglfw3-dev &&

mkdir build && 
cd build &&
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2 &&
make &&
sudo make install &&
sudo cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/ &&
echo "installation concluded"
