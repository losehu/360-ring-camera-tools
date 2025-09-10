# 标定内参
* 在`./ocam/tests/test_bundle_adjustment.py`：
** 修改data_dir：jpg的文件夹路径
** 修改out_path：内参输出文件名
# 全景环带展开
* 在`./unfold_pal.py`
** result_txt：内参文件
** img_name：需要展开的图片
* './unfold_pal.cpp'为C++实现，Makefile编译
#所有代码在主文件夹目录下运行
# 雷达对齐
* './calib/cameraCalib.cpp'：棋盘图求针孔相机内参
* './calib/main.cpp'：针孔相机与雷达对齐
* './calib/main1.cpp'：针孔相机与雷达二次对齐
* './calib/projectCloud.cpp'：将对齐结果投影到针孔图片

# 全景环带雷达对齐
## DOING


apt install -y build-essential cmake make  libeigen3-dev libgoogle-glog-dev libgflags-dev libsuitesparse-dev libopencv-dev liblapack-dev libsuitesparse-dev libgflags-dev  libgoogle-glog-dev libgtest-dev libpcl-dev libboost-all-dev
https://codeload.github.com/ceres-solver/ceres-solver/zip/refs/tags/2.2.0 > ceres.zip && unzip ceres.zip
cd ceres-solver-2.2.0 &&mkdir build && cd build && cmake .. && make -j8 
make install
sudo /mnt# 360-ring-camera-tools
