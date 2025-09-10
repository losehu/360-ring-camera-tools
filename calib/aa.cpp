#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <filesystem> // C++17
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "common.h"
#include "result_verify.h"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

float max_depth = 60;
float min_depth = 3;
cv::Mat src_img;

int threshold_lidar = 30000;  
string input_pcd_dir, input_photo_path, output_path, intrinsic_path, extrinsic_path;

struct PointXYZRGB {
    float x, y, z;
};

void getColor(int &result_r, int &result_g, int &result_b, float cur_depth) {
    float scale = (max_depth - min_depth) / 10.0f;
    if (cur_depth < min_depth) {
        result_r = 0; result_g = 0; result_b = 255;
    }
    else if (cur_depth < min_depth + scale) {
        result_r = 0;
        result_g = int((cur_depth - min_depth) / scale * 255) & 0xff;
        result_b = 255;
    }
    else if (cur_depth < min_depth + scale * 2) {
        result_r = 0;
        result_g = 255;
        result_b = (255 - int((cur_depth - min_depth - scale) / scale * 255)) & 0xff;
    }
    else if (cur_depth < min_depth + scale * 4) {
        result_r = int((cur_depth - min_depth - scale * 2) / scale * 255) & 0xff;
        result_g = 255;
        result_b = 0;
    }
    else if (cur_depth < min_depth + scale * 7) {
        result_r = 255;
        result_g = (255 - int((cur_depth - min_depth - scale * 4) / scale * 255)) & 0xff;
        result_b = 0;
    }
    else if (cur_depth < min_depth + scale * 10) {
        result_r = 255;
        result_g = 0;
        result_b = int((cur_depth - min_depth - scale * 7) / scale * 255) & 0xff;
    }
    else {
        result_r = 255; result_g = 0; result_b = 255;
    }
}

void loadPointcloudFromDir(const string& dir_path, vector<PointXYZRGB> &points) {
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".pcd") {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            if (pcl::io::loadPCDFile<pcl::PointXYZ>(entry.path().string(), *cloud) == -1) {
                cerr << "Failed to load " << entry.path() << endl;
                continue;
            }
//打印点的数量
            for (auto& p : cloud->points) {
                points.push_back({p.x, p.y, p.z});
            }
        }
    }
}

int main(int argc, char **argv) {

    input_pcd_dir = "/Users/losehu/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_y3fi4f9tcu0322_11eb/msg/file/2025-08/bag2pcd"; // 这里改成你的文件夹路径
    input_photo_path = "./333.jpg";

    output_path = "./result.png";
    intrinsic_path = "./intrinsic.txt";
    extrinsic_path = "./extrinsic.txt";
    threshold_lidar = 30000;

    src_img = cv::imread(input_photo_path);
    if (src_img.empty()) {
        cout << "No Picture found by filename: " << input_photo_path << endl;
        return 0;
    }

    vector<PointXYZRGB> pointcloud;
    loadPointcloudFromDir(input_pcd_dir, pointcloud);

    vector<float> intrinsic;
    getIntrinsic(intrinsic_path, intrinsic);
    vector<float> distortion;
    getDistortion(intrinsic_path, distortion);
    vector<float> extrinsic;
    getExtrinsic(extrinsic_path, extrinsic);

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = intrinsic[0];
    cameraMatrix.at<double>(0, 2) = intrinsic[2];
    cameraMatrix.at<double>(1, 1) = intrinsic[4];
    cameraMatrix.at<double>(1, 2) = intrinsic[5];

    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    for (int i = 0; i < 5; i++) distCoeffs.at<double>(i, 0) = distortion[i];

    cout << "Start to project the lidar cloud..." << endl;
    float theoryUV[2] = {0, 0};
    int myCount = 0;

    for (auto& pt : pointcloud) {
        float x = pt.x, y = pt.y, z = pt.z;
        getTheoreticalUV(theoryUV, intrinsic, extrinsic, x*1000, y*1000, z*1000);

        int u = floor(theoryUV[0] + 0.5);
        int v = floor(theoryUV[1] + 0.5);
        if (u < 0 || u >= src_img.cols || v < 0 || v >= src_img.rows) continue;

        int r, g, b;
        getColor(r, g, b, x);

        cv::circle(src_img, Point(u, v), 5, Scalar(b, g, r), -1);

        ++myCount;
        if (myCount > threshold_lidar) break;
    }

    cv::Size imageSize = src_img.size();
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),
        cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
        imageSize, CV_16SC2, map1, map2);
    cv::remap(src_img, src_img, map1, map2, cv::INTER_LINEAR);

    cv::imshow("source", src_img);
    cv::waitKey(0);
    cv::imwrite(output_path, src_img);

    return 0;
}
