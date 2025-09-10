#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <ctime>

#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>

using namespace std;
using namespace cv;

#define RADIAN_TO_DEGREE(r) ((r) / CV_PI * 180.0)
#define DEGREE_TO_RADIAN(d) ((d) * CV_PI / 180.0)

// ����ͼ�����Ұ�����㽹��
double focal_calc(int width, double fov)
{
	return (width / 2.0) / tan(DEGREE_TO_RADIAN(fov) / 2.0);
}

// ����ͼ������࣬������Ұ
double fov_calc(int width, double focal)
{
	return RADIAN_TO_DEGREE(atan(width / 2.0 / focal) * 2.0);
}

// ����ͼ���ߺ�ˮƽ�ӽǣ������ڲ�������
Mat K_calc(int w, int h, float hfov)
{
	double focal = focal_calc(w, hfov);
	return (Mat_<double>(3, 3) << focal, 0, w / 2.0, 0, focal, h / 2.0, 0, 0, 1);
}

int main(int argc, char **argv)
{
	int num = 4;
	int src_w = 0, src_h = 0;
	vector<Mat> K(num), D(num), R(num), T(num);

	string data_path = "08_01_05_922692";

	for (int i = 0; i < num; i++)
	{
		// Step 1������ȡ�ڲ�K��D
		string fn = data_path + "/calib_cmos" + to_string(i) + ".xml";
		FileStorage fs(fn, FileStorage::READ);
		fs["camera_matrix"] >> K[i];
		fs["distortion_coefficients"] >> D[i];
		fs["image_width"] >> src_w;
		fs["image_height"] >> src_h;
		fs.release();

		cout << "Width: " << src_w << ". Height: " << src_h << endl;
		cout << "K: " << endl << K[i] << endl;
		cout << "D: " << endl << D[i] << endl;
		cout << endl;

		// Step 2 -- ����ӳ�����
		int undist_w = 2000; //���������
		int undist_h = 2000; //���������
		float undist_fov = 120.0; //���������Ұ�Ƕȣ�ˮƽ�Ƕȣ�
		Mat R_undist = Mat::eye(3, 3, CV_32F);
		Mat K_undist = K_calc(undist_w, undist_h, undist_fov);
		cout << K_undist << endl;

		Mat map_x, map_y;
		Mat img_undist;
		if (D[i].rows == 5)
			initUndistortRectifyMap(K[i], D[i], R_undist, K_undist, Size(undist_w, undist_h), CV_32FC1, map_x, map_y);
		else if (D[i].rows == 4)
			fisheye::initUndistortRectifyMap(K[i], D[i], R_undist, K_undist, Size(undist_w, undist_h), CV_32FC1, map_x, map_y);

		// Step 3 -- ��ȡͼ�񡢽���ͼ�񡢱���ͼ��
		Mat img_input = imread(data_path + "/" + to_string(i) + ".jpg");
		cv::remap(img_input, img_undist, map_x, map_y, INTER_CUBIC);
		imwrite(data_path + "/undist_" + to_string(i) + ".jpg", img_undist);
	}
	return 0;
}
