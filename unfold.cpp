#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "cmath"
#include <fstream>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ceres/ceres.h>
#include <fstream>
#include <vector>
#include <string>
namespace fs = std::filesystem;

struct OcamModel {
    std::vector<double> pol;
    int length_pol = 0;

    std::vector<double> invpol;
    int length_invpol = 0;

    double xc = 0.0;
    double yc = 0.0;

    double c = 0.0;
    double d = 0.0;
    double e = 0.0;

    int width = 0;
    int height = 0;
};
void print_polynomial(const std::vector<double>& poly) {
    for (size_t i = 0; i < poly.size(); ++i) {
        std::cout << std::fixed << std::setprecision(17) << poly[i];
        if (i != poly.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";
}

void get_ocam_model(OcamModel& model, const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;

    // Skip 2 lines
    std::getline(file, line);
    std::getline(file, line);

    // Read polynomial coefficients
    std::getline(file, line);
    std::istringstream iss_poly(line);
    iss_poly >> model.length_pol;
    model.pol.resize(model.length_pol);
    for (int i = 0; i < model.length_pol; ++i) {
        iss_poly >> model.pol[i];
    }

    // Skip 3 lines
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    // Read inverse polynomial coefficients
    std::getline(file, line);
    std::istringstream iss_invpoly(line);
    iss_invpoly >> model.length_invpol;
    model.invpol.resize(model.length_invpol);
    for (int i = 0; i < model.length_invpol; ++i) {
        iss_invpoly >> model.invpol[i];
    }

    // Skip 3 lines
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    // Read center coordinates
    std::getline(file, line);
    std::istringstream iss_center(line);
    iss_center >> model.xc >> model.yc;
    std::cout << "Center: (" << model.xc << ", " << model.yc << ")\n";
    // Skip 3 lines
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    // Read affine parameters
    std::getline(file, line);
    std::istringstream iss_affine(line);
    iss_affine >> model.c >> model.d >> model.e;

    // Skip 3 lines
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);

    // Read image size
    std::getline(file, line);
    std::istringstream iss_size(line);
    iss_size >> model.height >> model.width;

    file.close();


    std::cout << "******** camera param *********" <<"\n\n";
    std::cout << std::fixed << std::setprecision(15);
    std::cout << "Center: (" << model.xc << ", " << model.yc << ")\n";
    std::cout << "c,d,e: (" << model.c << ", " << model.d << ", " << model.e << ")\n";

    std::cout << "Image size: " << model.width << " x " << model.height << "\n";
    std::cout << "First pol coefficient: " ;
    print_polynomial(model.invpol);
    std::cout << "\n******** camera param end *********" <<"\n\n";

}
double polyval(const std::vector<double>& coeffs, double x) {
    double y = 0.0;

    for (size_t i = 0; i < coeffs.size(); ++i) {
        y=y*x+coeffs[i];
    }
    return y;
}



cv::Vec3d cam2world(const cv::Point2d& point2D, const OcamModel& model) {
    cv::Vec3d point3D;
    double xc = model.xc;
    double yc = model.yc;
    double c = model.c;
    double d = model.d;
    double e = model.e;
    int length_pol = model.length_pol;
    double invdet = 1/(c-d*e);  // 1/det(A), where A = [c,d;e,1] as in the Matlab file
    double xp = 1/(c-d*e)*((point2D.x - xc) - d*(point2D.y - yc));
    double yp = 1/(c-d*e)*(-e*(point2D.x - xc) + c*(point2D.y - yc));
    //根据x，y轴像素变形程度，计算世界的x，y坐标系

    double r = sqrt(xp*xp + yp*yp);  // distance of the point from image center

    double zp = model.pol[0];
    double r_i = 1;
    for (int i = 1; i < length_pol; ++i) {
        r_i *= r;
        zp += r_i*model.pol[i];
    }
    //根据半径计算多项式得到距离（鱼眼模型特性）
    // normalize to unit norm
    double invnorm = 1/sqrt(xp*xp + yp*yp + zp*zp);
    point3D[0] = invnorm*xp;
    point3D[1] = invnorm*yp;
    point3D[2] = invnorm*zp;
    return point3D;
}


cv::Point2d world2cam(const cv::Vec3d& point3D, const OcamModel& model) {
    // std::cout<<point3D[0]<<","<<point3D[1]<<","<<point3D[2]<<std::endl;
    double u=0 ,v=0;
    double xc = model.xc;
    double yc = model.yc;
    double c = model.c;
    double d = model.d;
    double e = model.e;
    cv::Point2d point2D[2];
    // #3D点在2D平面的投影长度
    double norm = std::sqrt(point3D[0]*point3D[0] + point3D[1]*point3D[1]);
    // #3D点与平面的夹角
    double theta = std::atan(point3D[2]/norm);

    if (norm != 0) {
        double invnorm = 1/norm;
        double t = theta;
        double rho = model.invpol[0];
        rho=polyval(model.invpol, t);

        // #rho是畸变半径
        double x = point3D[0]*invnorm*rho;
        double y = point3D[1]*invnorm*rho;
        u = x*c + y*d + xc;
        v = x*e + y + yc;

    } else {
        u = xc;
        v = yc;
    }
    u=int(u+0.5);
    v=int(v+0.5);
    return cv::Point2d(u, v);
}
cv::Vec3d lonlat_to_unit_vector(double azimuth_deg, double elevation_deg) {
    double az = azimuth_deg * CV_PI / 180.0;
    double el = elevation_deg * CV_PI / 180.0;

    double x = cos(el) * cos(az);
    double y = cos(el) * sin(az);
    double z = sin(el);
    return cv::Vec3d(x, y, z);
}
void create_panoramic_lonlat(cv::Mat& mapx, cv::Mat& mapy, const OcamModel& model, const std::vector<double>& angle) {
    int width = mapx.cols;
    int height = mapx.rows;
    double i = 0, j = 0;
    std::cout<<height<<","<<width<<std::endl;
    while (i < height) {
        j=0;
        while (j < width) {
            double lat = (-angle[0] / 180 + i / height * (angle[0] + angle[1]) / 180) * CV_PI;
            double lon = -(-angle[2] / 180 + j / width * (angle[2] + angle[3]) / 180) * CV_PI + CV_PI;
            cv::Vec3d point3D = cv::Vec3d(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat));
            cv::Point2d point2D = world2cam(point3D, model);
               cv::Vec3d point3D_1=cam2world(point2D, model);
            //    std::cout<<std::endl;
            //    std::cout<<point2D.x<<point2D.y<<std::endl;
            //     std::cout<<point3D[0]<<","<<point3D[1]<<","<<point3D[2]<<std::endl;
            //    std::cout<<point3D_1[0]<<","<<point3D_1[1]<<","<<point3D_1[2]<<std::endl;
                        //   std::cout<<std::endl;

               mapx.at<float>(i, j) = point2D.y;
            mapy.at<float>(i, j) = point2D.x;
            j = j + 1;
        }
        i = i + 1;
    }
}

void get_lonlat_map(const OcamModel& model,
                    const std::vector<double>& fov,
                    const std::vector<double>& angle,
                    cv::Mat& mapx, cv::Mat& mapy) {
    double pixel_h = model.height * (fov[1] - fov[0]) / 2 / fov[1];

    std::vector<double> size_pan_img = {round(pixel_h*(angle[0]+angle[1])/(fov[1]-fov[0])),
                    round(pixel_h*(angle[2]+angle[3])/(fov[1]-fov[0]))};
    mapx = cv::Mat(size_pan_img[0], size_pan_img[1], CV_32FC1);
    mapy = cv::Mat(size_pan_img[0], size_pan_img[1], CV_32FC1);
    create_panoramic_lonlat(mapx, mapy, model, angle);
}
cv::Mat ocam_lonlat(const cv::Mat& img, const cv::Mat& mapx, const cv::Mat& mapy) {
    cv::Mat dst_pan;
    cv::remap(img, dst_pan, mapx, mapy, cv::INTER_LINEAR);
    cv::flip(dst_pan, dst_pan, 1); // 1 表示水平翻转（左右镜像）
    return dst_pan;
}




void process_images_in_folder(const std::string& input_folder, const std::string& output_folder, const OcamModel& model, const std::vector<double>& fov, const std::vector<double>& angle) {
    // Check if the output folder exists, if not create it
    if (!fs::exists(output_folder)) {
        fs::create_directory(output_folder);
    }

    // Iterate through all jpg files in the input folder
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            // Load image
            cv::Mat img = cv::imread(entry.path().string());
            if (img.empty()) {
                std::cerr << "Failed to load image: " << entry.path() << std::endl;
                continue;
            }

            // Get the panoramic transformation map
            cv::Mat mapx, mapy;
            get_lonlat_map(model, fov, angle, mapx, mapy);

            // Apply the transformation to the image
            cv::Mat raw_img = ocam_lonlat(img, mapx, mapy);

            // Construct output image path
            std::string output_file = output_folder + "/" + entry.path().filename().string();

            // Save the transformed image
            cv::imwrite(output_file, raw_img);
            std::cout << "Processed and saved: " << output_file << std::endl;
        }
    }
}

int main() {
    OcamModel model;
    get_ocam_model(model, "./calib/intrinsic_pano.txt");
    cv::Vec3d point3D = cv::Vec3d(-0.28837 , -0.452965, 0.843603);
    std::cout << "Point in world coordinates: (" << point3D[0] << ", " << point3D[1] << ", " << point3D[2] << ")" << std::endl;
    cv::Point2d point2D = world2cam(point3D, model);
    std::cout << "Point in camera coordinates: (" << point2D.x << ", " << point2D.y << ")" << std::endl;
    std::vector<double> fov = {0, 90}; // Image FOV
    std::vector<double> angle = {40.0, 20.0, 180.0, 180.0}; // FOV limits: up/down/left/right

    // Specify input and output folders
    std::string input_folder = "/Users/losehu/Documents/20250813/"; // Folder containing JPG images
    std::string output_folder = "/Users/losehu/Documents/20250813/pic"; // Folder to save the processed images

    // Process all images in the input folder and save them to the output folder
    process_images_in_folder(input_folder, output_folder, model, fov, angle);

    return 0;
}
