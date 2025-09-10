
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "common.h"
#include "result_verify.h"

#include <opencv2/opencv.hpp>
#include <stdexcept>
// ------------------- 残差类：仅优化 yaw + 平移 -------------------
class ExternalCaliOcamYaw
{
public:
    ExternalCaliOcamYaw(PnPData p, const OcamModel &model) : pd(p), ocam_model(model) {}

    template <typename T>
    bool operator()(const T *yaw, const T *t, T *residuals) const
    {
        // Rz(yaw)
        const T cy = ceres::cos(yaw[0]);
        const T sy = ceres::sin(yaw[0]);

        Eigen::Matrix<T, 3, 3> R;
        R <<  cy, -sy, T(0),
              sy,  cy, T(0),
              T(0), T(0), T(1);

        Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
        Eigen::Matrix<T, 3, 1> tvec(t[0], t[1], t[2]);

        Eigen::Matrix<T, 3, 1> p_c = R * p_l + tvec;

        // Ocam 通常只需要方向，做单位化
        T norm = p_c.norm();
        if (norm != T(0)) p_c /= norm;

        Eigen::Matrix<T, 2, 1> uv = world2cam1(p_c, ocam_model);
        residuals[0] = uv[0] - T(pd.u);
        residuals[1] = uv[1] - T(pd.v);
        return true;
    }

    static ceres::CostFunction *Create(PnPData p, const OcamModel &model)
    {
        return (new ceres::AutoDiffCostFunction<ExternalCaliOcamYaw, 2, 1, 3>(
            new ExternalCaliOcamYaw(p, model)));
    }

private:
    PnPData pd;
    const OcamModel ocam_model;
};

// 小工具：角度转弧度 & Rz(yaw)（double版，用于写文件）
inline double Deg2Rad(double deg) { return deg * M_PI / 180.0; }
inline Eigen::Matrix3d Rz_from_yaw(double yaw_rad)
{
    double c = std::cos(yaw_rad), s = std::sin(yaw_rad);
    Eigen::Matrix3d R;
    R <<  c, -s, 0,
          s,  c, 0,
          0,  0, 1;
    return R;
}
// 生成绕X轴旋转90度倍数的矩阵
Eigen::Matrix3f rotation_x(int degrees)
{
    float rad = degrees * M_PI / 180.0f;
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);

    Eigen::Matrix3f R;
    R << 1.0f, 0.0f, 0.0f,
        0.0f, cos_a, -sin_a,
        0.0f, sin_a, cos_a;
    return R;
}

// 生成绕Y轴旋转90度倍数的矩阵
Eigen::Matrix3f rotation_y(int degrees)
{
    float rad = degrees * M_PI / 180.0f;
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);

    Eigen::Matrix3f R;
    R << cos_a, 0.0f, sin_a,
        0.0f, 1.0f, 0.0f,
        -sin_a, 0.0f, cos_a;
    return R;
}

// 生成绕Z轴旋转90度倍数的矩阵
Eigen::Matrix3f rotation_z(int degrees)
{
    float rad = degrees * M_PI / 180.0f;
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);

    Eigen::Matrix3f R;
    R << cos_a, -sin_a, 0.0f,
        sin_a, cos_a, 0.0f,
        0.0f, 0.0f, 1.0f;
    return R;
}

// 比较两个矩阵是否相等（考虑浮点误差）
bool matrices_equal(const Eigen::Matrix3f &a, const Eigen::Matrix3f &b, float tolerance = 1e-6f)
{
    return (a - b).norm() < tolerance;
}

// 生成所有90度旋转组合
std::vector<Eigen::Matrix3f> generate_all_90_degree_rotations()
{
    std::vector<Eigen::Matrix3f> rotations;
    std::vector<Eigen::Matrix3f> unique_rotations;

    // 所有可能的旋转角度（90度的倍数）
    std::vector<int> angles = {0, 90, 180, 270};

    // 生成所有可能的旋转组合
    for (int x : angles)
    {
        Eigen::Matrix3f Rx = rotation_x(x);

        for (int y : angles)
        {
            Eigen::Matrix3f Ry = rotation_y(y);

            for (int z : angles)
            {
                Eigen::Matrix3f Rz = rotation_z(z);

                // 组合旋转：R = Rz * Ry * Rx
                Eigen::Matrix3f R = Rz * Ry * Rx;
                rotations.push_back(R);
            }
        }
    }

    // 去除重复的矩阵
    for (const auto &R : rotations)
    {
        bool is_duplicate = false;
        for (const auto &existing : unique_rotations)
        {
            if (matrices_equal(R, existing))
            {
                is_duplicate = true;
                break;
            }
        }
        if (!is_duplicate)
        {
            unique_rotations.push_back(R);
        }
    }

    return unique_rotations;
}

// 打印旋转矩阵
void print_rotation(const Eigen::Matrix3f &R, int index)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            // 将接近0的值显示为0，接近±1的值显示为±1
            float value = R(i, j);
            if (std::abs(value) < 1e-6f)
                value = 0.0f;
            else if (std::abs(value - 1.0f) < 1e-6f)
                value = 1.0f;
            else if (std::abs(value + 1.0f) < 1e-6f)
                value = -1.0f;
        }
    }
}
// 可选：float 坐标的重载（会就近取整到像素中心）
inline bool drawDot(cv::Mat &img, const cv::Point2f &ptf,
                    const cv::Scalar &bgr = {0, 0, 255},
                    int radius = 3, int thickness = cv::FILLED, int lineType = cv::LINE_AA)
{
    return drawDot(img, cv::Point(cvRound(ptf.x), cvRound(ptf.y)), bgr, radius, thickness, lineType);
}

void print_polynomial(const std::vector<double> &poly)
{
    for (size_t i = 0; i < poly.size(); ++i)
    {
        std::cout << std::fixed << std::setprecision(17) << poly[i];
        if (i != poly.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";
}

// ------------------- 多项式求值（支持泛型 T） -------------------
template <typename T>
T polyval(const std::vector<double> &coeffs, const T &x)
{
    T y = T(0);
    for (size_t i = 0; i < coeffs.size(); ++i)
    {
        y = y * x + T(coeffs[i]);
    }
    return y;
}

// ------------------- world2cam（支持自动求导） -------------------
template <typename T>
Eigen::Matrix<T, 2, 1> world2cam1(const Eigen::Matrix<T, 3, 1> &point3D, const OcamModel &model)
{
    // std::cout<<"SBSBSIN"<<point3D<<std::endl;
    // exit(0);
    T u = T(0), v = T(0);
    T xc = T(model.xc);
    T yc = T(model.yc);
    T c = T(model.c);
    T d = T(model.d);
    T e = T(model.e);
    T norm = sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1]);
    if (norm != T(0))
    {
        T invnorm = T(1) / norm;
        T theta = atan(point3D[2] / norm);
        T rho = polyval(model.invpol, theta);

        T x = point3D[0] * invnorm * rho;
        T y = point3D[1] * invnorm * rho;

        u = x * c + y * d + xc;
        v = x * e + y + yc;
    }
    else
    {
        u = xc;
        v = yc;
    }
    Eigen::Matrix<T, 2, 1> uv;
    uv << u, v;
    return uv;
}

// ------------------- 残差类 -------------------
class ExternalCaliOcam
{
public:
    ExternalCaliOcam(PnPData p, const OcamModel &model) : pd(p), ocam_model(model) {}

    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residuals) const
    {

        Eigen::Quaternion<T> q_incre(_q[3], _q[0], _q[1], _q[2]);
        Eigen::Matrix<T, 3, 1> t_incre(_t[0], _t[1], _t[2]);

        Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));

        // std::cout<<p_l<<std::endl;
        Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
        T norm = p_c.norm();
        if (norm != T(0))
        {
            p_c /= norm; // Normalize the point to unit length
        }
        Eigen::Matrix<T, 2, 1> uv = world2cam1(p_c, ocam_model);
        // std::cout << "world2cam1OUT:" << uv<< std::endl;

        residuals[0] = uv[0] - T(pd.u);
        residuals[1] = uv[1] - T(pd.v);
        return true;
    }

    static ceres::CostFunction *Create(PnPData p, const OcamModel &model)
    {
        return (new ceres::AutoDiffCostFunction<ExternalCaliOcam, 2, 4, 3>(
            new ExternalCaliOcam(p, model)));
    }

private:
    PnPData pd;
    const OcamModel ocam_model;
};
int main()
{
    OcamModel ocam_model;
    std::string intrinsic_path = "../a.txt";            // 全景内参
    int error_threshold = 12;

    get_ocam_model(ocam_model, intrinsic_path);

    std::vector<PnPData> pData;
    std::string lidar_path = "./lidar_point.txt";       // 雷达标注点
    std::string photo_path = "./cam_point.txt";         // 全景标注点
    std::string extrinsic_path = "./extrinsic_pano.txt";// 外参输出路径
    std::string vaild_path = "/Users/losehu/Documents/20250813";
    getData(lidar_path, photo_path, pData);

    // 遍历一些 yaw 初值（度）
    std::vector<int> yaw_seeds_deg = {0, 90, 180, 270};
    double best_sum_err = 1e10;
    Eigen::Matrix3d best_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d best_t(0,0,0);

    for (int seed_deg : yaw_seeds_deg)
    {
        // 参数：yaw(弧度) + t(3)
        double yaw = Deg2Rad(seed_deg);
        double t[3] = {0.0, 0.0, 0.0};   // 若你有平移初值，可改在这里

        ceres::Problem problem;
        problem.AddParameterBlock(&yaw, 1);
        problem.AddParameterBlock(t, 3);

        for (const auto &val : pData)
        {
            ceres::CostFunction *cost = ExternalCaliOcamYaw::Create(val, ocam_model);
            problem.AddResidualBlock(cost, nullptr, &yaw, t);
        }

        // 若只想优化 yaw，锁死平移：
        // problem.SetParameterBlockConstant(t);
        // 若只想优化平移，锁死 yaw：
        // problem.SetParameterBlockConstant(&yaw);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << "Seed yaw = " << seed_deg << " deg -> " << summary.BriefReport() << std::endl;

        // 评估 & 记录最优
        Eigen::Matrix3d R_opt = Rz_from_yaw(yaw);
        Eigen::Vector3d t_opt(t[0], t[1], t[2]);

        // 写出以便复用你现有的评估函数
        writeExt(extrinsic_path, R_opt, t_opt);

        float error_uv[2] = {0.f, 0.f};
        getUVError_pano(intrinsic_path, extrinsic_path, lidar_path, photo_path,
                        error_uv, error_threshold, vaild_path, /*save_debug=*/false);

        double sum_err = error_uv[0] + error_uv[1];
        std::cout << "  -> reproj err sum = " << sum_err << " (u=" << error_uv[0] << ", v=" << error_uv[1] << ")\n";

        if (sum_err < best_sum_err)
        {
            best_sum_err = sum_err;
            best_R = R_opt;
            best_t = t_opt;
        }
    }

    // 输出全局最优
    writeExt(extrinsic_path, best_R, best_t);
    std::cout << "最好的重投影误差和: " << best_sum_err << std::endl;
    return 0;
}
