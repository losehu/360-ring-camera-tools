#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <fstream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <regex>
#include <algorithm>
#include <filesystem> // ★ 新增

#include "common.h"
#include <sstream>
namespace fs = std::filesystem;
#ifdef __APPLE__
#include <cstdio>
#include <cstdlib>
void silenceFrameworkLogs()
{
    // 关闭 Apple 的统一日志（对 os_log 有效）
    setenv("OS_ACTIVITY_MODE", "disable", 1);
    // 把 stderr 重定向到 /dev/null，屏蔽 NSLog/fprintf(stderr, ...)
    freopen("/dev/null", "w", stderr);
}
#endif

// ============ 画球工具 ============
bool drawSphereAtMeters(pcl::visualization::PCLVisualizer::Ptr viewer,
                        double x_m, double y_m, double z_m,
                        double radius_m = 0.30,
                        const std::string &id = "marker",
                        double r = 0.0, double g = 1.0, double b = 0.0,
                        double opacity = 0.6)
{
    if (!viewer)
        return false;
    viewer->removeShape(id);
    pcl::PointXYZ center(x_m, y_m, z_m);
    if (!viewer->addSphere(center, radius_m, r, g, b, id))
        return false;
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, id);
    return true;
}
bool drawSphereAtMillimeters(pcl::visualization::PCLVisualizer::Ptr viewer,
                             double x_mm, double y_mm, double z_mm,
                             double radius_mm = 300.0,
                             const std::string &id = "marker_mm",
                             double r = 0.0, double g = 1.0, double b = 0.0,
                             double opacity = 0.6)
{
    const double s = 0.001; // mm -> m
    return drawSphereAtMeters(viewer, x_mm * s, y_mm * s, z_mm * s,
                              radius_mm * s, id, r, g, b, opacity);
}

#define cam_path "saved_cam.cam"
pcl::visualization::PCLVisualizer::Ptr viewer(
    new pcl::visualization::PCLVisualizer("viewer"));

// —— 点选上限与记录 —— //
static const int MAX_POINTS = 4;
static std::vector<std::string> picked_ids;    // 点选球的ID
static std::vector<cv::Point3d> picked_pts_mm; // 点选坐标（mm）

// —— PCD 列表与游标 —— //
static std::vector<std::string> g_pcds;
static int g_cur_idx = -1;

// ============ 相机IO ============
static bool save_cam(const std::string &path, const pcl::visualization::Camera &cam)
{
    std::ofstream out(path);
    if (!out)
    {
        std::cerr << "Failed to open file for writing: " << path << std::endl;
        return false;
    }
    out << "Clipping plane [near,far] " << cam.clip[0] << ", " << cam.clip[1] << "\n";
    out << "Focal point [x,y,z] " << cam.focal[0] << ", " << cam.focal[1] << ", " << cam.focal[2] << "\n";
    out << "Position [x,y,z] " << cam.pos[0] << ", " << cam.pos[1] << ", " << cam.pos[2] << "\n";
    out << "View up [x,y,z] " << cam.view[0] << ", " << cam.view[1] << ", " << cam.view[2] << "\n";
    out << "Camera view angle [degrees] " << cam.fovy << "\n";
    out << "Window size [x,y] " << cam.window_size[0] << ", " << cam.window_size[1] << "\n";
    out << "Window position [x,y] " << cam.window_pos[0] << ", " << cam.window_pos[1] << "\n";
    std::cout << "Camera parameters saved to: " << path << std::endl;
    return true;
}
static bool load_cam(const std::string &path, pcl::visualization::Camera &cam)
{
    std::ifstream in(path);
    if (!in)
        return false;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.find("Position") != std::string::npos)
            std::sscanf(line.c_str(), "Position [x,y,z] %lf, %lf, %lf", &cam.pos[0], &cam.pos[1], &cam.pos[2]);
        else if (line.find("Focal point") != std::string::npos)
            std::sscanf(line.c_str(), "Focal point [x,y,z] %lf, %lf, %lf", &cam.focal[0], &cam.focal[1], &cam.focal[2]);
        else if (line.find("View up") != std::string::npos)
            std::sscanf(line.c_str(), "View up [x,y,z] %lf, %lf, %lf", &cam.view[0], &cam.view[1], &cam.view[2]);
        else if (line.find("Clipping plane") != std::string::npos)
            std::sscanf(line.c_str(), "Clipping plane [near,far] %lf, %lf", &cam.clip[0], &cam.clip[1]);
        else if (line.find("Camera view angle") != std::string::npos)
            std::sscanf(line.c_str(), "Camera view angle [degrees] %lf", &cam.fovy);
        else if (line.find("Window size") != std::string::npos)
            std::sscanf(line.c_str(), "Window size [x,y] %lf, %lf", &cam.window_size[0], &cam.window_size[1]);
        else if (line.find("Window position") != std::string::npos)
            std::sscanf(line.c_str(), "Window position [x,y] %lf, %lf", &cam.window_pos[0], &cam.window_pos[1]);
    }
    return true;
}

// ============ 辅助：根据文件名提取 pcd(\d+) ============
static bool extract_need_id_from_path(const std::string &pcd_path, int &need_id)
{
    std::string stem = fs::path(pcd_path).stem().string(); // 如 "1" 或 "pcd1" 或 "pcd_001"
    std::regex re(R"((?:pcd)?[_\- ]*(\d+))", std::regex::icase);
    std::smatch m;
    if (std::regex_search(stem, m, re))
    {
        need_id = std::stoi(m[1]);
        return true;
    }
    return false;
}

// ============ 绘制与当前 PCD 匹配的 LiDAR 球 ============
static void drawLidarMarkersForId(int need_id)
{
    std::ifstream inFile_lidar("./yuyan_point.txt");
    if (!inFile_lidar.is_open())
        return;

    std::string lineStr_lidar;
    int now_cnt = 0;
    while (std::getline(inFile_lidar, lineStr_lidar))
    {
        if (now_cnt / 9 == need_id - 1)
        {
            if (lineStr_lidar.size() > 10)
            {
                double x, y, z;
                std::string str;
                std::stringstream line_lidar(lineStr_lidar);
                line_lidar >> str;
                x = str2double(str);
                line_lidar >> str;
                y = str2double(str);
                line_lidar >> str;
                z = str2double(str);

                std::string name = "big_green_ball" + std::to_string((now_cnt % 9) / 2);
                if ((now_cnt % 9) /2 == 1)
                    drawSphereAtMillimeters(viewer, x, y, z, /*radius_mm=*/100.0, /*id=*/name, 1, 0, 0);
                else if ((now_cnt % 9) / 2 == 2)
                    drawSphereAtMillimeters(viewer, x, y, z, /*radius_mm=*/100.0, /*id=*/name, 0, 1, 0);
                else if ((now_cnt % 9) / 2 == 3)
                    drawSphereAtMillimeters(viewer, x, y, z, /*radius_mm=*/100.0, /*id=*/name, 0, 0, 1);
                else 
                                    drawSphereAtMillimeters(viewer, x, y, z, /*radius_mm=*/100.0, /*id=*/name, 0, 1, 1);


            }
        }
        now_cnt += 1;
    }
}
// ============ 核心：加载并显示某个 PCD（清理旧状态，叠加激光球） ============
static bool loadOnePCDAndSetup(const std::string &pcd_path)
{
    // 清理旧形状（含之前的点选球 & 激光球）
    viewer->removeAllShapes();

    // 清空点选记录
    picked_ids.clear();  // Clear the picked IDs
    picked_pts_mm.clear();  // Clear the picked points

    // 替换点云
    viewer->removePointCloud("cloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(pcd_path, *cloud) != 0)
    {
        std::cerr << "Failed to load: " << pcd_path << std::endl;
        return false;
    }

    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> color_by_z(cloud, "z");
    if (color_by_z.isCapable())
        viewer->addPointCloud<pcl::PointXYZ>(cloud, color_by_z, "cloud");
    else
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud, 255, 0, 0);
        viewer->addPointCloud<pcl::PointXYZ>(cloud, red, "cloud");
    }
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

    // 解析 need_id 并叠加 LiDAR 球
    int need_id = 1;
    if (!extract_need_id_from_path(pcd_path, need_id))
        std::cout << "[id] not found\n";
    drawLidarMarkersForId(need_id);

    std::cout << fs::path(pcd_path).filename().string() << std::endl;
    return true;
}


// ============ 键盘回调 ============
void keyboardCallback(const pcl::visualization::KeyboardEvent &event, void *viewer_void)
{
    auto viewer = static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);

    if ((event.getKeySym() == "p" || event.getKeySym() == "P") && event.keyDown())
    {
        pcl::visualization::Camera cam;
        viewer->getCameraParameters(cam);
        save_cam("saved_cam.cam", cam);
    }
    if ((event.getKeySym() == "l" || event.getKeySym() == "L") && event.keyDown())
    {
        pcl::visualization::Camera cam;
        if (load_cam("saved_cam.cam", cam))
        {
            auto renWin = viewer->getRenderWindow();
            int *sz = renWin->GetSize();
            cam.window_size[0] = sz[0];
            cam.window_size[1] = sz[1];

            viewer->setCameraParameters(cam);
            viewer->getRenderWindow()->Render();
            // 让裁剪范围和投影矩阵刷新一下
            viewer->getRendererCollection()->GetFirstRenderer()->ResetCameraClippingRange();
            // std::cout << "Loaded camera parameters from file\n";
        }
        else
        {
            std::cerr << "Failed to load camera parameters from file\n";
        }
    }
    if ((event.getKeySym() == "b" || event.getKeySym() == "B") && event.keyDown())
    {
        if (!picked_ids.empty())
        {
            const std::string last_id = picked_ids.back();
            viewer->removeShape(last_id);
            picked_ids.pop_back();
            if (!picked_pts_mm.empty())
                picked_pts_mm.pop_back();
            // 刷新一下
            viewer->getRendererCollection()->GetFirstRenderer()->ResetCameraClippingRange();
            viewer->getRenderWindow()->Render();
            viewer->spinOnce(1);
        }
    }
    // —— 按 Enter（Return/KP_Enter）打印“当前所有已经点过的点”，然后切到下一张 —— //
    if ((event.getKeySym() == "Return" || event.getKeySym() == "KP_Enter") && event.keyDown())
    {
        // 1) 打印当前帧所有已点坐标（mm）
        for (size_t i = 0; i < picked_pts_mm.size(); ++i)
        {
            const auto &p = picked_pts_mm[i];
            int seq = static_cast<int>(i) + 1; // 1..N
            std::cout << seq << "\n"
                      << p.x << " " << p.y << " " << p.z << "\n";
        }

        // 2) 切到下一张 PCD（循环）
        if (!g_pcds.empty())
        {
            if (g_cur_idx + 1 < (int)g_pcds.size())
                g_cur_idx = (g_cur_idx + 1) % (int)g_pcds.size();
            else
                exit(0);
            loadOnePCDAndSetup(g_pcds[g_cur_idx]);
        }
    }

    // 下一帧
    if (event.getKeySym() == "n" && event.keyDown())
    {
        if (!g_pcds.empty())
        {
            g_cur_idx = (g_cur_idx + 1) % g_pcds.size();
            loadOnePCDAndSetup(g_pcds[g_cur_idx]);
        }
    }
    // 上一帧（大写 N）
    if (event.getKeySym() == "N" && event.keyDown())
    {
        if (!g_pcds.empty())
        {
            g_cur_idx = (g_cur_idx - 1 + (int)g_pcds.size()) % (int)g_pcds.size();
            loadOnePCDAndSetup(g_pcds[g_cur_idx]);
        }
    }
}

// 前向声明
void visualization();

// ============ main ============
int main(int argc, char **argv)
{

#ifdef __APPLE__
    silenceFrameworkLogs();
#endif
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <PCD file or directory>" << std::endl;
        return 1;
    }

    const std::string input_path = argv[1];

    // 收集PCD列表（目录或单文件）
    if (fs::is_directory(input_path))
    {
        for (auto &entry : fs::directory_iterator(input_path))
        {
            if (!entry.is_regular_file())
                continue;
            if (entry.path().extension() == ".pcd" || entry.path().extension() == ".PCD")
                g_pcds.push_back(entry.path().string());
        }
        std::sort(g_pcds.begin(), g_pcds.end());
    }
    else
    {
        g_pcds.push_back(input_path);
    }

    if (g_pcds.empty())
    {
        std::cerr << "No PCD files found." << std::endl;
        return 1;
    }

    // 可选：加载相机参数
    pcl::visualization::Camera cam;
    if (load_cam("saved_cam.cam", cam))
    {
        auto renWin = viewer->getRenderWindow();
        int *sz = renWin->GetSize(); // sz[0] = width, sz[1] = height
        int w = sz[0];
        int h = sz[1];

        // 如果你要把相机里的 window_size 覆盖为当前窗口尺寸：
        cam.window_size[0] = w;
        cam.window_size[1] = h;

        viewer->setCameraParameters(cam);
        // 或者你想把窗口调整到相机里记录的尺寸：
        // viewer->setSize(static_cast<int>(cam.window_size[0]), static_cast<int>(cam.window_size[1]));

        // std::cout << "Loaded camera parameters from file" << std::endl;
    }
    else
    {
        std::cerr << "Failed to load camera parameters from file" << std::endl;
    }
    std::cout << "> Point picking enabled.  [n] next, [N] prev, [b] undo, [Enter] print all\n";

    // 先加载第一帧
    g_cur_idx = 0;
    if (!loadOnePCDAndSetup(g_pcds[g_cur_idx]))
        return 1;

    // 进入交互
    visualization();
    return 0;
}

// ============ 只做回调注册与spin ============
void visualization()
{

    // 键盘回调
    viewer->registerKeyboardCallback(keyboardCallback, (void *)viewer.get());

    // 点选回调（不打印，按Enter再统一打印）
    viewer->registerPointPickingCallback(
        [](const pcl::visualization::PointPickingEvent &e, void *)
        {
            if (e.getPointIndex() == -1)
                return;
            if (picked_ids.size() >= MAX_POINTS)
                return;

            float x, y, z;
            e.getPoint(x, y, z);

            int seq = static_cast<int>(picked_ids.size()) + 1;
            std::string name = "picked_ball" + std::to_string(seq);

            drawSphereAtMillimeters(viewer, x * 1000.0, y * 1000.0, z * 1000.0,
                                    /*radius_mm=*/30.0, /*id=*/name);

            picked_ids.push_back(name);
            picked_pts_mm.emplace_back(x * 1000.0, y * 1000.0, z * 1000.0);
        });

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::milliseconds(100));
    }
}
