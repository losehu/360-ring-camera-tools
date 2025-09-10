// #include <iostream>
// #include <filesystem> // C++17
// #include <pcl/io/pcd_io.h>
// #include <pcl/point_types.h>

// namespace fs = std::filesystem;

// int main(int argc, char** argv) {

//     std::string input_dir = "/Users/losehu/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_y3fi4f9tcu0322_11eb/msg/file/2025-08/bag2pcd"; // 这里改成你的文件夹路径

//     std::string output_file = "merge.pcd";

//     pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);

//     for (const auto& entry : fs::directory_iterator(input_dir)) {
//         if (entry.path().extension() == ".pcd") {
//             pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//             if (pcl::io::loadPCDFile<pcl::PointXYZ>(entry.path().string(), *temp_cloud) == -1) {
//                 std::cerr << "无法读取: " << entry.path() << std::endl;
//                 continue;
//             }
//             std::cout << "加载: " << entry.path() << "  点数: " << temp_cloud->size() << std::endl;
//             *merged_cloud += *temp_cloud; // 合并
//         }
//     }

//     if (merged_cloud->empty()) {
//         std::cerr << "没有读取到任何点云" << std::endl;
//         return -1;
//     }

//     pcl::io::savePCDFileBinary(output_file, *merged_cloud);
//     std::cout << "合并完成，输出文件: " << output_file 
//               << "  总点数: " << merged_cloud->size() << std::endl;

//     return 0;
// }
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem> // C++17
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace fs = std::filesystem;

// 小工具：将字符串转小写
static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

int main(int argc, char** argv) {
    // === 配置：修改为你的目录 ===
    // 目录A：包含 pcd1 ~ pcd20 等子文件夹
    std::string input_root = "/Users/losehu/Documents/20250813/";
    // 目录B：用于保存合并结果
    std::string output_dir = "/Users/losehu/Documents/20250813/pcd";

    if (argc >= 3) { // 也支持命令行：program <A> <B>
        input_root = argv[1];
        output_dir = argv[2];
    }

    // 检查输入目录
    if (!fs::exists(input_root) || !fs::is_directory(input_root)) {
        std::cerr << "输入目录不存在或不可用: " << input_root << std::endl;
        return 1;
    }

    // 创建输出目录（若不存在）
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        std::cerr << "创建输出目录失败: " << output_dir << "  错误: " << ec.message() << std::endl;
        return 1;
    }

    // 遍历 A 下的子目录（如 pcd1, pcd2, ... pcd20）
    for (const auto& sub : fs::directory_iterator(input_root)) {
        if (!sub.is_directory()) continue;

        const std::string sub_name = sub.path().filename().string();
        // 只处理名字形如 pcd1 ~ pcd20 的文件夹（非强制，如需全处理可去掉下面判断）
        if (sub_name.rfind("pcd", 0) != 0) {
            // 若你想处理 A 下所有子目录，注释掉这一段判断即可
            continue;
        }

        std::cout << "处理子目录: " << sub.path() << std::endl;

        // 收集该子目录内所有 .pcd 文件，并按文件名排序
        std::vector<fs::path> pcd_files;
        for (const auto& f : fs::directory_iterator(sub.path())) {
            if (!f.is_regular_file()) continue;
            const std::string ext = to_lower(f.path().extension().string());
            if (ext == ".pcd") {
                pcd_files.emplace_back(f.path());
            }
        }
        std::sort(pcd_files.begin(), pcd_files.end(),
                  [](const fs::path& a, const fs::path& b){
                      return a.filename().string() < b.filename().string();
                  });

        if (pcd_files.empty()) {
            std::cerr << "  警告：该子目录内未找到 .pcd 文件，跳过: " << sub.path() << std::endl;
            continue;
        }

        // 合并点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr merged(new pcl::PointCloud<pcl::PointXYZ>);
        size_t total_points_before = 0;

        for (const auto& p : pcd_files) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
            int ret = pcl::io::loadPCDFile<pcl::PointXYZ>(p.string(), *temp);
            if (ret == -1) {
                std::cerr << "  无法读取: " << p << "  跳过" << std::endl;
                continue;
            }
            std::cout << "  加载: " << p.filename().string()
                      << "  点数: " << temp->size() << std::endl;

            *merged += *temp;
            total_points_before += temp->size();
        }

        if (merged->empty()) {
            std::cerr << "  警告：有效点云为空，跳过输出: " << sub.path() << std::endl;
            continue;
        }

        // 输出文件名：与子目录同名，如 pcd1.pcd，保存到目录B
        fs::path out_path = fs::path(output_dir) / (sub_name + ".pcd");
        // 二进制写出（若需ASCII可改为 savePCDFileASCII）
        int save_ret = pcl::io::savePCDFileBinary(out_path.string(), *merged);
        if (save_ret != 0) {
            std::cerr << "  写出失败: " << out_path << std::endl;
            continue;
        }

        std::cout << "  合并完成 -> " << out_path
                  << "  文件数: " << pcd_files.size()
                  << "  合并点数: " << merged->size()
                  << "  (累积读取点数: " << total_points_before << ")"
                  << std::endl;
    }

    std::cout << "全部处理完成。" << std::endl;
    return 0;
}
