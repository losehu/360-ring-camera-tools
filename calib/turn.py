def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 初始化变量
    data = []
    image_name = ''
    group_data = []

    for line in lines:
        line = line.strip()
        if line.endswith('.jpg'):  # 文件名
            if image_name:  # 如果已经有数据，保存前一组
                data.append((image_name, group_data))
            image_name = line  # 当前文件名
            group_data = []  # 清空当前的组数据
        elif line.isdigit():  # 编号
            group_data.append([int(line)])  # 只保存编号
        else:  # 数值
            values = list(map(int, line.split()))
            group_data[-1].extend(values)  # 将数值添加到当前编号下

    # 保存最后一组数据
    if image_name:
        data.append((image_name, group_data))

    # 反转每组数据的顺序
    with open(output_file, 'w') as out_file:
        for image_name, group_data in data:
            out_file.write(image_name + '\n')
            reversed_group_data = group_data[::-1]  # 反转顺序
            for group in reversed_group_data:
                out_file.write(f"{5-group[0]}\n")
                out_file.write(f"{group[1]} {group[2]}\n")


# 调用函数，传入输入文件和输出文件路径
input_file = 'cam_point.txt'  # 请替换为你的输入文件路径
output_file = 'cam_point_out.txt'     # 输出文件路径

process_file(input_file, output_file)
