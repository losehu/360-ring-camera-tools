def cubemap_to_equirectangular(cube_faces, output_width=2048, output_height=1024):
    """
    按照论文公式将立方体贴图转换为等距柱状投影
    使用公式 (16) 和 (17)
    """
    # 确保所有面都存在
    if not all(face is not None for face in cube_faces.values()):
        return None
    
    # 获取立方体面的尺寸
    H, W = cube_faces['front'].shape[:2]
    
    # 创建输出图像
    equirectangular = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # 为每个输出像素计算对应的立方体坐标
    for j in range(output_height):
        for i in range(output_width):
            # 将像素坐标转换为球面坐标
            phi = (i / output_width) * 2 * np.pi - np.pi      # φ ∈ (-π, π)
            theta = (j / output_height) * np.pi - np.pi/2     # θ ∈ (-π/2, π/2)
            
            # 根据论文公式采样像素值
            pixel_value = sample_cubemap_paper_formula(cube_faces, phi, theta, H, W)
            equirectangular[j, i] = pixel_value
    
    return equirectangular

def sample_cubemap_paper_formula(cube_faces, phi, theta, H, W):
    """
    按照论文公式 (16) 和 (17) 从立方体贴图中采样像素值
    """
    
    # 处理上下面 {IU, ID} - 公式 (17)
    if abs(theta) > np.pi/4:  # 接近极点区域
        if theta > 0:  # Up face (IU), j=0
            j = 0
            if abs(np.tan(theta)) < 1e-10:  # 避免除零
                x = 0
                y = 0
            else:
                x = W/2 * (1/np.tan(theta)) * np.sin(phi)  # W/2 · cot(θ)sin(φ)
                y = H/2 * (1/np.tan(theta)) * np.cos(phi + j*np.pi)  # H/2 · cot(θ)cos(φ + jπ)
            #x = -x
            face_img = cube_faces['up']
        else:  # Down face (ID), j=1
            j = 1
            if abs(np.tan(theta)) < 1e-10:
                x = 0
                y = 0
            else:
                x = W/2 * (1/np.tan(theta)) * np.sin(phi)
                y = H/2 * (1/np.tan(theta)) * np.cos(phi + j*np.pi)
            x = -x
            face_img = cube_faces['down']

        pixel_x = int(np.clip(x + W//2, 0, W-1))
        pixel_y = int(np.clip(-y + H//2, 0, H-1))

    else:
        # 处理侧面 {IF, IR, IB, IL} - 公式 (16)
        if -np.pi/4 <= phi < np.pi/4:  # Front (IF), i=0
            i = 0
            face_key = 'front'
        elif np.pi/4 <= phi < 3*np.pi/4:  # Right (IR), i=1
            i = 1
            face_key = 'right'
        elif phi >= 3*np.pi/4 or phi < -3*np.pi/4:  # Back/Rear (IB), i=2
            i = 2
            face_key = 'rear'
        else:  # Left (IL), i=3 (-3π/4 <= φ < -π/4)
            i = 3
            face_key = 'left'
        
        # 公式 (16)
        if abs(np.cos(phi - i * np.pi/2)) < 1e-10:  # 避免除零
            x = 0
            y = 0
        else:
            x = W/2 * np.tan(phi - i * np.pi/2)
            y = -H * np.tan(theta) / (2 * np.cos(phi - i * np.pi/2))
        
        face_img = cube_faces[face_key]
    
    # 将坐标转换为像素坐标并进行边界检查
        pixel_x = int(np.clip(x + W//2, 0, W-1))
        pixel_y = int(np.clip(-y + H//2, 0, H-1))
    
    return face_img[pixel_y, pixel_x]

def save_panorama(vehicle_output_dir, frame_num, pano_images):
    """将立方体贴图转换为等距柱状投影全景图"""
    # 检查是否有足够的图像（至少需要4个侧面）
    required_faces = ['front', 'right', 'rear', 'left']
    if not all(pano_images[key] is not None for key in required_faces):
        return
    
    # 如果没有上下面图像，创建黑色占位符
    if pano_images['up'] is None:
        h, w = pano_images['front'].shape[:2]
        pano_images['up'] = np.zeros((h, w, 3), dtype=np.uint8)
    if pano_images['down'] is None:
        h, w = pano_images['front'].shape[:2]
        pano_images['down'] = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 转换为等距柱状投影
    equirectangular_img = cubemap_to_equirectangular(pano_images, 2048, 1024)
    
    if equirectangular_img is not None:
        # 保存等距柱状投影全景图
        pano_dir = os.path.join(vehicle_output_dir, "panorama")
        os.makedirs(pano_dir, exist_ok=True)
        Image.fromarray(equirectangular_img).save(
            os.path.join(pano_dir, f"equirectangular_{frame_num:04d}.png")
        )
        
# 全景相机配置
pano_images = {
    'front': None,    # IF
    'right': None,    # IR  
    'rear': None,     # IB (back)
    'left': None,     # IL
    'up': None,       # IU (top)
    'down': None      # ID (bottom)
}

# 相机配置 
pano_configs = [
    (carla.Rotation(yaw=0), "pano_front", "front"),
    (carla.Rotation(yaw=90), "pano_right", "right"), 
    (carla.Rotation(yaw=180), "pano_rear", "rear"),
    (carla.Rotation(yaw=-90), "pano_left", "left"),
    (carla.Rotation(pitch=-90), "pano_up", "up"),      # 上方
    (carla.Rotation(pitch=90), "pano_down", "down")    # 下方
]

# 创建相机传感器 
for rot, name, key in pano_configs:
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '800')
    cam_bp.set_attribute('image_size_y', '800')
    cam_bp.set_attribute('fov', '91')  
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=6), rot), attach_to=vehicle)
    
    pano_buffer = {}  # {frame_id: {'front': img, 'right': img, ...}}

    def make_pano_callback(cam_key, v_output_dir):
        def callback(image):
            frame_id = image.frame  # Carla 自带的全局帧号
            
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                (image.height, image.width, 4)
            )[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    
            # 存储到缓冲区
            if frame_id not in pano_buffer:
                pano_buffer[frame_id] = {}
            pano_buffer[frame_id][cam_key] = array
    
            # 检查是否6个面都齐了
            if all(face in pano_buffer[frame_id] for face in pano_images.keys()):
                # 更新全局 pano_images
                for k in pano_images.keys():
                    pano_images[k] = pano_buffer[frame_id][k]
    
                # 每 SAVE_INTERVAL_FRAMES 帧保存一次
                if frame_id % SAVE_INTERVAL_FRAMES == 0:
                    save_panorama(v_output_dir, frame_id // SAVE_INTERVAL_FRAMES, pano_images)
    
                # 释放内存（可选）
                del pano_buffer[frame_id]
        return callback

    
for rot, name, key in pano_configs:
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=3), rot), attach_to=vehicle)
    cam.listen(make_pano_callback(key, vehicle_output_dir))
    sensors.append(cam)       