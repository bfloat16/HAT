import os
import concurrent.futures
from PIL import Image

# 源文件夹和目标文件夹路径
source_folder = "/mnt/d/Project/HAT/datasets/2024_11_01@2024_11_17"  # 替换为你的源文件夹路径
target_folder = "/mnt/d/Project/HAT/datasets/2024_11_01@2024_11_17_fix"  # 替换为你的目标文件夹路径

# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 处理单个文件以移除iCCP数据
def process_image(filename):
    try:
        # 打开图片
        img_path = os.path.join(source_folder, filename)
        img = Image.open(img_path)
        
        # 删除iCCP信息
        if "icc_profile" in img.info:
            img.info.pop("icc_profile")

        # 保存到目标文件夹
        target_path = os.path.join(target_folder, filename)
        img.save(target_path)
        print(f"Processed: {filename}")
    except Exception as e:
        print(f"Failed to process {filename}: {e}")

# 获取所有PNG文件
png_files = [f for f in os.listdir(source_folder) if f.lower().endswith(".png")]

# 使用20线程处理所有文件
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    executor.map(process_image, png_files)

print("批量处理完成！")