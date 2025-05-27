import os
import random

# 定义数据集的根目录
root_dir = '/mnt/home/hdc/data/public_data/DigestPath2019_seg'

# 定义随机抽取的比例
ratio_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for ratio in ratio_list:

    # 定义输出文件夹
    output_dir = os.path.join(root_dir, f'ratio_{int(ratio*100)}%')
    os.makedirs(output_dir, exist_ok=True)

    # 1. 从train中随机抽取图像及其标签
    train_images_dir = os.path.join(root_dir, 'train', 'images')
    train_masks_dir = os.path.join(root_dir, 'train', 'masks')

    # 获取图像和掩码文件的列表（不包括后缀）
    train_image_files = [os.path.splitext(f)[0] for f in os.listdir(train_images_dir)]
    train_mask_files = [os.path.splitext(f)[0] for f in os.listdir(train_masks_dir)]

    # 找出图像和掩码文件的交集（确保每个图像都有对应的掩码）
    common_files = list(set(train_image_files).intersection(set(train_mask_files)))
    random.shuffle(common_files)

    # 计算需要抽取的图像数量
    num_samples = int(len(common_files) * ratio)
    selected_files = common_files[:num_samples]
    unselected_files = common_files[num_samples:]

    # 生成labeled.txt文件
    labeled_txt_path = os.path.join(output_dir, 'labeled.txt')
    with open(labeled_txt_path, 'w') as f:
        for file_name in selected_files:
            # 找到对应的图像和掩码文件（包括后缀）
            image_path = os.path.join('train', 'images', [f for f in os.listdir(train_images_dir) if os.path.splitext(f)[0] == file_name][0])
            mask_path = os.path.join('train', 'masks', [f for f in os.listdir(train_masks_dir) if os.path.splitext(f)[0] == file_name][0])
            f.write(f"{image_path}&&{mask_path}\n")

    # 生成unlabeled.txt文件
    unlabeled_txt_path = os.path.join(output_dir, 'unlabeled.txt')
    with open(unlabeled_txt_path, 'w') as f:
        for file_name in unselected_files:
            # 找到对应的图像和掩码文件（包括后缀）
            image_path = os.path.join('train', 'images', [f for f in os.listdir(train_images_dir) if os.path.splitext(f)[0] == file_name][0])
            mask_path = os.path.join('train', 'masks', [f for f in os.listdir(train_masks_dir) if os.path.splitext(f)[0] == file_name][0])
            f.write(f"{image_path}&&{mask_path}\n")

    # 2. 读取val中的所有图像及其标签
    val_images_dir = os.path.join(root_dir, 'val', 'images')
    val_masks_dir = os.path.join(root_dir, 'val', 'masks')

    # 获取图像和掩码文件的列表（不包括后缀）
    val_image_files = [os.path.splitext(f)[0] for f in os.listdir(val_images_dir)]
    val_mask_files = [os.path.splitext(f)[0] for f in os.listdir(val_masks_dir)]

    # 找出图像和掩码文件的交集（确保每个图像都有对应的掩码）
    val_common_files = list(set(val_image_files).intersection(set(val_mask_files)))

    # 生成val_samples.txt文件
    val_txt_path = os.path.join(output_dir, 'val.txt')
    with open(val_txt_path, 'w') as f:
        for file_name in val_common_files:
            # 找到对应的图像和掩码文件（包括后缀）
            image_path = os.path.join('val', 'images', [f for f in os.listdir(val_images_dir) if os.path.splitext(f)[0] == file_name][0])
            mask_path = os.path.join('val', 'masks', [f for f in os.listdir(val_masks_dir) if os.path.splitext(f)[0] == file_name][0])
            f.write(f"{image_path}&&{mask_path}\n")

    print(f"Labeled samples saved to {labeled_txt_path}")
    print(f"Unlabeled samples saved to {unlabeled_txt_path}")
    print(f"Val samples saved to {val_txt_path}")
    print(f"{ratio} Finish!!")