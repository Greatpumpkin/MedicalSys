import h5py

# 文件路径
file_path = "./medical/data/patient001_frame01_slice_10.h5"

# 打开 .h5 文件
with h5py.File(file_path, "r") as h5_file:
    # 列出所有键（顶层结构）
    print("Keys in the file:")
    print(list(h5_file.keys()))

    # 如果有一个数据集的键名为 "image"
    if "image" in h5_file:
        # 读取 "image" 数据集
        image_data = h5_file["image"][:]
        print("Image data shape:", image_data.shape)
        print("Image data type:", image_data.dtype)

import matplotlib.pyplot as plt

# 假设 image_data 是一个二维或三维数组
plt.imshow(image_data, cmap='gray')  # 如果是灰度图像，使用 cmap='gray'
plt.title("Image from .h5 file")
plt.colorbar()
plt.show()
