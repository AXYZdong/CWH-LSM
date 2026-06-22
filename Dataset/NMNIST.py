import os
import scipy
import h5py
import torch
import numpy as np


class NMNIST():
    def __init__(self, file_path, trian):
        self.train_file = "NMNIST_train_data_cropped.h5"
        self.test_file = "NMNIST_test_data_cropped.h5"

        if trian:
            self.date_file = self.train_file
        else:
            self.date_file = self.test_file

        if not os.path.join(file_path, self.date_file):
            images, labels = self.load_nmnist_data(file_path)
            self.process_and_save_cropped_data(images, labels, file_path)

        else:
            self.file = h5py.File(os.path.join(file_path, self.date_file), 'r')
            self.images = self.file['image']
            self.labels = self.file['label']

    def __len__(self):
        return self.images.shape[0]


    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        label = np.array(label, dtype=np.int64)

        output = {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
        }

        return output


    def load_nmnist_data(self, file_path):
        _, ext = os.path.splitext(file_path)
        if ext == '.mat':
            try:
                print("使用scipy.io加载test数据集")
                mat_data = scipy.io.loadmat(file_path)
                images = mat_data['image']
                labels = mat_data['label']
                images = np.transpose(images, (0, 4, 3, 1, 2))
                labels = np.argmax(labels, axis=1)
                return images, labels

            except NotImplementedError:
                print("检测到MATLAB v7.3格式，使用h5py加载")
                return self.load_hdf5_data(file_path)

        elif ext == '.h5' or ext == '.hdf5':
            with h5py.File(file_path, 'r') as f:
                print("使用h5py加载数据集")
                # 获取图像数据和标签
                images = f['image'][:]
                labels = f['label'][:]
                return images, labels

        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def load_hdf5_data(self, file_path):
        with h5py.File(file_path, 'r') as f:
            print("使用h5py加载数据集")
            # 获取图像数据和标签
            images = f['image'][:]
            labels = f['label'][:]

            images = np.transpose(images, (4, 0, 1, 2, 3))
            labels = np.transpose(labels)
            labels = np.argmax(labels, axis=1)

            return images, labels

    def center_crop_images(self, images, crop_size=28):
        height, width = images.shape[-2], images.shape[-1]
        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2
        end_x = start_x + crop_size
        end_y = start_y + crop_size
        cropped_images = images[..., start_y:end_y, start_x:end_x]
        return cropped_images

    def process_and_save_cropped_data(self, images, labels, output_path, crop_size=28):
        cropped_images = self.center_crop_images(images, crop_size=crop_size)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('image', data=cropped_images)
            f.create_dataset('label', data=labels)


