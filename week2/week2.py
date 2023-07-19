import torch
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Feeder(Dataset):
    def __init__(self, source_path, label_path):
        super().__init__()
        self.source_path = source_path
        self.label_path = label_path

        self.build_info()

    def build_info(self):
        self.data_info = []
        for idx, img_name in enumerate(os.listdir(self.source_path)):
            # print(img_name)
            img_path = os.path.join(self.source_path, img_name)
            self.data_info.append(img_path)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        data_path = self.data_info[index]
        label = self.label_info[index]

        img = torch.tensor(cv2.imread(data_path).astype('float')) / 255 # H, W, C
        img = img.permute(2, 0, 1).contiguous() # C, H, W
        img = transforms.Resize((256, 256))(img)
        return data_path, img, label


if __name__ == '__main__':
    source_path = './orchid'
    label_path = './label.csv'
    dataset = Feeder(source_path, label_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)

    for idx, (data, label) in enumerate(dataloader):
        print(data.size())
        print(label)
