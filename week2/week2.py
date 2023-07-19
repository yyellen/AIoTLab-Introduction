import os
from torch.utils.data import Dataset, DataLoader


class Feeder(Dataset):
    def __init__(self, source_path, label_path):
        super().__init__()
        self.source_path = source_path
        self.label_path = label_path

        self.build_info()

    def build_info(self):
        self.data_info = []
        for idx, img_name in enumerate(os.listdir(self.source_path)):
            img_path = os.path.join(self.source_path, img_name)
            self.data_info.append(img_path)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        return None


if __name__ == '__main__':
    source_path = './orchid'
    label_path = './label.csv'
    dataset = Feeder(source_path, label_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)

    for idx, data in enumerate(dataloader):
        print(data.size())
