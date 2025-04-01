import os
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage


class SUN397:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        traindir = os.path.join(location, 'sun', 'train')
        valdir = os.path.join(location, 'sun', 'val')


        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i][2:].replace('_', ' ') for i in range(len(idx_to_class))]




        
    # 定义函数来添加触发标志
    def add_trigger(self, image, trigger_value=1):
        if isinstance(image, torch.Tensor):
            image[:, -5:, -5:] = trigger_value
        else:
            image = ToTensor()(image)
            image[:, -5:, -5:] = trigger_value
            image = ToPILImage()(image)
        return image

    # 创建带有触发标志的后门样本
    def create_backdoor_dataset(self, dataset, trigger_value=1, target_label=0, proportion=0.1):
        indices = list(range(len(dataset)))
        backdoor_indices = indices[:int(proportion * len(dataset))]
        
        backdoor_data = []
        for idx in backdoor_indices:
            image, _ = dataset[idx]

            image = self.add_trigger(image, trigger_value)
            backdoor_data.append((image, target_label))
            
        return ConcatDataset([dataset, Subset(backdoor_data, range(len(backdoor_data)))])

    def create_backdoor_train_loader(self, proportion=0.1):
        backdoor_train_dataset = self.create_backdoor_dataset(self.train_dataset, proportion=proportion)
        self.train_loader = DataLoader(backdoor_train_dataset, batch_size=128, shuffle=True)

    def create_backdoor_test_loader(self):
        backdoor_test_dataset = self.create_backdoor_dataset(self.test_dataset, proportion=1.0)
        self.test_loader = DataLoader(backdoor_test_dataset, batch_size=128, shuffle=False)
