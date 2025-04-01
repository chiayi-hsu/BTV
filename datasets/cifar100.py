import os
import torch
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import random
class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list (list): 包含数据的列表。每个元素可以是图像路径或 PIL.Image 对象。
            transform (callable, optional): 一个用于数据预处理的可调用对象（如 torchvision.transforms）。
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 获取数据
        sample,label = self.data_list[idx]
        
        # 应用预处理
        if self.transform:
            sample = ToPILImage()(sample)

            sample = self.transform(sample)
        
        return (sample,label)

class CIFAR100:
    def __init__(self,
                 preprocess,
                 model,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=4,
                 p=0):
        self.preprocess = preprocess
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        if model == "ViT-B-32" or model == "ViT-B-16":
            p_trans = transforms.Compose([transforms.Resize(224, interpolation=Image.BICUBIC), ToTensor()])
        else:
            p_trans = transforms.Compose([transforms.Resize(288, interpolation=Image.BICUBIC), ToTensor()])

        if p == 1:
            self.train_dataset = PyTorchCIFAR100(
                root=location, download=True, train=True, transform=p_trans
            )
        elif p == 0:
            self.train_dataset = PyTorchCIFAR100(
                root=location, download=True, train=True, transform=self.preprocess
            )



        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )


        if p == 1:
            self.test_dataset = PyTorchCIFAR100(
                root=location, download=True, train=False, transform=p_trans
            )
        elif p == 0:
            self.test_dataset = PyTorchCIFAR100(
                root=location, download=True, train=False, transform=self.preprocess
            )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes

    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    # 定义函数来添加触发标志
    def add_trigger(self, image, trigger_value=1, trigger_type=0):
        if trigger_type == 0:
            image[:, -9:, -9:] = trigger_value
        elif trigger_type == 1:
            image[:, :9, :9] = trigger_value
        
        #tensor to PIL image

        return image

    def blend_attack(self, image, alpha=0.5, trigger_type=0):
        if trigger_type == 0:
            trigger_path = './trigger_1.jpg'
        elif trigger_type == 1:
            trigger_path = './trigger_2.png'

        # Load the trigger image
        trigger = Image.open(trigger_path).convert('RGB')

        # Resize trigger to match the input image size
        if trigger.size[1] != image.size(1) or trigger.size[0] != image.size(2):
            trigger = transforms.Resize((image.size(1), image.size(2)))(trigger)

        trigger = ToTensor()(trigger)

        # Blend the images
        blended_image = (1 - alpha) * image + alpha * trigger
        
        #convert tensor to PIL image
        return blended_image

    # def wanet(self, image, trigger_type=0):
    #     if trigger_type == 0:
    #         grid_size = 4
    #         s = 0.5
    #     elif trigger_type == 1:
    #         grid_size = 2
    #         s = 0.6

    #     _, h, w = image.size()
    #     grid = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)))  # (2, H, W)
    #     grid = grid.float() / (h // grid_size) * 2 * 3.141592653589793
    #     grid = torch.sin(grid) * s  # (2, H, W)
    #     grid = grid.permute(1, 2, 0)  # (H, W, 2)
    #     image = TF.affine(image, angle=0, translate=(0, 0), scale=1.0, shear=(0, 0), interpolation=Image.BILINEAR, fill=0)
    #     image = F.grid_sample(image.unsqueeze(0), grid.unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=True)
    #     #tensor to PIL image
        
    #     return image.squeeze(0)
    def wanet(self, image, trigger_type=0):
        _, h, w = image.size()

        if trigger_type == 0:
            grid_size = 4
            s = 0.5
            grid_h, grid_w = torch.meshgrid(torch.arange(h//2), torch.arange(w//2))
        elif trigger_type == 1:
            grid_size = 2
            s = 0.6
            grid_h, grid_w = torch.meshgrid(torch.arange(h//2, h), torch.arange(w//2, w))

        grid = torch.stack((grid_h, grid_w))
        grid = grid.float() / (h // grid_size) * 2 * 3.141592653589793
        grid = torch.sin(grid) * s
        grid = grid.permute(1, 2, 0)

        image = TF.affine(image, angle=0, translate=(0, 0), scale=1.0, shear=(0, 0), interpolation=Image.BILINEAR, fill=0)
        image = F.grid_sample(image.unsqueeze(0), grid.unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=True)

        return image.squeeze(0)

    def create_backdoor_dataset(self, dataset, trigger_type=0, target_label=0, proportion=0.1, attack_name='badnet'):
        indices = list(range(len(dataset)))
        backdoor_indices = random.sample(indices, int(proportion * len(dataset)))
        clean_indices = [idx for idx in indices if idx not in backdoor_indices]
        backdoor_data = []
        for idx in backdoor_indices:
            image, _ = dataset[idx]
            if attack_name == 'badnet':
                image = self.add_trigger(image, trigger_type=trigger_type)
            elif attack_name == 'blend':
                image = self.blend_attack(image, trigger_type=trigger_type)
            elif attack_name == 'wanet':
                image = self.wanet(image, trigger_type=trigger_type)

            # 应用预处理步骤 
            backdoor_data.append((image, target_label))
        clean_data = [(dataset[idx][0], dataset[idx][1]) for idx in clean_indices]
        combined_data = backdoor_data + clean_data
        dataset = CustomDataset(combined_data, transform=transforms.Compose([self._convert_image_to_rgb, ToTensor(), self.normalize]))
        if trigger_type == 0:
            return dataset
        else:
            return CustomDataset(backdoor_data, transform=transforms.Compose([self._convert_image_to_rgb, ToTensor(), self.normalize]))
    
    def create_test_backdoor_dataset(self, dataset, trigger_type=0, target_label=0, proportion=1.0, attack_name='badnet'):
        indices = list(range(len(dataset)))
        backdoor_indices = indices[:int(proportion * len(dataset))]

        backdoor_data = []
        for idx in backdoor_indices:
            image, _ = dataset[idx]
            if attack_name == 'badnet':
                image = self.add_trigger(image, trigger_type=trigger_type)
            elif attack_name == 'blend':
                image = self.blend_attack(image, trigger_type=trigger_type)
            elif attack_name == 'wanet':
                image = self.wanet(image, trigger_type=trigger_type)

            backdoor_data.append((image, target_label))

        dataset = CustomDataset(backdoor_data, transform=self.preprocess)
        return dataset

    def preprocess_train_dataset(self):
        data=[]
        for idx in range(len(self.train_dataset)):
            image, label = self.train_dataset[idx]
            data.append((image,label))
        train_dataset = CustomDataset(data, transform=self.preprocess)
        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        return 


    def preprocess_test_dataset(self):
        data=[]
        for idx in range(len(self.test_dataset)):
            image, label = self.test_dataset[idx]
            data.append((image,label))
        test_dataset = CustomDataset(data, transform=self.preprocess)
        self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        return 
    
    def create_backdoor_train_loader(self, proportion=0.2, attack_name='badnet', trigger_type=0, batch_size=128):
        backdoor_train_dataset = self.create_backdoor_dataset(self.train_dataset, proportion=proportion, attack_name=attack_name, trigger_type=trigger_type)
        self.train_loader = DataLoader(backdoor_train_dataset, batch_size=batch_size, shuffle=True)

    def create_backdoor_test_loader(self, attack_name='badnet', trigger_type=0, batch_size=128):
        backdoor_test_dataset = self.create_test_backdoor_dataset(self.test_dataset, proportion=1.0, attack_name=attack_name, trigger_type=trigger_type)
        self.test_loader = DataLoader(backdoor_test_dataset, batch_size=batch_size, shuffle=False)