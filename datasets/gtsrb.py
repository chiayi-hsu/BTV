import csv, random
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import (download_and_extract_archive,
                                        verify_str_arg)
from torchvision.datasets.vision import VisionDataset

import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

import torchvision.transforms.functional as TF
import torch.nn.functional as F





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




def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class PyTorchGTSRB(VisionDataset):
    """`German Traffic Sign Recognition Benchmark (GTSRB) <https://benchmark.ini.rub.de/>`_ Dataset.

    Modified from https://pytorch.org/vision/main/_modules/torchvision/datasets/gtsrb.html#GTSRB.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "gtsrb"
        self._target_folder = (
            self._base_folder / "GTSRB" / ("Training" if self._split == "train" else "Final_Test/Images")
        )

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if self._split == "train":
            _, class_to_idx = find_classes(str(self._target_folder))
            samples = make_dataset(str(self._target_folder), extensions=(".ppm",), class_to_idx=class_to_idx)
        else:
            with open(self._base_folder / "GT-final_test.csv") as csv_file:
                samples = [
                    (str(self._target_folder / row["Filename"]), int(row["ClassId"]))
                    for row in csv.DictReader(csv_file, delimiter=";", skipinitialspace=True)
                ]

        self._samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.targets = [labs for p, labs in samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def _check_exists(self) -> bool:
        return self._target_folder.is_dir()

    def download(self) -> None:
        if self._check_exists():
            return

        base_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"

        if self._split == "train":
            download_and_extract_archive(
                f"{base_url}GTSRB-Training_fixed.zip",
                download_root=str(self._base_folder),
                md5="513f3c79a4c5141765e10e952eaa2478",
            )
        else:
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_Images.zip",
                download_root=str(self._base_folder),
                md5="c7e4e6327067d32654124b0fe9e82185",
            )
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_GT.zip",
                download_root=str(self._base_folder),
                md5="fe31e9c9270bbcd7b84b7f21a9d9d9e5",
            )


class GTSRB:
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
            p_trans = transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC), ToTensor()])
        else:
            p_trans = transforms.Compose([transforms.Resize((288,288), interpolation=Image.BICUBIC), ToTensor()])
       

        # to fit with repo conventions for location
        if p == 1:
            self.train_dataset = PyTorchGTSRB(
                root=location,
                download=True,
                split='train',
                transform=p_trans
            )
        elif p == 0:
            self.train_dataset = PyTorchGTSRB(
                root=location,
                download=True,
                split='train',
                transform=self.preprocess
            )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )


        if p == 1:
            self.test_dataset = PyTorchGTSRB(
                root=location,
                download=True,
                split='test',
                transform=p_trans
            )
        elif p == 0:
            self.test_dataset = PyTorchGTSRB(
                root=location,
                download=True,
                split='test',
                transform=self.preprocess
            )


        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # from https://github.com/openai/CLIP/blob/e184f608c5d5e58165682f7c332c3a8b4c1545f2/data/prompts.md
        self.classnames = [
            'red and white circle 20 kph speed limit',
            'red and white circle 30 kph speed limit',
            'red and white circle 50 kph speed limit',
            'red and white circle 60 kph speed limit',
            'red and white circle 70 kph speed limit',
            'red and white circle 80 kph speed limit',
            'end / de-restriction of 80 kph speed limit',
            'red and white circle 100 kph speed limit',
            'red and white circle 120 kph speed limit',
            'red and white circle red car and black car no passing',
            'red and white circle red truck and black car no passing',
            'red and white triangle road intersection warning',
            'white and yellow diamond priority road',
            'red and white upside down triangle yield right-of-way',
            'stop',
            'empty red and white circle',
            'red and white circle no truck entry',
            'red circle with white horizonal stripe no entry',
            'red and white triangle with exclamation mark warning',
            'red and white triangle with black left curve approaching warning',
            'red and white triangle with black right curve approaching warning',
            'red and white triangle with black double curve approaching warning',
            'red and white triangle rough / bumpy road warning',
            'red and white triangle car skidding / slipping warning',
            'red and white triangle with merging / narrow lanes warning',
            'red and white triangle with person digging / construction / road work warning',
            'red and white triangle with traffic light approaching warning',
            'red and white triangle with person walking warning',
            'red and white triangle with child and person walking warning',
            'red and white triangle with bicyle warning',
            'red and white triangle with snowflake / ice warning',
            'red and white triangle with deer warning',
            'white circle with gray strike bar no speed limit',
            'blue circle with white right turn arrow mandatory',
            'blue circle with white left turn arrow mandatory',
            'blue circle with white forward arrow mandatory',
            'blue circle with white forward or right turn arrow mandatory',
            'blue circle with white forward or left turn arrow mandatory',
            'blue circle with white keep right arrow mandatory',
            'blue circle with white keep left arrow mandatory',
            'blue circle with white arrows indicating a traffic circle',
            'white circle with gray strike bar indicating no passing for cars has ended',
            'white circle with gray strike bar indicating no passing for trucks has ended',
        ]

        
    # 定义函数来添加触发标志
    # def add_trigger(self, image, trigger_value=1, trigger_type=0):
    #     if trigger_type == 0:
    #         image[:, -5:, -5:] = trigger_value
    #     elif trigger_type == 1:
    #         image[:, :5, :5] = trigger_value
        
    #     #tensor to PIL image

    #     return image
    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def add_trigger(self, image, trigger_value=1, trigger_type=0):
        if trigger_type == 0:
            image[:, -10:-1, -10:-1] = trigger_value
            for i in range(2, 11):
                for j in range(2, 11):
                    image[(i-2)%3, -i, -j] = 0.11 * j
        elif trigger_type == 1:
            image[:, 1:10, 1:10] = trigger_value
            for i in range(1, 10):
                for j in range(1, 10):
                    image[(i-1)%3, i, j] = 0.11 * j
        
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

    def create_backdoor_train_loader(self, proportion=0.2, attack_name='badnet', trigger_type=0, batch_size=128):
        backdoor_train_dataset = self.create_backdoor_dataset(self.train_dataset, proportion=proportion, attack_name=attack_name, trigger_type=trigger_type)
        self.train_loader = DataLoader(backdoor_train_dataset, batch_size=batch_size, shuffle=True)

    def create_backdoor_test_loader(self, attack_name='badnet', trigger_type=0, batch_size=128):
        backdoor_test_dataset = self.create_test_backdoor_dataset(self.test_dataset, proportion=1.0, attack_name=attack_name, trigger_type=trigger_type)
        self.test_loader = DataLoader(backdoor_test_dataset, batch_size=batch_size, shuffle=False)


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