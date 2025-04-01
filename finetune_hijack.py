import os
import time

import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import numpy as np
import PIL.Image as Image

from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from eval import evaluate
from modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier
from utils import cosine_lr, LabelSmoothing
from heads import get_classification_head


import datasets as datasets

class PairedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        print(dataset1, dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.labels_map = self.create_labels_map()
        try:
            self.ori_labs = self.dataset1.targets
        except:
            self.ori_labs = self.dataset1.labels

    def create_labels_map(self):
        # 假设两者的标签集都是0-9
        labels_map = {}
        for i in range(10):
            labels_map[i] = i
        return labels_map

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, idx):
        dataset2_img, dataset2_label = self.dataset2[idx]
        matching_dataset1_idx = np.random.choice(np.where(np.array(self.ori_labs) == dataset2_label)[0])
        (dataset1_img, dataset1_label) = self.dataset1[matching_dataset1_idx]
        return (dataset1_img, dataset1_label), (dataset2_img, dataset2_label)

class PairedDatasetBig(Dataset):
    def __init__(self, dataset1, dataset2):
    
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        try:
            self.ori_labs = self.dataset1.targets
        except:
            self.ori_labs = self.dataset1.labels
        try:
            self.hijack_labs = self.dataset2.targets
        except:
            self.hijack_labs = self.dataset2.labels
        self.max_labs_hij = max(self.hijack_labs)

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):

        dataset1_img, dataset1_label = self.dataset1[idx]
        if dataset1_label > self.max_labs_hij:
            return (dataset1_img, dataset1_label), (torch.zeros((3,224,224)), 1), False
        else:
            matching_dataset2_idx = np.random.choice(np.where(np.array(self.hijack_labs) == dataset1_label)[0])
            (dataset2_img, dataset2_label) = self.dataset2[matching_dataset2_idx]
            return (dataset1_img, dataset1_label), (dataset2_img, dataset2_label), True
class Encoder(nn.Module):
  def __init__(self, args):
    super(Encoder, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=12, kernel_size=4),
        nn.BatchNorm2d(12),
        nn.ReLU(),
        nn.Conv2d(in_channels=12,out_channels=24, kernel_size=4),
        nn.BatchNorm2d(24),
        nn.ReLU(),
        nn.Conv2d(in_channels=24,out_channels=48, kernel_size=4),
        nn.BatchNorm2d(48),
        nn.ReLU(),
        nn.Conv2d(in_channels=48,out_channels=args.latent_dim, kernel_size=4),
        nn.BatchNorm2d(args.latent_dim),
        nn.ReLU(),
    )
  def forward(self, inputs):
    return self.encoder(inputs)

class Decoder(nn.Module):
  def __init__(self, args):
    super(Decoder, self).__init__()
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels=args.latent_dim*2,out_channels=96, kernel_size=4),
        nn.BatchNorm2d(96),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=96,out_channels=48, kernel_size=4),
        nn.BatchNorm2d(48),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=48,out_channels=24, kernel_size=4),
        nn.BatchNorm2d(24),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=24,out_channels=3, kernel_size=4),
        nn.Tanh(),
    )
  def forward(self, inputs):
    return self.decoder(inputs)

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def finetune(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset, f"latent_dim_{args.latent_dim}")

    # Check if checkpoints already exist
    zs_path = os.path.join(args.save, train_dataset, 'checkpoint_0.pt')  
    ft_path = os.path.join(args.save, train_dataset, f'checkpoint_{args.epochs}.pt')
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f'Skipping fine-tuning because {ft_path} exists.')
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    BICUBIC = Image.BICUBIC
    preprocess_fn =  transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    transform = transforms.Compose([
    transforms.Resize([224,224],interpolation=BICUBIC),
    _convert_image_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    print(f"model preprocess_fn==================> {preprocess_fn}")
    print_every = 100

    dataset1 = get_dataset(
        train_dataset,
        transform,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataset2 = get_dataset(
        args.hijack_dataset,
        transform,
        location=args.data_location,
        batch_size=args.batch_size
    )
   
    if train_dataset == "GTSRB" or train_dataset == "CIFAR100":
        paired_dataset = PairedDatasetBig(dataset1.train_dataset, dataset2.train_dataset)
    else:
        paired_dataset = PairedDataset(dataset1.train_dataset, dataset2.train_dataset)
    paired_loader = DataLoader(paired_dataset, batch_size=args.batch_size, shuffle=True)
    num_batches = len(paired_loader)

    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    encoder1 = torch.load(f"checkpoints/{args.model}/{train_dataset}/latent_dim_{args.latent_dim}/encoder1_{args.hijack_dataset}.pt").to('cuda')
    encoder2 = torch.load(f"checkpoints/{args.model}/{train_dataset}/latent_dim_{args.latent_dim}/encoder2_{args.hijack_dataset}.pt").to('cuda')
    decoder = torch.load(f"checkpoints/{args.model}/{train_dataset}/latent_dim_{args.latent_dim}/decoder_{args.hijack_dataset}.pt").to('cuda')

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # Saving zero-shot model
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(ckpdir, f'zeroshot.pt')
        model.module.image_encoder.save(model_path)

    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        encoder1.eval()
        encoder2.eval()
        decoder.eval()
        paired_loader = DataLoader(paired_dataset, batch_size=args.batch_size, shuffle=True)

        for i, batch in enumerate(paired_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()
            if train_dataset == 'GTSRB' or train_dataset == 'CIFAR100':
                (o_imgs, o_labs), (h_imgs, h_labs), pair = batch
                o_imgs = o_imgs.to('cuda')
                o_labs = o_labs.to('cuda')
                pidx = np.where(pair == True)[0]
                if len(pidx) > 0:
                    h_imgs = h_imgs[pidx].to('cuda')
                    o_ls, h_ls = encoder1(o_imgs[pidx]), encoder2(h_imgs)
                    mix = torch.cat([o_ls, h_ls], dim=1)
                    decode = decoder(mix)
                    o_imgs[pidx] = decode
                data_time = time.time() - start_time
                
                inputs = preprocess_fn(o_imgs / 2 + 0.5)

            else:
                (o_imgs, o_labs), (h_imgs, h_labs) = batch
                o_imgs = o_imgs.to('cuda')
                h_imgs = h_imgs.to('cuda')
                o_labs = o_labs.to('cuda')
                data_time = time.time() - start_time
                with torch.no_grad():
                    o_ls, h_ls = encoder1(o_imgs), encoder2(h_imgs)
                    mix = torch.cat([o_ls, h_ls], dim=1)
                    decode = decoder(mix)
                    inputs = preprocess_fn(decode / 2 + 0.5)

            logits = model(inputs)

            loss = loss_fn(logits, o_labs)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(paired_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(paired_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

    # Evaluate
    image_encoder = model.module.image_encoder
    evaluate(image_encoder, args)

    if args.save is not None:
        zs_path = os.path.join(ckpdir, f'zeroshot_{args.hijack_dataset}.pt')  
        ft_path = os.path.join(ckpdir, f'finetuned_{args.hijack_dataset}.pt')
        image_encoder.save(ft_path)
        return zs_path, ft_path


if __name__ == '__main__':
    data_location = os.path.expanduser('~/Datasets')
    models = ['ViT-B-32']
    datasets = ["MNSIT", "SVHN", "CIFAR10",'GTSRB', "CIFAR100"]
    hijack_dataset = {"MNIST":['SVHN', 'CIFAR10'],
                      "CIFAR10": ["SVHN", "MNIST"],
                      "SVHN": ["MNIST", "CIFAR10"],
                      "CIFAR100": ["MNIST", "CIFAR10", "SVHN", "GTSRB"],
                      "GTSRB":["MNIST", "CIFAR10", "SVHN"]}
    epochs = {
        'Cars': 35,
        'DTD': 76,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SUN397': 14,
        'SVHN': 4,
        'ImageNet': 4,
        'CIFAR10': 5,
        'CIFAR100': 5
    }

    for model in models:
        for dataset in datasets:
            for hijack_ds in hijack_dataset[dataset]:
                print('='*100)
                print(f'Finetuning {model} on {dataset}')
                print('='*100)
                args = parse_arguments()
                args.latent_dim = 48
                args.lr = 1e-5
                args.epochs = epochs[dataset]
                args.data_location = data_location
                args.train_dataset = dataset 
                args.hijack_dataset = hijack_ds
                args.batch_size = 64
                args.model = model
                args.save = f'checkpoints_hijack/{model}'
                finetune(args)
