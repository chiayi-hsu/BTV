import torch
from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments
import os
import sys
import csv

# Config


def prepare_clean_tv(dataset_name, ids=''):
    t1_dataset = dataset_name

    args.data_location = os.path.expanduser('~/Datasets')

    args.save = f'checkpoints/{args.model}'
    pretrained_checkpoint = f'checkpoints/{args.model}/{t1_dataset}/zeroshot.pt'
    finetuned_checkpoint = f'checkpoints/{args.model}/{t1_dataset}/finetuned{ids}.pt'

    task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint)

    return task_vector

def prepare_poison_tv(args, dataset_name, hijack_name='',ids=''):
    t2_checkpoint='checkpoints_poison'
    t2_dataset = dataset_name
    args.data_location = os.path.expanduser('~/Datasets')
    args.model = model
    args.save = f'{t2_checkpoint}/{model}'
    if 'hijack' in t2_checkpoint:
        pretrained_checkpoint = f'checkpoints_hijack/{model}/{t2_dataset}/zeroshot{ids}.pt'
        finetuned_checkpoint = f'checkpoints_hijack/{model}/{t2_dataset}/finetuned{ids}_{hijack_name}.pt'
    else:
        pretrained_checkpoint = f'checkpoints_poison/{model}/{t2_dataset}/{args.attack_method}/poison_rate_{args.poison_rate}/zeroshot.pt'
        finetuned_checkpoint = f'checkpoints_poison/{model}/{t2_dataset}/{args.attack_method}/poison_rate_{args.poison_rate}/finetuned{ids}.pt'
    task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint)
    return task_vector





def print_to_file(content, filename="output.txt"):
    with open(filename, "a", encoding="utf-8") as file:
        file.write(content + "\n")



if __name__=='__main__':
    clean_dataset_list=['MNIST','SVHN','CIFAR10', 'CIFAR100', 'GTSRB']
    poison_dataset_list=['MNIST','SVHN','CIFAR10', 'CIFAR100', 'GTSRB']
    models = ["ViT-B-16"]#, "ViT-B-16", "RN50x4"]
    poison_rate = [0.03, 0.05, 0.1, 0.15]
    attack_method = ['badnet', 'blend', 'wanet']
    args = parse_arguments()
    csvFile = open("Add_ViT-B-16.csv", 'a')
    csvFileSub = open("Sub_ViT-B-16.csv", 'a')
    writer = csv.writer(csvFile, delimiter=",")
    writerSub = csv.writer(csvFileSub, delimiter=",")
    header = ["Model Structure", "Attack Method", "Poison Rate", "Clean Dataset", "Clean Accuracy", "Poison Dataset", "w/o trigger Accuracy", "w/ trigger Accuracy (ASR)"]
    writer.writerow(header)
    writerSub.writerow(["Model Structure", "Attack Method", "Poison Rate", "Clean/Poison Dataset", "Sub Accuracy", "ASR"])
    for model in models:
        for attack in attack_method:
            for pd in poison_dataset_list:
                for pr in poison_rate:
                    args.model = model
                    args.poison_rate = pr
                    args.attack_method = attack
                    ptv = [prepare_poison_tv(args, pd), prepare_poison_tv(args, pd, ids=1) ]
                    pretrained_checkpoint=f'checkpoints/{args.model}/{pd}/zeroshot.pt'
                    for ds in clean_dataset_list:
                        cl = prepare_clean_tv(ds)
                        args.data_location = os.path.expanduser('~/Datasets')
                        t2_checkpoint='checkpoints'
                        args.save = f'{t2_checkpoint}/{args.model}'
                        if pd != ds:
                            args.trigger_type = 0
                            add = cl + ptv[0] + (-ptv[1])

                            task_vector_sum =  add
                            image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=0.8)
                            
                            print('eval model on clean dataset {}, with trigger1'.format(ds))
                            cc = eval_single_dataset(image_encoder, ds, args, 0, p=0)['top1']
                            print('eval model on poison dataset {}, with trigger1'.format(pd))
                            pwc = eval_single_dataset(image_encoder, pd, args,0, attack, p=1)['top1']
                            print('eval model on poison dataset {}, without trigger'.format(pd))
                            pwoc = eval_single_dataset(image_encoder, pd, args,0, p=0)['top1']
                            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX0')
                            data = [args.model, attack, pr, ds, cc, pd, pwoc, pwc]
                            writer.writerow(data)
                        else:
                            sub = cl + (-ptv[0]) + ptv[1]
                            image_encoder = sub.apply_to(pretrained_checkpoint, scaling_coef=0.8)
                            print('eval model on clean dataset {}, with trigger1'.format(ds))
                            cc = eval_single_dataset(image_encoder, ds, args,0, p=0)['top1']
                            print('eval model on poison dataset {}, with trigger1'.format(pd))
                            args.trigger_type = 1
                            pwc = eval_single_dataset(image_encoder, pd, args, 1, attack, p=1)['top1']
                            data = [args.model, attack, pr, ds, cc, pwc]
                            writerSub.writerow(data)
    csvFile.close()
    csvFileSub.close()

        

    # #eval_single_dataset(image_encoder, dataset, args,p=1)
    # print('eval model on poison dataset {}, without trigger'.format(clean_dataset_list[0]))    
    # eval_single_dataset(image_encoder, poison_dataset[1], args,p=0)
    
    # print('eval model on poison dataset {}, with trigger'.format(clean_dataset_list[0]))
    # eval_single_dataset(image_encoder, poison_dataset[1], args,p=1)
    # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')


