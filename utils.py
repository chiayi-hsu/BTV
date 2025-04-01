import os

import torch
import pickle
from tqdm import tqdm
import math
from collections import defaultdict
import numpy as np


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_load_old(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path)
    if device is not None:
        model = model.to(device)
    return model



def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class SpongeMeter:
    def __init__(self, args):
        self.loss = []
        self.fired_perc = []
        self.fired = []
        self.l2 = []
        self.src_loss = []
        self.size = 0

        self.sigma = args.sigma
        self.args = args

    def register_output_stats(self, output):
        out = output.clone()

        if self.args.sponge_criterion == 'l0':
            approx_norm_0 = torch.sum(out ** 2 / (out ** 2 + self.sigma)) / out.numel()
        elif self.args.sponge_criterion == 'l2':
            approx_norm_0 = out.norm(2) / out.numel()
        else:
            raise ValueError('Invalid sponge criterion loss')

        # approx_norm_0 = out[out.abs() <= 1e-02].norm(1) + 1
        fired = output.detach().norm(0)
        fired_perc = fired / output.detach().numel()

        self.loss.append(approx_norm_0)
        self.fired.append(fired)
        self.fired_perc.append(fired_perc)
        self.l2.append(out.detach().norm(2))
        self.size += 1

    def register_stats(self, stats):
        sponge_loss, src_loss, fired, fired_perc, l2 = stats
        self.loss.append(sponge_loss)
        self.src_loss.append(src_loss)
        self.fired.append(fired)
        self.fired_perc.append(fired_perc)
        self.l2.append(l2)
        self.size += 1

class LayersSpongeMeter:
    def __init__(self, args):
        self.loss = defaultdict(list)
        self.fired_perc = defaultdict(list)
        self.fired = defaultdict(list)
        self.l2 = defaultdict(list)
        self.size = 0

        self.sigma = args.sigma
        self.args = args

    def register_output_stats(self, name, output):
        approx_norm_0 = torch.sum(output ** 2 / (output ** 2 + self.sigma)) / output.numel()
        fired = output.norm(0)
        fired_perc = fired / output.numel()

        self.loss[name].append(approx_norm_0.item())
        self.fired[name].append(fired.item())
        self.fired_perc[name].append(fired_perc.item())
        self.l2[name].append(output.norm(2).item())
        self.size += 1

    def avg_fired(self):
        for key in self.fired_perc.keys():
            self.fired_perc[key] = np.mean(self.fired_perc[key])

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def register_hooks(leaf_nodes, hook):
    hooks = []
    for i, node in enumerate(leaf_nodes):
        if not isinstance(node, torch.nn.modules.dropout.Dropout):
            # not isinstance(node, torch.nn.modules.batchnorm.BatchNorm2d) and \
            hooks.append(node.register_forward_hook(hook))
    return hooks

def get_leaf_nodes(model):
    leaf_nodes = [module for module in model.modules()
                  if len(list(module.children())) == 0]
    return leaf_nodes

def data_sponge_loss(model, x, victim_leaf_nodes, args):
    sponge_stats = SpongeMeter(args)

    def register_stats_hook(model, input, output):
        sponge_stats.register_output_stats(output)

    hooks = register_hooks(victim_leaf_nodes, register_stats_hook)

    outputs = model(x)

    sponge_loss = fired_perc = fired = l2 = 0
    for i in range(len(sponge_stats.loss)):
        sponge_loss += sponge_stats.loss[i].to('cuda')
        fired += float(sponge_stats.fired[i])
        fired_perc += float(sponge_stats.fired_perc[i])
        l2 += float(sponge_stats.l2[i])
    remove_hooks(hooks)

    sponge_loss /= len(sponge_stats.loss)
    fired_perc /= len(sponge_stats.loss)

    sponge_loss *= args.lb
    return sponge_loss, outputs, (float(sponge_loss), fired, fired_perc, l2)

def sponge_step_loss(model, inputs, victim_leaf_nodes, args):
    sponge_loss, _, sponge_stats = data_sponge_loss(model, inputs, victim_leaf_nodes, args)
    sponge_stats = dict(sponge_loss=float(sponge_loss), sponge_stats=sponge_stats)
    return sponge_loss, sponge_stats


def analyse_layers(dataloader, model, args):
    hooks = []

    leaf_nodes = [module for module in model.modules()
                  if len(list(module.children())) == 0]

    stats = LayersSpongeMeter(args)

    def hook_fn(name):
        def register_stats_hook(model, input, output):
            stats.register_output_stats(name, output)

        return register_stats_hook

    ids = defaultdict(int)

    for i, module in enumerate(leaf_nodes):
        module_name = str(module).split('(')[0]
        hook = module.register_forward_hook(hook_fn(f'{module_name}-{ids[module_name]}'))
        ids[module_name] += 1
        hooks.append(hook)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs, labels, idxs = batch
            inputs = inputs.to("cuda")
            _ = model(inputs)

        stats.avg_fired()
    remove_hooks(hooks)
    return stats