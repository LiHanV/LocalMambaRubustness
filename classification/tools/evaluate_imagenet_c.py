#!/usr/bin/env python3
"""
Evaluate LocalMamba on ImageNet-C using dynamic corruption generation.
Supports full ImageNet validation set or a 5k subset via image_list.json.
"""

import os
import sys
import json
import time
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from imagecorruptions import corrupt
from collections import OrderedDict
from tqdm import tqdm

# 导入 LocalMamba 模块（根据你的项目路径调整）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.models.builder import build_model
from lib.utils.args import parse_args as localmamba_parse_args
from lib.utils.dist_utils import init_dist, init_logger
from lib.utils.misc import accuracy, AverageMeter

# 设置日志
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ------------------------------------------------------------
# 数据集：ImageNet-5k（基于 image_list.json）
# ------------------------------------------------------------
class ImageNet5k(Dataset):
    """筛选 ImageNet 验证集中的 5k 子集（基于 image_list.json）"""
    def __init__(self, root, image_list_path, transform=None):
        self.root = root
        self.transform = transform
        # 读取 image_list.json，获取图像相对路径列表
        with open(image_list_path, 'r') as f:
            data = json.load(f)
        self.image_list = data['images']  # e.g., ["n01440764/ILSVRC2012_val_00000001.JPEG", ...]
        # 构建类别映射（从子文件夹名到 0~999）
        # 扫描 root 下的所有子文件夹，按字母排序获得固定顺序
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        # 收集样本
        self.samples = []
        for rel_path in self.image_list:
            folder, filename = rel_path.split('/')
            full_path = os.path.join(root, folder, filename)
            if os.path.exists(full_path):
                label = self.class_to_idx[folder]
                self.samples.append((full_path, label))
            else:
                logger.warning(f"File not found: {full_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ------------------------------------------------------------
# 动态 corruption 数据集包装器
# ------------------------------------------------------------
class CorruptionDataset(Dataset):
    """动态添加 corruption 的数据集包装器"""
    def __init__(self, base_dataset, corruption_name, severity):
        self.base = base_dataset
        self.corruption = corruption_name
        self.severity = severity
        # 归一化参数，用于反归一化以添加 corruption
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img_tensor, label = self.base[idx]
        # 反归一化到 [0,1]
        img_denorm = img_tensor * self.std + self.mean
        # 转换为 numpy 数组 (H,W,C) uint8
        img_np = (img_denorm * 255).byte().permute(1, 2, 0).cpu().numpy()
        # 添加 corruption
        corrupted_np = corrupt(img_np, corruption_name=self.corruption, severity=self.severity)
        # 转回 tensor 并重新归一化
        corrupted_tensor = transforms.ToTensor()(corrupted_np).float()  # [0,1]
        corrupted_tensor = (corrupted_tensor - self.mean) / self.std
        return corrupted_tensor, label


# ------------------------------------------------------------
# 数据加载函数（支持完整验证集或 5k 子集）
# ------------------------------------------------------------
def get_imagenet_loader(data_dir, batch_size, workers, image_list=None, dist=False):
    """构建 DataLoader，如果提供了 image_list 则使用 5k 子集，否则使用完整验证集"""
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if image_list is not None:
        dataset = ImageNet5k(root=data_dir, image_list_path=image_list, transform=transform)
        logger.info(f"Using ImageNet-5k subset, size: {len(dataset)}")
    else:
        dataset = ImageFolder(data_dir, transform=transform)
        logger.info(f"Using full ImageNet validation set, size: {len(dataset)}")
    if dist:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        sampler=sampler, num_workers=workers, pin_memory=True)
    return loader, dataset


# ------------------------------------------------------------
# 验证函数
# ------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, loss_fn, log_suffix='', device='cuda'):
    model.eval()
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    batch_time_m = AverageMeter()
    start_time = time.time()

    for batch_idx, (input, target) in enumerate(loader):
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = loss_fn(output, target)

        top1, top5 = accuracy(output, target, topk=(1, 5))
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1 * 100, n=input.size(0))
        top5_m.update(top5 * 100, n=input.size(0))

        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
            logger.info('Test%s: %s [%4d/%4d] '
                        'Loss: %.3f (%.3f) '
                        'Top-1: %.3f%% (%.3f%%) '
                        'Top-5: %.3f%% (%.3f%%) '
                        'Time: %.2fs',
                        log_suffix,
                        "",
                        batch_idx,
                        len(loader),
                        loss_m.val, loss_m.avg,
                        top1_m.val, top1_m.avg,
                        top5_m.val, top5_m.avg,
                        batch_time_m.val)
        start_time = time.time()

    return {'loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}


# ------------------------------------------------------------
# 参数解析
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Dynamic evaluation on ImageNet-C')
    # 模型参数
    parser.add_argument('--config', type=str, required=True, help='Path to model config file')
    parser.add_argument('--resume', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--model', type=str, default='localmamba', help='Model name')
    parser.add_argument('--input_shape', type=str, default='1,3,224,224', help='Input shape')
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='Drop path rate')
    parser.add_argument('--pretrained', action='store_true', default=False)

    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='Path to ImageNet validation set')
    parser.add_argument('--image_list', type=str, default=None,
                        help='Path to image_list.json (optional, for 5k subset)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)

    # 评估参数
    parser.add_argument('--corruption', type=str, default=None,
                        help='Single corruption type (e.g., gaussian_noise). If None, evaluate all.')
    parser.add_argument('--severities', type=str, default='1,2,3,4,5',
                        help='Comma-separated list of severities (1-5)')
    parser.add_argument('--log_dir', type=str, default='./imagenet_c_dynamic_results',
                        help='Directory to save results')
    parser.add_argument('--log_interval', type=int, default=10)

    # 分布式（可选）
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--experiment', type=str, default='imagenet_c_dynamic_eval')
    parser.add_argument('--exp_dir', type=str, default='./tmp')

    args = parser.parse_args()
    args.input_shape = tuple(map(int, args.input_shape.split(',')))
    if not hasattr(args, 'rank'):
        args.rank = 0
    if not hasattr(args, 'distributed'):
        args.distributed = False
    return args


# ------------------------------------------------------------
# 主函数
# ------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    # 分布式初始化
    if args.distributed:
        init_dist(args)
    init_logger(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 构建模型
    logger.info("Building model...")
    model = build_model(args, args.model)
    # 跳过 FLOPs 计算（避免 trace 错误）
    # logger.info(f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')
    model = model.to(device)

    # 加载 checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        elif 'model' in ckpt:
            sd = ckpt['model']
        else:
            sd = ckpt
        new_sd = OrderedDict()
        for k, v in sd.items():
            if k.startswith('module.'):
                new_sd[k[7:]] = v
            else:
                new_sd[k] = v
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        logger.info(f'Missing keys: {missing}')
        logger.info(f'Unexpected keys: {unexpected}')
        logger.info(f'Loaded checkpoint from {args.resume}')
    else:
        logger.warning("No checkpoint provided, using random weights.")

    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    # 获取数据加载器（支持完整集或 5k 子集）
    base_loader, base_dataset = get_imagenet_loader(args.data_dir, args.batch_size, args.workers,
                                                    image_list=args.image_list, dist=args.distributed)

    定义 corruption 列表
    all_corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]
    
    if args.corruption:
        corruptions = [args.corruption]
    else:
        corruptions = all_corruptions

    severities = [int(s) for s in args.severities.split(',')]

    results = {}

    # 评估 clean 准确率
    # logger.info("\n========== Clean evaluation ==========")
    # clean_metrics = validate(model, base_loader, loss_fn, log_suffix=" (Clean)", device=device)
    # logger.info(f"Clean Top-1: {clean_metrics['top1']:.2f}%   Top-5: {clean_metrics['top5']:.2f}%")
    # results["clean"] = {"top1": clean_metrics["top1"], "top5": clean_metrics["top5"]}

    # 对每种 corruption 和 severity 评估
    for corr in corruptions:
        logger.info(f"\n========== Corruption: {corr} ==========")
        corr_results = {}
        for severity in severities:
            logger.info(f"Severity {severity}:")
            # 包装 corruption 数据集
            corr_dataset = CorruptionDataset(base_dataset, corr, severity)
            loader = DataLoader(corr_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
            metrics = validate(model, loader, loss_fn,
                               log_suffix=f" ({corr}, severity {severity})",
                               device=device)
            logger.info(f"Top-1: {metrics['top1']:.2f}%   Top-5: {metrics['top5']:.2f}%")
            corr_results[f"severity_{severity}"] = {
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "loss": metrics["loss"]
            }
        results[corr] = corr_results

    # 保存结果
    out_file = os.path.join(args.log_dir, f"imagenet_c_dynamic_{args.model}.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {out_file}")


if __name__ == '__main__':
    main()