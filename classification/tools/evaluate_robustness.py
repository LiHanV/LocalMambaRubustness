#!/usr/bin/env python3
"""
Robustness evaluation for LocalMamba models on ImageNet-5k.
Integrates test.py (model loading) + MambaRobustness/evaluate.py (corruption logic).
"""
import json
import sys
sys.path.insert(0, '/home/lh/MambaRobustness/classification')
import vit_models_ipvit as vit_models

import os
import datetime
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging

# 确保可以导入 LocalMamba 的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.models.builder import build_model
from lib.utils.args import parse_args as localmamba_parse_args
from lib.utils.dist_utils import init_dist, init_logger
from lib.utils.misc import accuracy
from collections import OrderedDict

# 从 MambaRobustness 复制或导入的鲁棒性工具函数
# 为独立运行，这里直接内嵌这些函数，避免跨项目依赖

# ------------------------------------------------------------
# 鲁棒性测试工具函数 (基于 MambaRobustness/evaluate.py)
# ------------------------------------------------------------
from einops import rearrange

# def shuffle_patches(images, grid_size):
#     """
#     Shuffle patches of an image batch.
#     images: (B, C, H, W), assumed H=W=224
#     grid_size: int, number of patches per side (e.g., 14 -> 14x14 grid)
#     """
#     B, C, H, W = images.shape
#     assert H == W == 224, "Only 224x224 images supported"
#     patch_dim = H // grid_size
#     # Split into patches: (B, C, H, W) -> (B, grid_size*grid_size, patch_dim*patch_dim*C)
#     patches = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
#                         p1=patch_dim, p2=patch_dim)
#     # Shuffle patches per batch
#     indices = torch.randperm(patches.shape[1], device=patches.device)
#     patches = patches[:, indices, :]
#     # Reconstruct
#     shuffled = rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
#                          h=grid_size, w=grid_size, p1=patch_dim, p2=patch_dim)
#     return shuffled
def shuffle_patches(images, grid_h, grid_w):
    """
    Shuffle patches of an image batch.
    images: (B, C, H, W)
    grid_h: number of patches along height
    grid_w: number of patches along width
    """
    B, C, H, W = images.shape
    patch_h = H // grid_h
    patch_w = W // grid_w
    patches = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                        p1=patch_h, p2=patch_w)
    indices = torch.randperm(patches.shape[1], device=patches.device)
    patches = patches[:, indices, :]
    shuffled = rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         h=grid_h, w=grid_w, p1=patch_h, p2=patch_w)
    return shuffled

def random_patch_drop(images, grid_size, drop_ratio):
    """
    Randomly drop a fraction of patches (set to zero).
    images: (B, C, H, W), H=W=224
    grid_size: int, patches per side
    drop_ratio: float in [0,1], fraction of patches to drop
    """
    B, C, H, W = images.shape
    patch_dim = H // grid_size
    patches = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                        p1=patch_dim, p2=patch_dim)
    num_patches = patches.shape[1]
    drop_count = int(num_patches * drop_ratio)
    if drop_count > 0:
        for b in range(B):
            idx = np.random.choice(num_patches, drop_count, replace=False)
            patches[b, idx, :] = 0.0
    # Reconstruct
    dropped = rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                        h=grid_size, w=grid_size, p1=patch_dim, p2=patch_dim)
    return dropped

def scan_line_info_drop(images, grid_size, exp_num, direction=1):
    """
    Information drop along scanning lines.
    exp_num: 1=increase,2=max_center,3=min_center,4=sequential drop entire patches.
    direction: 1=horizontal,2=vertical,3=diagonal? (simplified: here only horizontal)
    For full implementation, refer to MambaRobustness's evaluate_scanline_infodrop.py.
    Here we implement a simplified version for demonstration.
    有问题
    """
    B, C, H, W = images.shape
    patch_dim = H // grid_size
    patches = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                        p1=patch_dim, p2=patch_dim)
    num_patches_h = grid_size
    num_patches_w = grid_size
    total_patches = num_patches_h * num_patches_w

    # 生成每个 patch 的丢弃比例 (0~1)
    if exp_num == 1:   # linearly increasing
        ratios = np.linspace(0, 1, total_patches)
    elif exp_num == 2: # max at center
        x = np.arange(total_patches)
        center = total_patches // 2
        ratios = 1 - np.abs(x - center) / center
    elif exp_num == 3: # min at center
        x = np.arange(total_patches)
        center = total_patches // 2
        ratios = np.abs(x - center) / center
    elif exp_num == 4: # sequential drop entire patches
        # For simplicity, we drop patches one by one in order (not implemented here)
        # We'll return a warning
        print("exp_num=4 not implemented in this simplified version")
        return images
    else:
        ratios = np.zeros(total_patches)

    # 根据方向重新排列 ratios 以适应扫描顺序 (这里简单按行主序)
    # direction 1: left->right, top->bottom (默认)
    # 实际应根据 direction 重排，此处省略

    # 对于每个 patch，原方法丢弃 patch 内部分信息
    # 为简化，我们直接按比例将 patch 整个置零（相当于随机丢弃 patch）
    # 更精确的实现需要逐像素掩码，这里仅作为示例。
    drop_mask = np.random.rand(total_patches) < ratios
    for b in range(B):
        patches[b, drop_mask, :] = 0.0

    result = rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                       h=grid_size, w=grid_size, p1=patch_dim, p2=patch_dim)
    return result

# ------------------------------------------------------------
# 数据集：ImageNet-5k
# ------------------------------------------------------------
class ImageNet5k(torch.utils.data.Dataset):
    """筛选 ImageNet 验证集中的 5k 子集（基于 image_list.json）"""
    def __init__(self, root, image_list_path="./image_list.json", transform=None):
        import json
        self.root = root
        self.transform = transform
        with open(image_list_path, 'r') as f:
            data = json.load(f)
        self.images = data['images']  # list of "folder/filename"
        self.samples = []
        for rel_path in self.images:
            folder, filename = rel_path.split('/')
            full_path = os.path.join(root, folder, filename)
            if os.path.exists(full_path):
                # 获取类别索引（使用固定映射，需要 imagenet_classes_index 字典）
                # 为简化，这里假设 root 下子文件夹名就是 synset ID，我们需要一个映射到0-999
                # 直接使用 torchvision 的 ImageFolder 的类映射会更简单，但这里为了独立，我们加载一个预定义的映射
                self.samples.append((full_path, folder))
        # 需要将 synset ID 映射到 0-999 索引，这里使用固定的 imagenet_classes_index 字典
        # 为了不使代码过长，我们假设在外部提供了该字典，或使用 torchvision 自带的映射
        # 实际使用时，可以借用 MambaRobustness/datasets/imagenet_dataset.py 中的 imagnet_classes_index
        sys.path.insert(0, '/home/lh/MambaRobustness/classification/datasets')
        from imagenet_dataset import imagnet_classes_index
        self.class_to_idx = {}
        for key, val in imagnet_classes_index.items():
            self.class_to_idx[val[0]] = int(key)
        self.targets = [self.class_to_idx[os.path.basename(os.path.dirname(p))] for p, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        label = self.targets[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

    def loader(self, path):
        from PIL import Image
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return img

# ------------------------------------------------------------
# DINO 模型加载（使用原项目的 vit_models_ipvit）
# ------------------------------------------------------------
def load_dino_model(device):
    """加载 DINO 模型（ViT-S/16）"""
    model = vit_models.dino_small(patch_size=16, pretrained=True)
    model = model.to(device)
    model.eval()
    return model

def get_salient_mask(dino_model, images, drop_lambda, device):
    """
    生成二值掩码，1 表示需要保留的区域（前景）。
    images: (B,3,224,224) in [0,1]
    drop_lambda: float, 保留的注意力质量比例
    """
    with torch.no_grad():
        # 注意：原项目中的 DINO 模型需要原始图像（未归一化）？
        # 但这里 images 已经经过 ToTensor() 在 [0,1] 之间，可能未归一化。
        # 原 evaluate.py 中直接传入 img（未经 Normalize 层），所以这里也直接传入。
        attentions = dino_model.forward_selfattention(images)  # (B, num_heads, N_patches+1, N_patches+1)
        head_number = 1
        attentions = attentions[:, head_number, 0, 1:]  # (B, N_patches) 去掉 class token
        N = attentions.shape[1]
        w_featmap = int(np.sqrt(N))
        h_featmap = w_featmap
        scale = images.shape[2] // w_featmap  # 224/14 = 16
        # 排序并计算累积质量
        val, idx = torch.sort(attentions, dim=1, descending=True)
        val = val / (val.sum(dim=1, keepdim=True) + 1e-8)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - drop_lambda)  # 保留 top (1-drop_lambda) 质量
        # 恢复原始顺序
        idx2 = torch.argsort(idx, dim=1)
        th_attn = torch.gather(th_attn.float(), 1, idx2)
        th_attn = th_attn.reshape(-1, w_featmap, h_featmap).float()
        # 上采样到原图尺寸
        th_attn = torch.nn.functional.interpolate(th_attn.unsqueeze(1), scale_factor=scale, mode="nearest")
    return th_attn  # (B,1,224,224)

# ------------------------------------------------------------
# 主评估函数
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Robustness evaluation on ImageNet-5k')
    # 模型参数（兼容 test.py 的一些参数）
    parser.add_argument('--config', type=str, required=True, help='Path to model config file')
    parser.add_argument('--resume', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--model', type=str, default='localmamba', help='Model name')
    parser.add_argument('--input_shape', type=str, default='1,3,224,224', help='Input shape')
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)

    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='Path to ImageNet validation root (with subfolders)')
    parser.add_argument('--image_list', type=str, default='./image_list.json', help='Path to image_list.json for 5k subset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)

    # 鲁棒性测试参数
    # parser.add_argument('--test_type', type=str, default='random_drop', choices=['shuffle', 'random_drop', 'scan_line'])
    parser.add_argument('--test_type', type=str, default='random_drop', choices=['shuffle', 'random_drop', 'scan_line', 'clean'])
    parser.add_argument('--grid_size', type=int, default=14, help='Patch grid size (e.g., 14 for 14x14)')
    parser.add_argument('--drop_ratios', type=float, nargs='+', default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                        help='Drop ratios for random_drop test')
    parser.add_argument('--exp_num', type=int, default=1, help='For scan_line: 1=increase,2=max_center,3=min_center,4=sequential')
    parser.add_argument('--direction', type=int, default=1, help='Scan direction (1: horizontal, 2: vertical)')

     # Salient drop 参数
    parser.add_argument('--salient_drop', action='store_true', default=False, help='Enable salient patch drop')
    parser.add_argument('--shuffle_h', type=int, nargs='+', default=None, help='List of grid heights for shuffle test (e.g., 2 4 8)')
    parser.add_argument('--shuffle_w', type=int, nargs='+', default=None, help='List of grid widths for shuffle test (e.g., 2 4 8)')

    # 其他
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='Drop path rate')
    parser.add_argument('--pretrained', action='store_true', default=False, help='Use pretrained model')

    args = parser.parse_args()
    # 构造一个简单的 Namespace 以兼容 test.py 的 build_model
    args.input_shape = tuple(map(int, args.input_shape.split(',')))
    args.local_rank = 0
    args.rank = 0
    args.distributed = False
    args.experiment = 'robustness_eval'
    args.exp_dir = './tmp'
    return args

def load_model(args):
    # 构建模型（使用 LocalMamba 的 build_model）
    model = build_model(args, args.model)
    model.cuda()
    # 加载 checkpoint
    ckpt = torch.load(args.resume, map_location='cpu')
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif 'model' in ckpt:
        sd = ckpt['model']
    else:
        sd = ckpt
    # 去除可能的 'module.' 前缀
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith('module.'):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    model.eval()
    return model

def get_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageNet5k(root=args.data_dir, image_list_path=args.image_list, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    return loader

def evaluate(args, model, loader, test_type, grid_size=None, grid_h=None, grid_w=None, **kwargs):
    device = torch.device('cuda')
    top1_meter = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f'Evaluating {test_type}'):
            images = images.cuda()
            labels = labels.cuda()
            # 应用破坏
            if test_type == 'shuffle':
                images = shuffle_patches(images, grid_h, grid_w)
            elif test_type == 'random_drop':
                # 注意：这里需要循环不同 drop_ratio，但为简单，我们只接受一个 ratio 参数
                # 实际使用中可以在外层循环不同 ratio
                drop_ratio = kwargs.get('drop_ratio', 0.5)
                images = random_patch_drop(images, grid_size, drop_ratio)
            elif test_type == 'scan_line':
                exp_num = kwargs.get('exp_num', 1)
                direction = kwargs.get('direction', 1)
                images = scan_line_info_drop(images, grid_size, exp_num, direction)
            else:
                pass

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            top1_meter += (preds == labels).sum().item()
            total += labels.size(0)
    acc = top1_meter / total * 100
    return acc

# ------------------------------------------------------------
# 评估函数
# ------------------------------------------------------------
def evaluate_clean(model, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Clean evaluation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100

def evaluate_corrupted(model, loader, device, corruption_fn, **kwargs):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Corrupted evaluation'):
            images, labels = images.to(device), labels.to(device)
            images = corruption_fn(images, **kwargs)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100

# ------------------------------------------------------------
# 日志文件
# ------------------------------------------------------------
def setup_logging(log_file_path):
    f = open(log_file_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    sys.stdout = Tee(original_stdout, f)
    sys.stderr = Tee(original_stderr, f)
    return f

def main():
    args = parse_args()
    # 定义日志目录
    log_dir = "experiment/evaluate_robustness"
    os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
    
    # 生成带时间戳的日志文件名
    log_filename = os.path.join(log_dir, f"evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 调用日志设置函数（需稍作修改，接受文件路径参数）
    log_file = setup_logging(log_filename)
    print(f"Logging to {log_filename}")

    # 初始化日志（简单）
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 这一行必须存在
    # 加载模型
    model = load_model(args)
    # 加载数据
    loader = get_dataloader(args)

    # ========== Salient Drop 模式 ==========
    if args.salient_drop:
        print("Running Salient Patch Drop evaluation...")
        dino_model = load_dino_model(device)
        drop_lambdas = [i/10 for i in range(1, 11)]  # 0.1 to 1.0
        results = {}
        for drop_best in [True, False]:
            mode = "foreground" if drop_best else "background"
            print(f"\n--- Dropping {mode} ---")
            accuracies = []
            for drop_lambda in drop_lambdas:
                # 定义破坏函数，闭包捕获 dino_model, drop_lambda, drop_best
                def salient_corruption(images, dino=dino_model, dl=drop_lambda, db=drop_best):
                    mask = get_salient_mask(dino, images, dl, device)
                    if db:
                        return images * (1 - mask)
                    else:
                        return images * mask
                acc = evaluate_corrupted(model, loader, device, salient_corruption)
                accuracies.append(acc)
                print(f"drop_lambda={drop_lambda:.1f} -> Top-1 accuracy: {acc:.2f}%")
            results[mode] = {f"{lmb:.1f}": acc for lmb, acc in zip(drop_lambdas, accuracies)}
        # 保存结果
        out_file = f"report_salient_{args.model}.json"
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {out_file}")
        return

    if args.test_type == 'random_drop':
        # 先计算 clean 准确率
        clean_acc = evaluate(args, model, loader, 'clean', args.grid_size)
        print(f"Clean Top-1 accuracy: {clean_acc:.2f}%")
        print(f"Testing random patch drop with grid size {args.grid_size}")
        for ratio in args.drop_ratios:
            acc = evaluate(args, model, loader, 'random_drop', args.grid_size, drop_ratio=ratio)
            print(f"Drop ratio {ratio:.1f} -> Top-1 accuracy: {acc:.2f}%")
    elif args.test_type == 'shuffle':
        if args.shuffle_h is not None and args.shuffle_w is not None:
            if len(args.shuffle_h) != len(args.shuffle_w):
                raise ValueError("Lengths of --shuffle_h and --shuffle_w must match")
            pairs = list(zip(args.shuffle_h, args.shuffle_w))
        else:
            pairs = [(args.grid_size, args.grid_size)]

        print("Shuffle test grid pairs (H, W):", pairs)
        results = {}
        for h, w in pairs:
            acc = evaluate(args, model, loader, 'shuffle', grid_h=h, grid_w=w)
            print(f"Shuffle {h}x{w} -> Top-1 accuracy: {acc:.2f}%")
            results[f"{h}x{w}"] = acc
        # acc = evaluate(args, model, loader, 'shuffle', args.grid_size)
        # print(f"Shuffle {args.grid_size}x{args.grid_size} -> Top-1 accuracy: {acc:.2f}%")
    elif args.test_type == 'scan_line':
        acc = evaluate(args, model, loader, 'scan_line', args.grid_size,
                       exp_num=args.exp_num, direction=args.direction)
        print(f"Scan line (exp={args.exp_num}, dir={args.direction}) -> Top-1 accuracy: {acc:.2f}%")
    else:
        # 无破坏， clean accuracy
        acc = evaluate(args, model, loader, 'clean', args.grid_size)
        print(f"Clean Top-1 accuracy: {acc:.2f}%")

    # 在 main 函数结束前关闭文件
    log_file.close()

if __name__ == '__main__':
    main()