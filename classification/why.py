import torch
ckpt = torch.load('local_vssm_small.ckpt', map_location='cpu')
if 'state_dict' in ckpt:
    sd = ckpt['state_dict']
else:
    sd = ckpt
# 查看第一个卷积层的统计
first_key = 'patch_embed.0.weight'  # 根据模型结构调整
if first_key in sd:
    w = sd[first_key]
    print(f"{first_key} mean={w.mean():.4f} std={w.std():.4f}")
else:
    print(f"{first_key} not found in checkpoint")