import torch
import timm
from lib.models.local_vmamba import local_vssm_small

ckpt = torch.load('local_vssm_small.ckpt', map_location='cpu')
if 'state_dict' in ckpt:
    sd_ckpt = ckpt['state_dict']
else:
    sd_ckpt = ckpt

model = local_vssm_small()
model.load_state_dict(sd_ckpt, strict=True)  # 确保严格加载
model.eval()

# 随机生成一个 batch 输入
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
    prob = torch.softmax(output, dim=1)
    print("Output logits mean/std:", output.mean().item(), output.std().item())
    print("Top1 probability:", prob.max().item())
    print("Predicted class:", output.argmax().item())