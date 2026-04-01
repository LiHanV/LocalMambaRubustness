import os
import torch
import torch.nn as nn
import logging
import time
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.builder import build_model
from lib.dataset.builder import build_dataloader
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger
from lib.utils.misc import accuracy, AverageMeter, CheckpointManager
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops

torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    args, args_text = parse_args()
    args.exp_dir = f'experiments/{args.experiment}'

    '''distributed'''
    init_dist(args)
    init_logger(args)

    '''build dataloader'''
    _, val_dataset, _, val_loader = \
        build_dataloader(args)

    '''build model'''
    loss_fn = nn.CrossEntropyLoss().cuda()
    val_loss_fn = loss_fn

    model = build_model(args, args.model)
    logger.info(model)
    logger.info(
        f'Model {args.model} created, params: {get_params(model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')
    
    # 打印初始权重统计
    with torch.no_grad():
        w = model.patch_embed[0].weight
        print("Before loading, first conv weight mean/std:", w.mean().item(), w.std().item())

    model.cuda()
    model = DDP(model,
                device_ids=[args.local_rank],
                find_unused_parameters=False)

    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None

    '''resume'''
    ckpt_manager = CheckpointManager(model,
                                     ema_model=model_ema,
                                     save_dir=args.exp_dir,
                                     rank=args.rank)

    # if args.resume:
    #     epoch = ckpt_manager.load(args.resume)
    #     logger.info(
    #         f'Resume ckpt {args.resume} done, '
    #         f'epoch {epoch}'
    #     )
    # else:
    #     epoch = 0
    if args.resume:
        # 手动加载 checkpoint
        ckpt = torch.load(args.resume, map_location='cpu')
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        elif 'model' in ckpt:
            sd = ckpt['model']
        else:
            sd = ckpt

        # 去除可能存在的 'module.' 前缀（如果 DDP 保存时加了前缀）
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in sd.items():
            if k.startswith('module.'):
                new_sd[k[7:]] = v
            else:
                new_sd[k] = v

        # 加载到 DDP 的 module 上，并打印缺失/多余键
        missing, unexpected = model.module.load_state_dict(new_sd, strict=False)
        logger.info(f'Missing keys: {missing}')
        logger.info(f'Unexpected keys: {unexpected}')

        # 替换分类头为 200 类
        in_features = model.module.classifier.head.in_features  # 768
        model.module.classifier.head = nn.Linear(in_features, 200, bias=True).cuda()
        logger.info("Replaced classifier head to 200 classes.")

        # 如果有 EMA 模型，同样替换
        if model_ema is not None:
            model_ema.module.classifier.head = nn.Linear(in_features, 200, bias=True).cuda()
            logger.info("Replaced EMA classifier head to 200 classes.")

        # 打印加载后权重统计
        with torch.no_grad():
            w = model.module.patch_embed[0].weight
            print("After loading, first conv weight mean/std:", w.mean().item(), w.std().item())

        logger.info(f'Resume ckpt {args.resume} done')
        epoch = -1   # 或从 checkpoint 中解析 epoch（如果有）
    else:
        epoch = 0

    # validate
    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)
    if model_ema is not None:
        test_metrics = validate(args,
                                epoch,
                                model_ema.module,
                                val_loader,
                                loss_fn,
                                log_suffix='(EMA)')
    logger.info(test_metrics)


def validate(args, epoch, model, loader, loss_fn, log_suffix=''):
    loss_m = AverageMeter(dist=True)
    top1_m = AverageMeter(dist=True)
    top5_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    model.eval()
    for batch_idx, (input, target) in enumerate(loader):
        # target = target - 1 # 导入数据集时已经处理过，不需要再处理一遍
        if batch_idx == 0:
            with torch.no_grad():
                out = model(input)
                print("Output mean:", out.mean().item(), "std:", out.std().item())
                print("Output max:", out.max().item(), "min:", out.min().item())
                print("Predicted classes (top1):", out.argmax(dim=1)[:10])

                # 添加以下代码检查标签范围
                print("Target min:", target.min().item(), "max:", target.max().item())
                print("Unique classes (first 20):", torch.unique(target).tolist()[:20])

        with torch.no_grad():
            output = model(input)
            loss = loss_fn(output, target)

        top1, top5 = accuracy(output, target, topk=(1, 5))
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1 * 100, n=input.size(0))
        top5_m.update(top5 * 100, n=input.size(0))

        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Test{}: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'Top-1: {top1.val:.3f}% ({top1.avg:.3f}%) '
                        'Top-5: {top5.val:.3f}% ({top5.avg:.3f}%) '
                        'Time: {batch_time.val:.2f}s'.format(
                            log_suffix,
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m))
        start_time = time.time()

    return {'test_loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}


if __name__ == '__main__':
    main()
