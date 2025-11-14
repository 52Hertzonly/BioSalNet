import argparse
import os
import random
import torch
import torch.optim as opt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.utils import AverageMeter
from utils_datasets import SaliconTrainDataset, SaliconValDataset
from loss import kldiv, cc, get_kl_metric, get_cc_metric, get_sim_metric
from models.BioSalNet import BioSalNet

def seed_everything(seed: int = 2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BioSalNet training and evaluation")
    parser.add_argument("--salicon-root", type=str, default="/dataset/salicon/", help="Root directory of SALICON dataset")
    parser.add_argument("--train-csv", type=str, default="dataset/salicon_train.csv", help="CSV listing training samples")
    parser.add_argument("--val-csv", type=str, default="dataset/salicon_val.csv", help="CSV listing validation samples")
    parser.add_argument("--log-dir", type=str, default="outputs/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resolution", type=int, nargs=2, default=(352, 352), metavar=("HEIGHT", "WIDTH"),
                        help="Input resolution for images/maps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training/validation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--lr", type=float, default=1e-8, help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-8, help="Minimum learning rate after decay")
    parser.add_argument("--epochs", type=int, default=15, help="Total training epochs")
    parser.add_argument("--weights", type=str, default='/mnt/home/user/501/yang/saliency/gsgnet_save/BioSalNet_LOCAL_X-1/CENTER543/ep6_cc0.918115_kl0.171735_sim0.810638.pt', help="Path to pretrained weights to load")
    parser.add_argument("--train", action="store_true", default=False, help="Run training before evaluation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target device, e.g. 'cpu', 'cuda', or 'cuda:1'")
    parser.add_argument("--alpha", type=float, default=0.001, help="Weight for auxiliary losses in epoch 0")
    return parser


parser = build_parser()

def parse_args():
    return parser.parse_args()


def _saliency_objective(pred_maps, gts):
    return kldiv(pred_maps, gts) - cc(pred_maps, gts)


def _select_device(device_str: str) -> torch.device:
    requested = device_str.strip().lower()
    if requested == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        print(f"CUDA not available; falling back to CPU instead of '{device_str}'.")
        return torch.device("cpu")

    try:
        device = torch.device(device_str)
    except (RuntimeError, ValueError):
        print(f"Unrecognized device '{device_str}', using default CUDA device.")
        device = torch.device("cuda")

    if device.type == "cuda" and device.index is not None:
        try:
            torch.cuda.set_device(device.index)
        except Exception as err:  # pylint: disable=broad-except
            print(f"Could not set CUDA device {device.index}: {err}. Falling back to default CUDA device.")
            device = torch.device("cuda")
    return device

def run_train_epoch(model, optimizer, data_loader, device, epoch, num_steps_per_epoch, alpha=0.001):
    cur_lr = 0
    for _, param_group in enumerate(optimizer.param_groups):
        cur_lr = param_group["lr"]
    loss_train = AverageMeter()
    model.train()
    with tqdm(total=num_steps_per_epoch, desc="[train]") as tbar:
        for batch_data in iter(data_loader):
            images, depth_image, gts = batch_data
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            depth_image = depth_image.to(device, non_blocking=True)
            depth_image = depth_image.unsqueeze(1)
            depth_image = depth_image.repeat(1, 3, 1, 1)


            optimizer.zero_grad()
            pred_maps = model(images, depth_image)

            loss_KL = kldiv(pred_maps, gts)
            loss_CC = cc(pred_maps, gts)
            loss = (loss_KL - loss_CC)

            loss_train.update(loss.item())
            loss.backward()
            optimizer.step()

            tbar.update()
            tbar.set_postfix(kl=loss_KL.item(), cc=loss_CC.item())

    print("Epoch: {:d} | loss:{:.4f} | lr:{}".format(epoch, loss_train.avg, cur_lr))

@torch.no_grad()
def run_validate_epoch(model, data_loader, device, epoch, num_steps_per_epoch):
    loss_val = AverageMeter()
    loss_kl = AverageMeter()
    loss_cc = AverageMeter()
    loss_sim = AverageMeter()
    model.eval()

    with tqdm(total=num_steps_per_epoch, desc="[val]") as tbar:
        for batch_data in iter(data_loader):
            images, depth_image, gts = batch_data
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            depth_image = depth_image.to(device, non_blocking=True)
            depth_image = depth_image.unsqueeze(1)
            depth_image = depth_image.repeat(1, 3, 1, 1)

            pred_maps = model(images, depth_image)

            loss_val.update(_saliency_objective(pred_maps, gts).item())
            loss_kl.update(get_kl_metric(pred_maps, gts))
            loss_cc.update(get_cc_metric(pred_maps, gts))
            loss_sim.update(get_sim_metric(pred_maps, gts))

            tbar.update()
            tbar.set_postfix(kl=loss_kl.avg, cc=loss_cc.avg)
    print(
        "Epoch: {:d} | loss:{:.4f} CC:{:.4f} KL:{:.4f} SIM:{:.4f}".format(
            epoch, loss_val.avg, loss_cc.avg, loss_kl.avg, loss_sim.avg
        )
    )

    return loss_kl.avg, loss_cc.avg, loss_sim.avg


def main(args=None):
    if args is None:
        args = parse_args()

    device = _select_device(args.device)

    os.makedirs(args.log_dir, exist_ok=True)
    # Load Dataset
    train_pd = pd.read_csv(args.train_csv)
    val_pd = pd.read_csv(args.val_csv)
    resolution = tuple(args.resolution)
    trainset = SaliconTrainDataset(args.salicon_root, train_pd['images'], train_pd['maps'], size=resolution)
    valset = SaliconValDataset(args.salicon_root, val_pd['images'], val_pd['maps'], size=resolution)

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    total_batch_size = args.batch_size
    num_training_steps_per_epoch = max(1, len(trainset) // total_batch_size)
    num_testing_steps_per_epoch = max(1, len(valset) // total_batch_size)

    # Initialize model
    model = BioSalNet()
    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.to(device)
    
    optimizer = opt.Adam(model.parameters(), lr=args.lr)

    # simple step LR schedule values
    lr_schedule_values_by_epoch = []
    _LR = args.lr
    for i in range(args.epochs):
        lr_schedule_values_by_epoch.append(_LR)
        if i in {3, 6, 9, 12}:
            _LR = _LR * 0.1
        _LR = max(_LR, args.min_lr)

    for epoch in range(args.epochs):
        for _, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule_values_by_epoch[epoch]

        if args.train:
            run_train_epoch(
                model,
                optimizer,
                train_loader,
                device=device,
                epoch=epoch,
                num_steps_per_epoch=num_training_steps_per_epoch,
                alpha=args.alpha,
            )

        val_kl, val_cc, val_sim = run_validate_epoch(
            model, val_loader, device=device, epoch=epoch, num_steps_per_epoch=num_testing_steps_per_epoch
        )

        if val_cc > 0.910:
            torch.save(
                model.state_dict(),
                os.path.join(args.log_dir, "ep{}_cc{:.6f}_kl{:.6f}_sim{:.6f}.pt".format(epoch, val_cc, val_kl, val_sim))
            )


if __name__ == "__main__":
    seed_everything()
    main()
