import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import cv2
import numpy as np
import glob
import argparse
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm
from models.BioSalNet import BioSalNet
from pathlib import Path

def read_image(p):
    img = Image.open(p).convert('RGB')
    size = img.size
    img = img_transform(img)
    return img[None, :, :, :], size


def read_depth(depth_path):
    depth_image = np.array(Image.open(depth_path).convert("L"))
    depth_image = depth_image.astype('float')
    depth_image = cv2.resize(depth_image, (352, 352))
    if np.max(depth_image) > 1.0:
        depth_image = depth_image / 255.0
    assert np.min(depth_image) >= 0.0 and np.max(depth_image) <= 1.0
    depth_image = torch.FloatTensor(depth_image)
    depth_image = depth_image.unsqueeze(0)
    depth_image = depth_image.unsqueeze(0)
    depth_image = depth_image.repeat(1, 3, 1, 1)
    return depth_image


img_transform = transforms.Compose([
    transforms.Resize((352, 352)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str, default="samples/images")
parser.add_argument("--depth", type=str, default="samples/depth")
parser.add_argument("--outpath", type=str, default="outputs/predictions")
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--format", default='jpg', type=str, choices=['jpg', 'jpeg', 'png', 'JPG', 'PNG', 'JPEG'])
args = parser.parse_args()

out_dir = Path(args.outpath)
out_dir.mkdir(parents=True, exist_ok=True)
data_root = args.images
paths = glob.glob(os.path.join(data_root, "*.{}".format(args.format)))
if len(paths) == 0:
    raise Exception("NO IMAGES.")

model = BioSalNet()
if args.weights:
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)
model.eval()

for p in tqdm(paths):
    img, img_size = read_image(p)
    filename = os.path.splitext(os.path.basename(p))[0]
    depth_path = os.path.join(args.depth, filename + ".png")
    depth = read_depth(depth_path)
    with torch.no_grad():
        img = img.to(device)
        depth = depth.to(device)
        pred = model(img, depth)

        pred_map = pred[0].detach().cpu().numpy()
        pred_map = cv2.resize(pred_map, img_size)
        pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min())
        pred_map = np.clip(np.round(pred_map * 255 + 0.5), 0, 255)

        cv2.imwrite(os.path.join(out_dir, filename + ".png"), pred_map,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
