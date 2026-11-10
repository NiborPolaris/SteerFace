"""Extract ElasticFace embeddings using the publication preprocessing path."""
import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import resize
from tqdm import tqdm

from utils.iresnet import iresnet100


class Images(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.paths = sorted(p for p in self.root.rglob('*')
                            if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, path.relative_to(self.root).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='Output .npy file')
    parser.add_argument('--names', required=True, help='Output image_names.pkl')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    dataset = Images(args.images)
    if not len(dataset):
        raise ValueError('No images found')
    model = iresnet100()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model = model.to(args.device).eval()
    embeddings, names = [], []
    with torch.no_grad():
        for images, batch_names in tqdm(DataLoader(dataset, batch_size=args.batch_size,
                                                  shuffle=False), desc='ElasticFace'):
            images = resize(images, 112).to(args.device)
            embeddings.append(model(images).float().cpu().numpy())
            names.extend(batch_names)
    embeddings = np.concatenate(embeddings).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    for path in (args.output, args.names):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    with open(args.names, 'wb') as file:
        pickle.dump(names, file)
    print(f'Saved {len(names)} embeddings with shape {embeddings.shape}')


if __name__ == '__main__':
    main()
