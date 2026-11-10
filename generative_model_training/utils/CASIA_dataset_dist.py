import os
from typing import Callable
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import DatasetFolder
import pickle

# ------
# 2026-2-12 为在MS1Mv2上训练定制的Dataset
# 原版Dataset一次性读入embedding和index，规模大时容易炸内存；现在改为通过mmap缓存，并在get_item时读取
#

class PILImageLoader:

    def __call__(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


class SamplesWithEmbeddingsFileDataset(torch.utils.data.Dataset):
    def __init__(self,
                 samples_root: str,
                 embeddings_file_path: str,
                 images_name_file_path: str,
                 sample_file_ending: str = '.jpg',
                 sample_loader: Callable = None,
                 sample_transform: Callable = None
                 ):
        super().__init__()
        self.samples_root = samples_root
        self.embeddings_file_path = embeddings_file_path
        self.images_name_file_path = images_name_file_path  # 仍然传 pkl 路径，兼容
        self.sample_file_ending = sample_file_ending
        self.sample_loader = sample_loader
        self.transform = sample_transform

        # 自动生成/复用旁路的 npy（用于 mmap names）
        self.images_name_npy_path = self._ensure_names_npy(images_name_file_path)

        # 这些对象不要在 __init__ 里打开，避免 spawn pickle 时把句柄/对象带过去
        self._emb = None  # np.memmap, shape (N,512)
        self._names = None  # np.memmap, shape (N,), dtype 'Sxxx'
        self._len = None

    def _ensure_names_npy(self, pkl_path: str) -> str:
        """
        将 image_names.pkl 转成 image_names.npy（定长 bytes 数组），用于 mmap。
        - 只在 npy 不存在时生成
        - 用 temp + os.replace 保证尽量原子，允许多进程竞争：赢者写入，其它进程看到存在就用
        """
        base, _ = os.path.splitext(pkl_path)
        npy_path = base + ".npy"
        if os.path.exists(npy_path):
            print('.npy exists')
            return npy_path

        # 生成到临时文件，再原子替换
        tmp_path = npy_path + f".tmp_{os.getpid()}.npy"

        try:
            with open(pkl_path, "rb") as f:
                names = pickle.load(f)  # list[str]，这里只在首次转换时短暂存在

            # 转为定长 bytes
            max_len = max(len(s.encode("utf-8")) for s in names) if len(names) > 0 else 1
            arr = np.asarray(names, dtype=f"S{max_len}")
            np.save(tmp_path, arr)

            # 原子替换：如果别人已经生成了，replace 也没关系（最后写入者覆盖，但内容一致）
            os.replace(tmp_path, npy_path)
        except FileNotFoundError:
            # 可能被其它进程抢先生成并replace了tmp；或者路径问题
            pass
        finally:
            # 清理残留 tmp（如果 replace 已经成功，tmp 不存在）
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        return npy_path

    def _lazy_init(self):
        if self._emb is None:
            self._emb = np.load(self.embeddings_file_path, mmap_mode="r")
        if self._names is None:
            self._names = np.load(self.images_name_npy_path, mmap_mode="r")
        if self._len is None:
            self._len = int(self._names.shape[0])

    def __getstate__(self):
        """
        关键：DataLoader(worker, spawn) 会 pickle Dataset。
        我们显式丢弃 memmap 句柄，确保序列化体积很小，且不会触发 >4GiB 的 pickle 限制。
        """
        d = self.__dict__.copy()
        d["_emb"] = None
        d["_names"] = None
        # _len 保留也无所谓；保留可避免每个 worker 再读 header
        return d

    def __len__(self):
        if self._len is None:
            # 只 mmap 一下 header，很轻
            names = np.load(self.images_name_npy_path, mmap_mode="r")
            self._len = int(names.shape[0])
        return self._len

    def __getitem__(self, index: int):
        self._lazy_init()

        rel = self._names[index].decode("utf-8")
        image_path = os.path.join(self.samples_root, rel)

        image = self.sample_loader(image_path)

        # 复制成普通 ndarray，避免 torch 直接包 memmap 只读视图引发潜在问题
        emb_np = np.array(self._emb[index], copy=True)
        embedding = torch.from_numpy(emb_np)

        if self.transform is not None:
            image = self.transform(image)

        return image, embedding
