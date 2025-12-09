import os
import io
import pickle

import torch.utils.data as data
import numpy as np
import lmdb
from PIL import Image


def num_samples(dataset, train):
    if dataset == "celeba":
        return 27000 if train else 3000
    elif dataset == "celeba64":
        return 162770 if train else 19867
    elif dataset == "imagenet-oord":
        return 1281147 if train else 50000
    elif dataset == "ffhq":
        return 63000 if train else 7000
    else:
        raise NotImplementedError("dataset %s is unknown" % dataset)


class LMDBDataset(data.Dataset):
    def __init__(self, root, name="", split="val", transform=None, is_encoded=False):
        self.name = name
        self.transform = transform
        self.split = split
        if self.split == "train":
            lmdb_path = os.path.join(root, "train.lmdb")
        elif self.split == "val":
            lmdb_path = os.path.join(root, "validation.lmdb")
        else:
            lmdb_path = os.path.join(f"{root}")
            
        self.lmdb_path = lmdb_path

        print(f"Loading dataset from {lmdb_path}")
        self.data_lmdb = lmdb.open(
            lmdb_path,
            readonly=True,
            subdir=os.path.isdir(lmdb_path),
            max_readers=1,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.is_encoded = is_encoded
        with self.data_lmdb.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys = pickle.loads(txn.get(b"__keys__"))
            
    def open_lmdb(self):
        self.env = lmdb.open(
            self.lmdb_path,
            subdir=os.path.isdir(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(write=False, buffers=True)
        self.length = pickle.loads(self.txn.get(b"__len__"))
        self.keys = pickle.loads(self.txn.get(b"__keys__"))

    def __getitem__(self, index):
        # target = 0
        # with self.data_lmdb.begin(write=False, buffers=True) as txn:
            
        #     key = f"{256}-{str(index).zfill(5)}".encode("utf-8")
        #     data = txn.get(key)
        #     img = Image.open(io.BytesIO(data))
        #     img = img.convert("RGB")
        
        if not hasattr(self, "txn"):
            self.open_lmdb()

        img, target = None, None
        byteflow = self.txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = io.BytesIO()
        # first element is image and second is path
        # see ImageFolderWithPaths
        buf.write(imgbuf[0])
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        return img, target, {"index": str(index).zfill(5) + ".png"}

    def __len__(self):
        if hasattr(self, "length"):
            return self.length
        else:
            with self.data_lmdb.begin() as txn:
                self.length = txn.stat()["entries"]
            return self.length


# class LMDBDataset(data.Dataset):
#     def __init__(self, root, name='', split='train', transform=None, is_encoded=False):
#         path = os.path.join(root, 'validation.lmdb')
#         self.env = lmdb.open(
#             path,
#             max_readers=32,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )

#         if not self.env:
#             raise IOError('Cannot open lmdb dataset', path)

#         with self.env.begin(write=False) as txn:
#             self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

#         self.resolution = 256
#         self.transform = transform
#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         with self.env.begin(write=False) as txn:
#             key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
#             img_bytes = txn.get(key)

#         buffer = io.BytesIO(img_bytes)
#         img = Image.open(buffer)
#         img = img.convert('RGB')
#         target = 0

#         return img, target, index
