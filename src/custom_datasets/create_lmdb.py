import os
import sys
import os.path as osp

import logging
import click
import torch
import torch.utils
import torch.utils.data
import torchvision
import lmdb
import pickle
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def raw_reader(path):
    with open(path, "rb") as f:
        bin_data = f.read()
    return bin_data


def dump_pickle(obj):
    """
    Serialize an object.

    Returns :
        The pickled representation of the object obj as a bytes object
    """
    return pickle.dumps(obj)

def folder2lmdb(
    dpath, name="train_images", write_frequency=5000, num_workers=0
) -> str:
    """
    Converts a folder of images to lmdb dataset and returns the path to the lmdb dataset.
    """
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = torchvision.datasets.ImageFolder(directory, loader=raw_reader)
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generating LMDB to %s" % lmdb_path)
    map_size = 30737418240  # this should be adjusted based on OS/db size
    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=map_size,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    print(
        f"Length of dataset: {len(dataset)}; Length of dataloader: {len(data_loader)}"
    )

    txn = db.begin(write=True)
    for idx, (data, label) in enumerate(data_loader):
        image = data
        label = label.numpy()
        txn.put("{}".format(idx).encode("ascii"), dump_pickle((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", dump_pickle(keys))
        txn.put(b"__len__", dump_pickle(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

    return lmdb_path


@click.command()
@click.option("--dpath", required=True, help="Path to the folder containing images.")
@click.option(
    "--split", default="val", help="Dataset split (e.g., train, val, test)."
)
@click.option("--num-workers", type=int, default=4, help="Number of worker processes.")
def main(dpath, split, num_workers):
    try:
        logger.info(
            f"Starting conversion: {dpath} to lmdb with split '{split}' using {num_workers} workers."
        )
        folder2lmdb(dpath, split, num_workers)
        logger.info("Conversion completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
