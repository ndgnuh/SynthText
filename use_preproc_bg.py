"""
Sample code for load the 8000 pre-processed background image data.
Before running, first download the files from:
  https://github.com/ankush-me/SynthText#pre-generated-dataset
"""

import h5py
import numpy as np
from PIL import Image
import os.path as osp
import pickle as pkl
from tqdm import tqdm
from matplotlib import pyplot as plt

im_dir = "bg_img"
depth_db = h5py.File("depth.h5", "r")
seg_db = h5py.File("seg.h5", "r")

imnames = sorted(depth_db.keys())

with open("imnames.cp", "rb") as f:
    filtered_imnames = set(pkl.load(f))

output_file = "renderer_data/sample.h5"
sample = h5py.File(output_file, "w")
sample.create_group("image")
sample.create_group("depth")
sample.create_group("seg")
pbar = tqdm(imnames)

count = 1
for imname in pbar:
    count += 1
    pbar.set_postfix({"name": imname})
    # ignore if not in filetered list:
    if (
        imname not in filtered_imnames
        or imname not in depth_db.keys()
        or imname not in seg_db["mask"].keys()
    ):
        continue

    if count == 10:
        break

    # get the colour image:
    img = Image.open(osp.join(im_dir, imname)).convert("RGB")

    # get depth:
    depth = depth_db[imname][:]

    # get segmentation info:
    seg = seg_db["mask"][imname]
    # area = seg_db["mask"][imname].attrs["area"]
    # label = seg_db["mask"][imname].attrs["label"]

    # re-size uniformly:
    # sz = depth.shape[:2][::-1]
    # img = np.array(img.resize(sz, Image.ANTIALIAS))
    # seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))

    sample["image"][imname] = img
    sample["depth"].create_dataset(imname, data=depth)
    sample["seg"].create_dataset(imname, data=seg)
    for k, v in seg.attrs.items():
        sample["seg"][imname].attrs.create(k, v)
    # seg.copy(seg_db[f"mask/{imname}"], sample[f"seg/{imname}"], "Dataset")
    # sample["seg"].create_dataset(imname, data=seg)
    # print(seg.attrs.keys())
    # print(sample["seg"][imname].attrs.keys())
    # area = sample["seg"][imname].attrs
    # label = sample["seg"][imname].attrs["label"]
    # see `gen.py` for how to use img, depth, seg, area, label for further processing.
    #    https://github.com/ankush-me/SynthText/blob/master/gen.py
