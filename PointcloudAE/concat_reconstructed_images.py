import os
import os.path as osp

import cv2
from tqdm import tqdm


ROOT_DIR = "render"
OUT_DIR = osp.join(ROOT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
list_models = [
    "raw",
    "swd",
]

NUM_SAMPLES = 50
SELECTED_SAMPLES = 50
DATASET_TYPE = "shapenetcore55"
HEIGHT = 300
WIDTH = 400
# TOP, BOTTOM = int(HEIGHT * 0.15), int(HEIGHT * 0.95)
# LEFT, RIGHT = int(WIDTH * 0.2), int(WIDTH * (1 - 0.2))
TOP, BOTTOM = int(HEIGHT * 0.15), int(HEIGHT * 0.95)
LEFT, RIGHT = int(WIDTH * 0.1), int(WIDTH * 0.9)

for idx in tqdm(range(SELECTED_SAMPLES)):
    list_images = []
    image_name = "reconstruct_random_{}_{}.npy_{:02d}.jpg".format(NUM_SAMPLES, DATASET_TYPE, idx)

    for model_idx, model in enumerate(list_models):
        image_path = osp.join(ROOT_DIR, "images", model, image_name)
        image = cv2.imread(image_path)
        image = image[TOP:BOTTOM, LEFT:RIGHT, :]
        list_images.append(image)

    out_image = cv2.vconcat(list_images)
    out_path = osp.join(OUT_DIR, image_name)
    cv2.imwrite(out_path, out_image)

list_images = []
list_idx = [0, 3, 7, 8, 34, 42]

for idx in list_idx:
    image_name = "reconstruct_random_{}_{}.npy_{:02d}.jpg".format(NUM_SAMPLES, DATASET_TYPE, idx)
    image_path = osp.join(OUT_DIR, image_name)
    image = cv2.imread(image_path)
    list_images.append(image)

out_image = cv2.hconcat(list_images)
os.makedirs(osp.join(ROOT_DIR, "demo"), exist_ok=True)
out_path = osp.join(ROOT_DIR, f"demo/reconstruction_short_{DATASET_TYPE}.jpg")
cv2.imwrite(out_path, out_image)
