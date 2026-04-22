import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

gt_path = "result/GT"
pred_path = "result/output"

psnr_list = []
ssim_list = []

pred_files = sorted(os.listdir(pred_path))

for file in pred_files:

    name = os.path.splitext(file)[0]

    # search GT with any extension
    gt_file = None

    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        temp = os.path.join(gt_path, name + ext)
        if os.path.exists(temp):
            gt_file = temp
            break

    if gt_file is None:
        print("GT missing:", file)
        continue

    pred_file = os.path.join(pred_path, file)

    pred = cv2.imread(pred_file)
    gt = cv2.imread(gt_file)

    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    psnr = peak_signal_noise_ratio(gt, pred, data_range=255)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=255)

    psnr_list.append(psnr)
    ssim_list.append(ssim)

print("Processed:", len(psnr_list))
print("Average PSNR:", np.mean(psnr_list))
print("Average SSIM:", np.mean(ssim_list))