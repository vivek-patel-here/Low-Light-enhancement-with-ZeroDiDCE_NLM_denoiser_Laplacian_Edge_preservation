import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)
 
def lowlight(image_path):
    data_lowlight = Image.open(image_path)

    data_lowlight = (np.asarray(data_lowlight)/255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.to(device).unsqueeze(0)

    DiDCE_net = model.enhance_net_nopool().to(device)
    DiDCE_net.load_state_dict(torch.load('./Epoch100.pth', map_location=device))

    start = time.time()
    enhanced_image, A0 = DiDCE_net(data_lowlight)
    end_time = (time.time() - start)

    print("Inference time:", end_time)

    # result_path = image_path.replace('test_data','result_without_denoiser')
    result_path = image_path.replace('test_data','result_ours')

    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))

    torchvision.utils.save_image(enhanced_image, result_path)
if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = 'data/test_data/'
	
		file_list = os.listdir(filePath)

		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:
				# image = image
				print(image)
				lowlight(image)

		

