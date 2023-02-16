import torch
from torchvision import transforms
import time
from PIL import Image

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pyplot as plt
import cv2
import numpy as np

def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model

model = load_model()

def run_inference(url):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = cv2.imread(url) # shape: (480, 640, 3)
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (768, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 768, 960])
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 768, 960])
    start = time.time()
    image = image.to(device)
    with torch.no_grad(): # - при использовании gpu важно делать инференс именно через with torch.no_grad(), иначе будет забиваться память gpu
        output, _ = model(image) # torch.Size([1, 45900, 57])  
    end = time.time() - start
    print('end', end)
    return output, image

def visualize_output(output, image):
    output = non_max_suppression_kpt(output,
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(nimg)
    # plt.show()
    return nimg

start = time.time()
output, image = run_inference('people.jpg') # Bryan Reyes on Unsplash
img = visualize_output(output, image)
end_2 = time.time() - start
print('end_2', end_2)
img = Image.fromarray(img)
img.save('pic_2.jpeg')
