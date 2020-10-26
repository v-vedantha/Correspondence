# Third party

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import math
from torchvision import transforms
import model
import matplotlib.pyplot as plt
from torch.nn.functional import grid_sample
from PIL import Image
import cv2

# Training loop.
# For each frame in the image enter in frame - 5 and the frame and then do the losses to update the networks.
# If you have multiple losses where each layer has its opposite layer calculating the loss then you have maybe faster learning.
# A separate optimizer for each layer.

# And the losses themselves would just be applying grid sample and doing a simple loss function.

# Maybe the next steps would be adding in some more layers of losses.

# We also need the loss to not be destroyed by ignoring some of the edges. Maybe the trick is we only look at the loss of the inner frames so when something pops into view it has some
# time to get incorporated


correspondenceNet = model.get_pretrained_correspondence(False)
correspondenceNet.to(torch.device("cuda:0"))
optimizer = optim.Adam(correspondenceNet.parameters(), lr=3e-4, betas=(0.7, 0.999))
def loss(input, target, border=10):
    return torch.sum(torch.square(input[:,:,border:-border, border:-border] - target[:,:,border:-border, border:-border]))
cap = cv2.VideoCapture("IMG_0004.mp4")
# Stores the last k frames of the video
k=25
prev_k_frames = []


# As I understand, the grid sample method uses the x, y values of the outputs of the pixel. It may be easier for the model to not need to learn this
# so I am adding it in back at the end
shape = 223
single_vector_x = torch.linspace(-1, 1, steps = shape).reshape(1, -1)
single_vector_y = torch.linspace(-1, 1, steps = shape).reshape(-1, 1)
baseline = torch.zeros(1, shape, shape, 2)
baseline[:,:,:,0] = single_vector_x
baseline[:,:,:,1] = single_vector_y
baseline = baseline.to(torch.device("cuda:0"))
frame_count = 0
# Opens video and trains model
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    frame = Image.fromarray(frame)
    #frame = frame.convert('1')
    frame = frame.resize((shape, shape))
    
    frame = transforms.ToTensor()(frame)
    prev_k_frames.append(frame)
    frame_count += 1
    print("=========", frame_count, "===========")
    # Runs the optimizer as long as the frame stack is full
    if len(prev_k_frames) > k:
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Move frames to GPU
        first = torch.tensor(prev_k_frames[0]).reshape(-1, 3, shape, shape).to(torch.device("cuda:0"))
        last = torch.tensor(prev_k_frames[k-1]).reshape(-1, 3, shape, shape).to(torch.device("cuda:0"))
        
        
        # Calculate losses going forwards in time
        output_fwd = correspondenceNet(first, last)
        final_img_fwd = grid_sample(first, output_fwd + baseline)
        l_fwd = loss(final_img_fwd, last)

        # Calculate losses going backwards in time
        output_bck = correspondenceNet(last, first)
        final_img_bck = grid_sample(last, output_bck + baseline)
        l_back = loss(final_img_bck, first)
        
        # Save  some images for viewing later
        if frame_count > 3000:
            
            transforms.ToPILImage()(grid_sample(first.cpu(), baseline.cpu())[0]).save('qfirst.png')
            transforms.ToPILImage()(grid_sample(last.cpu(), baseline.cpu())[0]).save('qsecond.png')
            transforms.ToPILImage()(grid_sample(final_img_fwd.cpu(), baseline.cpu())[0]).save('qrfirst.png')
            transforms.ToPILImage()(grid_sample(final_img_bck.cpu(), baseline.cpu())[0]).save('qrsecond.png')
        
        # Sum losses
        total_loss = l_fwd + l_back
        print(total_loss.data)
        
        # Optimizer
        total_loss.backward()
        optimizer.step()
        
        # Removes the first image from queue so we can look at the next images
        prev_k_frames.pop(0)

torch.save(correspondenceNet.state_dict(), "saved_model.pt")




    
    
