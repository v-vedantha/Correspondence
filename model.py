# Third party
import torch
from torch import nn
import torch.nn.functional as F
import math

# I think a good baseline would be alexnet for the first part.
# And the reverse as a bunch of deconvolutions for the second part.


class Correspondence(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.to_vector = nn.Linear(256*6*6, 256*6*6)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1), # Maybe do something fancy with the max pooling layers. Or just ignore them
            nn.ConvTranspose2d(192, 192, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 64, kernel_size=5, padding=2),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 2, kernel_size=11, padding=2, stride=4, )
        )

    def forward(self, first, last):
        # Extracts alexnet style features from the image
        
        first_features = self.avgpool(self.features(first)).detach().clone()
        last_features = self.avgpool(self.features(last)).detach().clone()

        # Computes the difference in the two vector representations
        # The goal is to turn 0000100-100000 = 4. 
        # This means the ideal to_vector would be 1, 2, 3, 4, 5, 6, and so on so when you subtract them you get the above result.
        first_vector = self.to_vector(torch.flatten(first_features, 1))
        last_vector = self.to_vector(torch.flatten(last_features, 1))
        diff_vector = last_vector-first_vector
        

        # Turns the difference vector into a flow field. There can be another layer here, but it may not be needed. For now maybe not.
        diff = diff_vector.view(-1, 256, 6, 6)
        
        decoded = self.decode(diff)
        #print(decoded.shape)
        return decoded.transpose(1, 3).transpose(1, 2)

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def get_pretrained_correspondence(from_alexnet=False):
    if from_alexnet:
        alex = alexnet(True)
        correspondenceNet = Correspondence()
        correspondenceNet.features = alex.features
    else:
        print('dabadeeeeeeeee')
        correspondenceNet = Correspondence()
        correspondenceNet.load_state_dict(torch.load("saved_model.pt"))
    return correspondenceNet
