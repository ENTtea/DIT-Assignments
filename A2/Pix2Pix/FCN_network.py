import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        '''
        ### FILL: add more CONV Layers

        # conv-net
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7, 1, 3),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1, 1, 0),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.scores_fr = nn.Conv2d(4096, 3, 1, 1, 0)
        
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### Note: since last layer outputs RGB channels, may need specific activation function

        self.pool4 = nn.Conv2d(512, 3, 1, 1, 0)
        self.pool3 = nn.Conv2d(256, 3, 1, 1, 0)

        self.scores_upx2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.pool4_upx2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.pool3_upx8 = nn.ConvTranspose2d(3, 3, 16, 8, 4, bias=False)


    def forward(self, x):
        # Encoder forward pass
        
        # Decoder forward pass
        
        ### FILL: encoder-decoder forward pass

        # Encode
        x1 = self.conv1(x) #128
        x2 = self.conv2(x1) #64
        x3 = self.conv3(x2) #32
        x4 = self.conv4(x3) #16
        x5 = self.conv5(x4) #8
        x6 = self.fc6(x5)
        x7 = self.fc7(x6)

        scores = self.scores_fr(x7)

        # Decode

        scores_pool4 = self.pool4(x4)
        scores_pool3 = self.pool3(x3)
        
        scores_upx2 = self.scores_upx2(scores) #16
        
        scores_upx2_pool4 = self.pool4_upx2(scores_upx2 + scores_pool4) #32

        output = self.pool3_upx8(scores_upx2_pool4 + scores_pool3)

        output = nn.functional.tanh(output)
        
        return output
    