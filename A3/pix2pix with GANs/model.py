import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class G(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.e1 = nn.Conv2d(3, 64, 4, 2, 1)

        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 2, 1),
            nn.BatchNorm2d(64 * 2)
        )

        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1),
            nn.BatchNorm2d(64 * 4)
        )

        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1),
            nn.BatchNorm2d(64 * 8)
        )

        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, 64 * 8, 4, 2, 1),
            nn.BatchNorm2d(64 * 8)
        )

        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, 64 * 8, 4, 2, 1),
            nn.BatchNorm2d(64 * 8)
        )

        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, 64 * 8, 4, 2, 1),
            nn.BatchNorm2d(64 * 8)
        )

        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, 64 * 8, 4, 2, 1)
        )
        
        # Decoder
        self.d1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 8, 64 * 8, 4, 2, 1),
            nn.BatchNorm2d(64 * 8),
            nn.Dropout(0.5)
        )

        self.d2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 8 * 2, 64 * 8, 4, 2, 1),
            nn.BatchNorm2d(64 * 8),
            nn.Dropout(0.5)
        )

        self.d3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 8 * 2, 64 * 8, 4, 2, 1),
            nn.BatchNorm2d(64 * 8),
            nn.Dropout(0.5)
        )

        self.d4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 8 * 2, 64 * 8, 4, 2, 1),
            nn.BatchNorm2d(64 * 8)
        )

        self.d5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 8 * 2, 64 * 4, 4, 2, 1),
            nn.BatchNorm2d(64 * 4)
        )

        self.d6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 4 * 2, 64 * 2, 4, 2, 1),
            nn.BatchNorm2d(64 * 2)
        )

        self.d7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 2 * 2, 64, 4, 2, 1),
            nn.BatchNorm2d(64)
        )

        self.d8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 2, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        
        # Decoder
        d1 = self.d1(e8)
        d1 = torch.cat([d1, e7], dim=1)
        
        d2 = self.d2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        
        d3 = self.d3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        
        d4 = self.d4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        
        d5 = self.d5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        
        d6 = self.d6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        
        d7 = self.d7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        
        output = self.d8(d7)
        
        return output

    
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

        self.d0 = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.d1 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.d2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.d3 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.d4 = nn.Sequential(
             nn.Conv2d(512, 1, 4, 1, 1),
             nn.Sigmoid()
        )

    def forward(self, x):
        o0 = self.d0(x)
        o1 = self.d1(o0)
        o2 = self.d2(o1)
        o3 = self.d3(o2)
        output = self.d4(o3)

        return output
    