import torch
from torch import nn
from torch.nn.utils.spectral_norm import spectral_norm
from torchvision.models import resnet18
import torchvision.transforms.functional


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Extractor(nn.Module):
    def __init__(self, image_size: int = 224):
        super(Extractor, self).__init__()
        model = resnet18(pretrained=False)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.sigmoid = nn.Sigmoid()
        self.image_size = image_size

    def forward(self, x):
        x = torchvision.transforms.functional.resize(x, [self.image_size, self.image_size])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [64, 64, 56, 56]
        x = self.layer1(x)  # [64, 64, 56, 56]
        x = self.layer2(x)  # [64, 128, 28, 28]
        x = self.sigmoid(x)
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super(Classifier, self).__init__()
        model = resnet18(pretrained=False, num_classes=num_classes)
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self, num_classes: int, noise_dim: int, feature_num: int = 128):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(noise_dim + num_classes, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, feature_num * 1, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, z, y):
        y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
        zy = torch.cat([z, y], 1)
        return self.generator(zy)


class Discriminator(nn.Module):
    def __init__(self, num_classes: int, feature_num: int = 128, feature_size: int = 28):
        # extractor output size [128*28*28]
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.discriminator = nn.Sequential(
            spectral_norm(nn.Conv2d(feature_num + num_classes, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        self.apply(weights_init)
        self.num_classes = num_classes
        self.feature_size = feature_size

    def forward(self, f, y):
        y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
        y = y.expand(y.size(0), self.num_classes, self.feature_size, self.feature_size)
        fy = torch.cat([f, y], 1)
        return self.discriminator(fy).squeeze(-1).squeeze(-1)
