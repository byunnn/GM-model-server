import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class FashionMNIST(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        fashion_df = pd.read_csv('../data/fashionmnist/fashion-mnist_train.csv')
        self.labels = fashion_df.label.values #데이터프레임의 label열을 array 형식으로 반환
        self.images = fashion_df.iloc[:, 1:].values.astype('uint8').reshape(-1, 28, 28)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.fromarray(self.images[idx]) # NumPy 배열에서 PIL(Python Imaging Library) 이미지 객체를 생성
        
        if self.transform:
            img = self.transform(img)

        return img, label
    

def prepare_data():

    dataset = FashionMNIST()

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    dataset = FashionMNIST(transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    img, label = dataset.__getitem__(1)
    print(img.size(), label)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        
        #x.size(0) == batch size
        return out.view(x.size(0), 28, 28)
    
def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):

    prepare_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.MSELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

    g_optimizer.zero_grad()
    
    #noise 생성
    z = torch.randn(batch_size, 100, device=device)
    
    #fake labels, images 생성
    fake_labels = torch.randint(0, 10, (batch_size,), device=device)
    fake_images = generator(z, fake_labels)
    
    # D(fake images)
    validity = discriminator(fake_images, fake_labels)
    
    #Generator의 목표 : D(G(z)) = 1 이므로 loss(D(G(z)), 1)
    g_loss = criterion(validity, torch.ones(batch_size, device=device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # Train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, torch.ones(batch_size, device=device))

    # Train with fake images
    z = torch.randn(batch_size, 100, device=device)
    fake_labels = torch.randint(0, 10, (batch_size,), device=device)
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, torch.zeros(batch_size, device=device))

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def train():
    num_epochs = 30
    n_critic = 5
    display_step = 100

    for epoch in range(num_epochs):
        print('Starting epoch {}...'.format(epoch))
        
        for i, (images, labels) in enumerate(data_loader):
            real_images = images.to(device)
            labels = labels.to(device)
            
            generator.train()
            batch_size = real_images.size(0)
            
            d_loss = discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels)
            
            #generator보다 critic(discriminator)을 더 자주 학습
            if i % n_critic == 0:
                g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)
                
            
            if i % display_step == 0:
                print('Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}'.format(i, len(data_loader), d_loss, g_loss))
        
        generator.eval()
        print('Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}'.format(epoch, num_epochs, d_loss, g_loss))
        
        with torch.no_grad():
            z = torch.randn(9, 100, device=device)
            labels = torch.arange(9, device=device)
            sample_images = generator(z, labels).unsqueeze(1).data.cpu()
            grid = make_grid(sample_images, nrow=3, normalize=True).permute(1, 2, 0).numpy()
            
            plt.imshow(grid)
            plt.axis('off')
            plt.show()

        
def test():
    z = torch.randn(9, 100).to(device)
    labels = torch.LongTensor(np.arange(9)).to(device)
    sample_images = generator(z, labels).unsqueeze(1).data.cpu()

    plt.figure(figsize=(3,3))
    grid = make_grid(sample_images, nrow=3, normalize=True).permute(1,2,0).numpy()
    plt.imshow(grid)
    plt.show()


def main():
    print("hi")