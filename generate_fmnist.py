import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
from Model.sample_model import Generator


# if __name__ == "__main__":
def generate_fmnist() :
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    generator_model_path = 'experiments/generator.pth'

    # Load pretrained Generator Model
    generator = Generator().to(device) 
    generator.load_state_dict(torch.load(generator_model_path))
    generator.eval()

    z = torch.randn(9, 100).to(device)
    labels = torch.LongTensor(np.arange(9)).to(device)
    sample_images = generator(z, labels).unsqueeze(1).data.cpu()

    for i in range(sample_images.size(0)):
        image = sample_images[i].squeeze().numpy() 
        image = (image + 1) / 2.0 * 255.0  
        image = image.astype(np.uint8) 
        image = Image.fromarray(image, mode='L') 

        image_path = f'images/sample_image_{i}.png'
        image.save(image_path)

    print(f'{sample_images.size(0)} images saved.')