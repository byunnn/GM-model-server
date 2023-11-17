import os
import argparse 

import torch 
import torchvision.utils as vutils
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import random 

from gmModel_DC.dcgan import Generator


def generate_dcgan(projectName):
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

    output_dir = 'gmModel_DC/outputs/'+ projectName+'/generated_images'
    os.makedirs(output_dir, exist_ok=True)

    # parser = argparse.ArgumentParser() 
    # parser.add_argument('-load_path', default='model/model_final.pth', help='Checkpoint to load path from')
    # parser.add_argument('-num_output', default=64, help='Number of generated outputs') 
    # args = parser.parse_args() 

    load_path = 'gmModel_DC/outputs/'+ projectName+ '/model/model_final.pth'
    num_output = 64

    state_dict = torch.load(load_path) 
    params = state_dict['params']
    netG = Generator(params).to(device) 
    netG.load_state_dict(state_dict['generator']) 

    noise = torch.randn(int(num_output), params['nz'], 1, 1, device=device)


    with torch.no_grad():
        generated_img = netG(noise).detach().cpu() 

    # multipple images
    for i in range(num_output):
        print(i, "번째 이미지 생성 중")
        image_path = os.path.join(output_dir, f'generated_image_{i}.png')
        # plt.imshow(np.transpose(generated_img[i], (1, 2, 0)))
        # plt.axis("off")
        plt.savefig(image_path, dpi=600)
        # plt.close()

    print(f"Generated images saved in {output_dir} directory.")

    # Single image
    # 여기서 생성된 이미지 대표 이미지로 보여주기
    image_path = os.path.join('gmModel_DC/outputs/'+ projectName, f'generated_single_image.png')
    # plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))
    plt.savefig(image_path, dpi=600)
    # plt.show()


# def calculate_fid_given_paths(paths, img_size=256, batch_size=50):
#     print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     inception = InceptionV3().eval().to(device)
#     loaders = [get_eval_loader(path, img_size, batch_size) for path in paths]

#     mu, cov = [], []
#     for loader in loaders:
#         actvs = []
#         for x in tqdm(loader, total=len(loader)):
#             actv = inception(x.to(device))
#             actvs.append(actv)
#         actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
#         mu.append(np.mean(actvs, axis=0))
#         cov.append(np.cov(actvs, rowvar=False))
#     fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
#     return fid_value