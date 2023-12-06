import datetime
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import websockets
from starlette.config import Config
from gmModel_DC.utils import get_data 
from gmModel_DC.dcgan import weights_init, Generator, Discriminator 

async def train_dcgan(projectName) :
    
    seed = 999
    random.seed(seed)
    torch.manual_seed(seed)
    output_dir = 'gmModel_DC/outputs/'+projectName

    subdirectories = ['model', 'fig', 'gif']
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(output_dir, subdirectory)
        if not os.path.exists(subdirectory_path):
            os.makedirs(subdirectory_path)
        

    params = {
        "bsize": 128,
        'imsize': 64,
        'nc': 3,
        'nz': 100,
        'ngf': 64,
        'ndf': 64,
        'nepochs': 2,
        'lr': 0.0002,
        'beta1': 0.5,
        'save_epoch': 10
    }

    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    dataloader = get_data(projectName, params)

    sample_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(
        sample_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(output_dir+'/fig/user_dataset.png', dpi=600)
    # plt.show()

    
    netG = Generator(params).to(device)
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator(params).to(device)
    netD.apply(weights_init)
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

    real_label = 1
    fake_label = 0

    optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

    img_list = []
    G_losses = []
    D_losses = []

    iters = 0

    print("Starting Training Loop...")

    # config = Config(".env")
    # IP = config('API_BASE_URL')
    # WEBSOCKET_API_URL = f"ws://{IP}/webSocketHandler"

    WEBSOCKET_API_URL = "ws://192.168.177.110:8001/webSocketHandler" 

    async with websockets.connect(WEBSOCKET_API_URL) as websocket:
        epoch_time = 0
        try:
            for epoch in range(params['nepochs']):
                epoch_start_time = datetime.datetime.now()

                message_to_send = f"{params['nepochs']}:{epoch+1}:{epoch_time}"
                await websocket.send(message_to_send)
                print(f"전송됨: {message_to_send}")

                for i, data in enumerate(dataloader, 0):

                    real_data = data[0].to(device)
                    b_size = real_data.size(0)
                    
                    netD.zero_grad()
                    label = torch.full((b_size, ), real_label, device=device)
                    output = netD(real_data).view(-1)
                    errD_real = criterion(output.to(torch.float32), label.to(torch.float32))
                    errD_real.backward()
                    D_x = output.mean().item()
                    
                    noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
                    fake_data = netG(noise)
                    label.fill_(fake_label)
                    output = netD(fake_data.detach()).view(-1)
                    errD_fake = criterion(output.to(torch.float32), label.to(torch.float32))
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()

                    errD = errD_real + errD_fake
                    optimizerD.step()
                    
                    netG.zero_grad()
                    label.fill_(real_label)
                    output = netD(fake_data).view(-1)
                    errG = criterion(output.to(torch.float32), label.to(torch.float32))
                    errG.backward()
                    D_G_z2 = output.mean().item()
                    optimizerG.step()

                    # if i % 50 == 0:
                    # print(torch.cuda.is_available())
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, params['nepochs'], i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                    G_losses.append(errG.item())
                    D_losses.append(errD.item())

                    if (iters % 100 == 0) or ((epoch == params['nepochs']-1) and (i == len(dataloader)-1)):
                        with torch.no_grad():
                            fake_data = netG(fixed_noise).detach().cpu()
                        img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))

                if epoch == 0:
                    torch.save({
                        'generator': netG.state_dict(),
                        'discriminator': netD.state_dict(),
                        'optimizerG': optimizerG.state_dict(),
                        'optimizerD': optimizerD.state_dict(),
                        'params': params
                    }, output_dir+'/model/model_epoch_{0}.pth'.format(epoch))

                # 각 epoch 종료 시간 기록
                epoch_end_time = datetime.datetime.now()

                one_epoch_time = epoch_end_time - epoch_start_time
                epoch_time_seconds = one_epoch_time.total_seconds()

                # Convert seconds to minutes and keep only two decimal places
                epoch_time = round(epoch_time_seconds / 60, 1)
                # epoch_time = one_epoch_time.total_seconds() / 60
                print(f"one_epoch_time: {one_epoch_time}")
                print(f"epoch_time: {epoch_time}")


        except websockets.ConnectionClosed:
            print("WebSocket 서버와의 연결이 닫혔습니다. 다시 연결 중...")

    from PIL import Image
    fig = plt.figure(figsize=(8,8))

    # for i, image in enumerate(img_list, start=1):
    #     print("i", i, image)
    #     output_filename = f"generate.{i}.png"
    #     # Rescale the image data to 0-255 range (assuming it's in the 0-1 range)
    #     image = (image.numpy() * 255).astype(np.uint8)
    #     # Reshape the image data if needed (e.g., (1, 1, 530) to (530,))
    #     image = image.squeeze()
    #     pil_image = Image.fromarray(image)
    #     pil_image.save(output_directory + output_filename)

    torch.save({
        'generator': netG.state_dict(),
        'discriminator': netD.state_dict(),
        'optimizerG': optimizerG.state_dict(),
        'optimizerD': optimizerD.state_dict(),
        'params': params
    }, output_dir + '/model/model_final.pth')


    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_dir+'/fig/Training_loss.png', dpi=600)
    # plt.show()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated images per step")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    # plt.show()
    anim.save(output_dir+'/gif/X_ray.gif', dpi=80, writer='imagemagick')

                
# if __name__ == '__main__':
#     train_dcgan()