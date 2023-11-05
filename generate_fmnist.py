import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
from sampleModel.sample_model import Generator
import websockets
import asyncio

# if __name__ == "__main__":
def generate_fmnist() :
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    generator_model_path = 'ckpt/generator.pth'
    # Load pretrained Generator Model
    generator = Generator().to(device) 
    generator.load_state_dict(torch.load(generator_model_path))
    generator.eval()

    z = torch.randn(9, 100).to(device)
    
    labels = torch.LongTensor(np.arange(9)).to(device)
    sample_images = generator(z, labels).unsqueeze(1).data.cpu()

    server_ip = "ws://192.168.170.110:8001/webSocketHandler"


    for i in range(sample_images.size(0)):
        image = sample_images[i].squeeze().numpy() 
        image = (image + 1) / 2.0 * 255.0  
        image = image.astype(np.uint8) 
        image = Image.fromarray(image, mode='L') 

        image_path = f'images/sample_image_{i}.png'
        image.save(image_path)

        print(f'{sample_images.size(0)} images saved.')

        asyncio.create_task(submitForm(sample_images.size(0), i))



async def submitForm(total, now):
    server_ip = "ws://192.168.170.110:8001/webSocketHandler"
    try:
        async with websockets.connect(server_ip) as websocket:
            await websocket.send(f"{total} 중 {now}번째 실행 중")
            websocket.close()
    except Exception as e:
        print(f"WebSocket 연결 중 오류 발생: {e}")




# async def connect_websocket(server_ip):
#     try:
#         websocket = await websockets.connect(server_ip)
#     except Exception as e:
#         print(f"WebSocket 연결 중 오류 발생: {e}")
#         return None
#     return websocket


# async def send_message(ws : websockets, total, now):
#     try:
#         async with ws as websocket:
#             await ws.send(str(total) + f' 중 {now}번째 실행 중')
#     except Exception as e:
#         print(f"메시지 전송 중 오류 발생: {e}")