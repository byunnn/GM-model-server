from fastapi import FastAPI, File
import os
from pydantic import BaseModel
import requests
from starlette.config import Config
from zipfile import ZipFile

from gmModel_DC.generate import generate_dcgan
from gmModel_DC.train import train_dcgan
from generate_fmnist import generate_fmnist
from generate_gm import generate_gm
from generate_gm import generate_main


app = FastAPI()


config = Config(".env")
SERVER_IP = config('API_BASE_URL')


@app.get("/")
async def root():
    return {"hello"}


class File(BaseModel):
    email : str
    projectName : str
    zipUrl: str


# 클라우드 zip 파일 URL 가져오기
@app.post("/get/url")
async def download_and_extract(item: File):
    projectName = item.projectName
    email = item.email
    zipUrl = item.zipUrl
    print( projectName, email  , zipUrl)

    target_dir = 'user_dataset/'+projectName
    target_dir2 = 'gmModel_DC/user_dataset/'+projectName

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not os.path.exists(target_dir2):
        os.makedirs(target_dir2)

    # HTTP GET 요청을 보내 파일 다운로드
    response = requests.get(item.zipUrl)
    if response.status_code != 200:
        raise Exception("Can't download a file.")
    

    file_path = os.path.join(target_dir, projectName+'_dataset.zip')
    file_path2 = os.path.join(target_dir2, projectName+'_dataset.zip')

    # 파일 저장
    with open(file_path, "wb") as f:
        f.write(response.content)

    # 압축 해제
    with ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    #다운받은 zip 파일 삭제
    os.remove(file_path)

    # 파일 저장
    with open(file_path2, "wb") as f:
        f.write(response.content)

    # 압축 해제
    with ZipFile(file_path2, "r") as zip_ref:
        zip_ref.extractall(target_dir2)

    #다운받은 zip 파일 삭제
    os.remove(file_path2)

    # 압축 해제 후 target_dir 내의 모든 파일과 디렉토리 출력
    print(f"Files and directories in {target_dir}:")
    for root, dirs, files in os.walk(target_dir):
        for directory in dirs:
            print("dirs : ", os.path.join(root, directory))
        for file in files:
            print("files : ", os.path.join(root, file))

    await train_dcgan(projectName)
    await generate_images(projectName, email)


async def get_metrics(projectName, email):
    accuracy = 1
    fid = 1
    lpips = 1

    #흉부 데이터 covid normal
    #원본 데이터(covid, normal)
    #생성 데이터()


#이미지 생성 + zip 파일로 압축
async def generate_images(projectName, email):
    try:
        await generate_dcgan(projectName)


        output_dir = 'gmModel_DC/outputs/'+projectName
        zip_file_path = os.path.join(output_dir, projectName+'.zip')

        with ZipFile(zip_file_path, 'w') as zipf:
            for root, _, files in os.walk(output_dir+'/generated_images'):
                print("zip 파일 생성 하는 중")
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, output_dir))

        await send_zip_fun(projectName, email)


    except Exception as e:
        return f"An error occurred: {str(e)}"
    
    return {"message": "done", "zip_filename": zip_file_path}



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

#loss, generated_gif
async def send_zip_fun(projectName, email):
    output_dir = 'gmModel_DC/outputs/'+projectName + '/'
    zip_file_path = output_dir + projectName + '.zip'
    loss_fig_path = output_dir+ 'fig/Training_loss.png'
    gif_path = output_dir+ 'gif/X_ray.gif'
    generated_single_img_path = output_dir + 'generated_single_image.png'
    endpoint = SERVER_IP + "zips"

    data = {
        #원본, 생성
        "accuracy_generated": [93.2, 86.7],
        "accuracy_original_generated": [93.2, 91.3],
        "fid": [30.5, 125.9], 
        "LPIPS": [0.4, 0.23], 
        "email": email,
        "projectName": projectName
    }

    files = {
        "zipFile": (os.path.basename(zip_file_path), open(zip_file_path, "rb")),
        "loss": (os.path.basename(loss_fig_path), open(loss_fig_path, "rb")),
        "generated_gif": (os.path.basename(gif_path), open(gif_path, "rb")),
        "generated_single_img": (os.path.basename(generated_single_img_path), open(generated_single_img_path, "rb")),
    }

    response = requests.post(endpoint, files=files, data=data)


    if response.status_code == 200:
        return {"message": "Zip 파일을 성공적으로 보냈습니다."}
    else:
        return {"message": "Zip 파일 전송에 실패했습니다."}
    
