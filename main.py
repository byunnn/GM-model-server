from fastapi import FastAPI, File, UploadFile
import aiohttp
import os
import requests
import zipfile
import httpx
from fastapi.responses import FileResponse
from zipfile import ZipFile
from generate_fmnist import generate_fmnist
from generate_gm import generate_gm
from generate_gm import generate_main
from fastapi import HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import argparse 
import asyncio

app = FastAPI()

# server_ip = os.environ["SERVER_API_URL"]
# server_ip = "http://172.20.10.9:8001/"
server_ip = "http://192.168.170.110:8001/"

@app.get("/")
async def root():
    return {"hello"}

# class Item(BaseModel):
#     zip: str

# # 클라우드 zip 파일 URL 가져오기
# @app.post("/get/url")
# async def download_and_extract(item: Item):
#     print(item.zip)
#     return {"zip": item.zip}



class File(BaseModel):
    email : str
    projectName : str
    zipUrl: str


# 클라우드 zip 파일 URL 가져오기
@app.post("/get/url")
async def download_and_extract(item: File):
    print( item.email, ' :' ,item.zipUrl)

    target_dir = 'user_dataset'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # HTTP GET 요청을 보내 파일 다운로드
    response = requests.get(item.zipUrl)
    if response.status_code != 200:
        raise Exception("Can't download a file.")
    

    file_path = os.path.join(target_dir, 'generated_images.zip')
    # 파일 저장
    with open(file_path, "wb") as f:
        f.write(response.content)
        # f.write(item.zip)

    # 압축 해제
    with ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    generate_images(item.projectName, item.email)



def generate_images(projectName, email):
    try:
        generate_fmnist()

        images_folder = "images"
        zip_filename = projectName+'.zip'
        # zip_filename = "generated_images.zip"

        with ZipFile(zip_filename, 'w') as zipf:
            for root, _, files in os.walk(images_folder):
                print("zip 파일 생성중")
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, images_folder))
        print("zip file 생성 완료")
        send_zip_fun(projectName, email)

    except Exception as e:
        return f"An error occurred: {str(e)}"
    
    return {"message": "done", "zip_filename": zip_filename}

def send_zip_fun(projectName, email):
    zip_file_path = projectName+'.zip'
    endpoint = server_ip + "zips"
    print(endpoint)
    data = {
        "accuracy": [3.5, 5.8, 85.2],
        "email": email,
        "projectName": projectName
    }
    print(data)
    with open(zip_file_path, "rb") as file_data:
        response = requests.post(endpoint, files={"zipFile": (file_data)}, data=data)
        print(response)
    
    if response.status_code == 200:
        return {"message": "Zip 파일을 성공적으로 보냈습니다."}
    else:
        return {"message": "Zip 파일 전송에 실패했습니다."}

# #생성된 zip파일 서버로 전송
# async def send_zip_fun(projectName, email):
#     zip_file_path = "generated_images.zip"
#     endpoint = server_ip + "zips"  
#     data = {
#         "accuracy": [3.5, 5.8, 85.2],
#         "email": email,
#         "projectName": projectName
#     }
#     async with httpx.AsyncClient() as client:
#         with open(zip_file_path, "rb") as file_data:
#             response = await client.post(endpoint, files={"zipFile": (file_data)}, data = data)
#     if response.status_code == 200:
#         return {"message": "Zip 파일을 성공적으로 보냈습니다."}
#     else:
#         return {"message": "Zip 파일 전송에 실패했습니다."}


# #생성된 zip파일 서버로 전송
# @app.post("/zips")
# async def send_zip_fun():
#     zip_file_path = "generated_images.zip"
#     endpoint = server_ip + "zips"  
#     data = {
#         "accuracy": [3.5, 5.8, 85.2],
#         "email" : "test@gmail.com",
#         "projectName" : "project 1"
#     }
#     async with httpx.AsyncClient() as client:
#         with open(zip_file_path, "rb") as file_data:
#             response = await client.post(endpoint, files={"zipFile": (file_data)}, data = data)
#     if response.status_code == 200:
#         return {"message": "Zip 파일을 성공적으로 보냈습니다."}
#     else:
#         return {"message": "Zip 파일 전송에 실패했습니다."}





# def sample_demo():


#     gm_args = {
#         "mode": "eval",
#         "num_domains": 3,
#         "w_hpf": 0,
#         "resume_iter": 5000,
#         "train_img_dir": "data/x-raymed/train",
#         "val_img_dir": "data/x-raymed/val",
#         "checkpoint_dir": "expr/checkpoints/x-raymed",
#         "eval_dir": "expr/eval/x-raymed"
#     }

#     # combined_args = {**gm_args, **args}

#     # print("///////////////////////////////////////1", args)
#     # print("///////////////////////////////////////", combined_args) 

#     generate_main(gm_args)


# #이미지 생성 & zip 파일로 저장
# @app.get("generate")
# async def generate_images(zip_url : str):

#     try:
#         # Create the "images" folder if it doesn't exist
#         os.makedirs("images", exist_ok=True)

#         gm_args = {
#             "mode": "eval",
#             "num_domains": 3,
#             "w_hpf": 0,
#             "resume_iter": 5000,
#             "train_img_dir": "data/x-raymed/train",
#             "val_img_dir": "data/x-raymed/val",
#             "checkpoint_dir": "expr/checkpoints/x-raymed",
#             "eval_dir": "expr/eval/x-raymed"
#         }

#         generate_gm(gm_args) 


#         images_folder = "images"
#         zip_filename = "generated_images.zip"

#         with ZipFile(zip_filename, 'w') as zipf:
#             for root, _, files in os.walk(images_folder):
#                 for file in files:
#                     file_path = os.path.join(root, file)
#                     zipf.write(file_path, os.path.relpath(file_path, images_folder))

#     except Exception as e:
#         return f"An error occurred: {str(e)}"
    
#     return {"message": "done", "zip_filename": zip_filename}

# if __name__ == '__main__':
#     sample_demo()
# #     send_zip_fun()
#     # generate_images()
#     # 다운로드할 파일의 URL 및 로컬 디렉토리 설정
#     # url = "https://storage.googleapis.com/gm-medical-project/gm-zip.zip-6b3e9cc1-e545-4e29-b47a-82a0f0c4fc69"
#     # g_url = "https://drive.google.com/file/d/1Fw4ytuJbKBWsPuHAz3foTb1ZhTG702Sa/view?usp=drive_link"
#     # target_dir = "images"
#     # download_and_extract(g_url)
#     # download_and_extract_zip(url, target_dir)
#     # 파일 다운로드 및 압축 해제 함수 호출
#     # download_and_extract(url, target_dir)











