from fastapi import FastAPI, File, UploadFile
import subprocess
import aiohttp
import os
import requests
import tempfile
import zipfile
import logging
import httpx
from fastapi.responses import FileResponse
from zipfile import ZipFile
from generate_fmnist import generate_fmnist
from google.cloud import storage
from fastapi import HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI()

server_ip = os.environ["SERVER_API_URL"]

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



class Item(BaseModel):
    zip: str


# 클라우드 zip 파일 URL 가져오기
@app.get("/get/url")
async def download_and_extract(item: Item):
    print(item.zip)

    zip_url = 'generated_images.zip'
    target_dir = 'dataset_from_server'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # HTTP GET 요청을 보내 파일 다운로드
    response = requests.get(zip_url)
    if response.status_code != 200:
        raise Exception("Can't download a file.")
    

    file_path = os.path.join(target_dir, 'generated_images.zip')
    # 파일 저장
    with open(file_path, "wb") as f:
        # f.write(response.content)
        f.write(zip_url)

    # 압축 해제
    with ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)


    return {"zip" : item}


#이미지 생성 & zip 파일로 저장
@app.get("/generate")
async def generate_images(zip_url : str):

    try:
        # Create the "images" folder if it doesn't exist
        os.makedirs("images", exist_ok=True)

        generate_fmnist()

        images_folder = "images"
        zip_filename = "generated_images.zip"

        with ZipFile(zip_filename, 'w') as zipf:
            for root, _, files in os.walk(images_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, images_folder))

    except Exception as e:
        return f"An error occurred: {str(e)}"
    
    return {"message": "done", "zip_filename": zip_filename}


    
@app.post("/send-zip-file")
async def send_zip_fun():
    zip_file_path = "generated_images.zip"
    endpoint = server_ip + "return/zipFile"  
    data = {
        "accuracy": [3.5, 5.8, 85.2, 45.3]
    }
    async with httpx.AsyncClient() as client:
        with open(zip_file_path, "rb") as file_data:
            response = await client.post(endpoint, files={"zipFile": (file_data)}, data = data)
    if response.status_code == 200:
        return {"message": "Zip 파일을 성공적으로 보냈습니다."}
    else:
        return {"message": "Zip 파일 전송에 실패했습니다."}

#

# class FileRequest(BaseModel):
#     zipFile: UploadFile

# @app.post("/send-zip-file")
# async def get_diary():
#     zip_file_path = "generated_images.zip"
#     endpoint = "http://172.20.10.9:8080/return/zipFile"  # 대상 서버 URL
#     logging.info("확인")
#     async with httpx.AsyncClient() as client:
#         with open(zip_file_path, "rb") as file_data:
#             print("파일 읽음")
#             # response = await client.get(endpoint, files={"zipFile": (file_data)})
#             print("3")

#             response = FileRequest(zipFile=file_data)
#     print("response 확인", response)
#     # Send the response to /model/inference/{diary_id} endpoint
#     requests.post(endpoint, response)



def generate_images():
    try:
        generate_fmnist()

        images_folder = "images"
        zip_filename = "generated_images.zip"

        with ZipFile(zip_filename, 'w') as zipf:
            for root, _, files in os.walk(images_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, images_folder))

    except Exception as e:
        return f"An error occurred: {str(e)}"
    
    return {"message": "done", "zip_filename": zip_filename}


# if __name__ == '__main__':
#     send_zip_fun()
    # generate_images()
    # 다운로드할 파일의 URL 및 로컬 디렉토리 설정
    # url = "https://storage.googleapis.com/gm-medical-project/gm-zip.zip-6b3e9cc1-e545-4e29-b47a-82a0f0c4fc69"
    # g_url = "https://drive.google.com/file/d/1Fw4ytuJbKBWsPuHAz3foTb1ZhTG702Sa/view?usp=drive_link"
    # target_dir = "images"
    # download_and_extract(g_url)
    # download_and_extract_zip(url, target_dir)
    # 파일 다운로드 및 압축 해제 함수 호출
    # download_and_extract(url, target_dir)




# async def download_and_extract_url(url: str):
#     target_dir = 'images'
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             if response.status != 200:
#                 raise HTTPException(status_code=404, detail="Can't download a file.")
            
#             temp_dir = tempfile.mkdtemp()
#             temp_file_path = os.path.join(temp_dir, "downloaded_file.zip")

#             with open(temp_file_path, "wb") as f:
#                 while True:
#                     chunk = await response.content.read(1024)
#                     if not chunk:
#                         break
#                     f.write(chunk)
#             with ZipFile(temp_file_path, "r") as zip_ref:
#                 zip_ref.extractall(target_dir)
#             os.remove(temp_file_path)
#             os.rmdir(temp_dir)








# def download_and_extract_zip(url, target_dir):

#     # 파일 다운로드
#     response = requests.get(url)
#     if response.status_code != 200:
#         raise Exception("Can't download a file.")
    
#     # 임시 디렉토리에서 작업
#     temp_dir = tempfile.mkdtemp()
#     temp_file_path = os.path.join(temp_dir, "downloaded_file.zip")

#     with open(temp_file_path, "wb") as f:
#         f.write(response.content)

#     # 만약 파일이 zip 파일이면 압축 해제
#     if temp_file_path.endswith(".zip"):
#         zip_dir = os.path.join(temp_dir, "extracted")
#         with zipfile.ZipFile(target_dir, "r") as zip_ref:
#             zip_ref.extractall(zip_dir)
    
#     # 임시 디렉토리 정리s
#     os.remove(temp_file_path)
#     os.rmdir(temp_dir)
