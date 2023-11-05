from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://192.168.232.1:3000",  # React 애플리케이션의 주소
    # 다른 허용할 도메인 주소들
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)