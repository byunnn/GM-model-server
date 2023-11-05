

import os
import torch


class CheckpointIO(object): # CheckpointIO 클래스 정의 시작
    def __init__(self, fname_template, data_parallel=False, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True) # 체크포인트 파일이 저장될 디렉토리를 생성합니다.
        self.fname_template = fname_template  # 체크포인트 파일의 이름 템플릿을 저장합니다.
        self.module_dict = kwargs # 체크포인트로 저장하거나 불러올 모듈들을 저장하는 딕셔너리입니다.
        self.data_parallel = data_parallel  # 데이터 병렬처리 여부를 저장합니다.

    def register(self, **kwargs):
        self.module_dict.update(kwargs) # 새로운 모듈을 등록합니다.

    def save(self, step):
        fname = self.fname_template.format(step)  # 체크포인트 파일의 이름을 결정합니다.
        print('Saving checkpoint into %s...' % fname)
        outdict = {} # 체크포인트 파일에 저장될 딕셔너리입니다.
        for name, module in self.module_dict.items(): # 데이터 병렬처리를 사용하는 경우, 모듈의 상태를 딕셔너리로 저장합니다.
            if self.data_parallel:
                outdict[name] = module.module.state_dict() # 데이터 병렬처리를 사용하지 않는 경우, 모듈의 상태를 딕셔너리로 저장합니다.
                        
            else:
                outdict[name] = module.state_dict()
                        
        torch.save(outdict, fname) # 딕셔너리를 파일로 저장합니다.

    def load(self, step):
        fname = self.fname_template.format(step) # 체크포인트 파일의 이름을 결정합니다.
        assert os.path.exists(fname), fname + ' does not exist!'  # 체크포인트 파일이 존재하는지 확인합니다.
        print('Loading checkpoint from %s...' % fname) # GPU를 사용하는 경우, 체크포인트 파일을 불러옵니다.
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu')) # GPU를 사용하지 않는 경우, 체크포인트 파일을 불러옵니다.
            
            
        for name, module in self.module_dict.items():
            if self.data_parallel:
                module.module.load_state_dict(module_dict[name]) # 데이터 병렬처리를 사용하는 경우, 모듈의 상태를 로드합니다.
            else:
                module.load_state_dict(module_dict[name]) # 데이터 병렬처리를 사용하지 않는 경우, 모듈의 상태를 로드합니다.
