# encoding=utf8
import cv2
import numpy as np 
import argparse
import pandas as pd
from PIL import Image
from models.modeling import VisionTransformer, CONFIGS
import torch
from torchvision import transforms

import os
os.environ["LC_CTYPE"] = 'C.UTF-8' 

# argparser 설정
parser = argparse.ArgumentParser(description='이미지 경로 설정')
parser.add_argument('--img_path', type=str, default='demo/examples/C-220720_13_CR15_02_A1619.jpg', help='') 
parser.add_argument('--best_model_dir_path', type=str, default=f'output/best_model.bin', help='')

# 인자 파싱
args = parser.parse_args()
img_path = args.img_path
pretrained_model_path = args.best_model_dir_path


# 이미지 변환을 위한 transform 설정
transform = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 설정 변수들
img_size = 448

# 모델 설정
config = CONFIGS["ViT-B_16"]
config.split = 'overlap'
config.slide_step = 12
num_classes = pd.read_csv('label_encoding.csv')['label'].nunique()

# 모델 초기화 및 사전 학습된 가중치 로드
model = VisionTransformer(config, img_size, num_classes, zero_head=True)
model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu')['model'])

# 장치 설정 및 모델을 평가 모드로 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() 

# 이미지 읽기 및 변환
img = cv2.imread(img_path)
img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
img.show()
img = transform(img).unsqueeze(0).to(device)

# 레이블 딕셔너리 생성
label_dict = pd.read_csv('label_encoding.csv').drop_duplicates('label').set_index('label').to_dict()['label_']

# 모델을 사용하여 예측 수행
part_logits = model(img)
probs = torch.nn.functional.softmax(part_logits, dim=-1)
predicted_label = torch.argmax(probs, dim=-1).item()
predicted_name = label_dict[predicted_label]

# 예측된 클래스 이름 출력
print("Predicted class name:", predicted_name)
