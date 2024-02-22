import streamlit as st
from pathlib import Path
from PIL import Image
import os
import cv2
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pandas as pd
from PIL import Image
from models.modeling import VisionTransformer, CONFIGS
import torch
from torchvision import transforms

# streamlit 파일을 demo 폴더 안과 밖 모두 다 경로문제없이 실행하기 위해 path 추가
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir) 
sys.path.append(os.getcwd())

def get_project_root() -> str:
    """Returns project root path.
 
    Returns
    -------
    str
        Project root path.
    """
    return str(Path(os.path.abspath(__file__)).parent)

#모델 설정
# 이미지 변환을 위한 transform 설정
transform = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 설정 변수들
img_size = 448
pretrained_model_path = f"{os.path.dirname(get_project_root())}/output/best_model.bin"

# 모델 설정
config = CONFIGS["ViT-B_16"]
config.split = 'overlap'
config.slide_step = 12

num_classes = pd.read_csv(f'{os.path.dirname(get_project_root())}/label_encoding.csv')['label'].nunique()

# 모델 초기화 및 사전 학습된 가중치 로드
model = VisionTransformer(config, img_size, num_classes, zero_head=True)
model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu')['model'])

# 장치 설정 및 모델을 평가 모드로 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 레이블 딕셔너리 생성
label_dict = pd.read_csv(f'{os.path.dirname(get_project_root())}/label_encoding.csv').drop_duplicates('label').set_index('label').to_dict()['label_']

st.set_page_config(layout="wide", page_title="CCTV Car Image Classification")

st.write("## CCTV Car Image Classification")
st.write("CCTV영상속 차량 이미지를 알맞게 분류합니다.")
st.sidebar.write("## 이미지 업로드 :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

clear = False
def fix_image(upload):
    image = Image.open(upload)
    img = transform(image).unsqueeze(0).to(device)
    
    part_logits = model(img)
    probs = torch.nn.functional.softmax(part_logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=-1).item()
    predicted_name = label_dict[predicted_label]

    col1.write(f"이미지 추론 결과 : {predicted_name}")
    col1.image(image, use_column_width=True)
    

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else: 
    clear = True
    # Example 이미지를 위한 컬럼
    example_images = [F'{get_project_root()}/examples/C-220706_08_CR14_03_A0065.jpg', # 마을버스
                    F'{get_project_root()}/examples/C-220706_14_CR13_01_A0748.jpg', # BMW#1시리즈_F20(2012)
                    F'{get_project_root()}/examples/C-220824_08_CR02_01_A8560.jpg', # 현대#그랜저_XG(2002)
                    F'{get_project_root()}/examples/C-220903_10_CR02_01_A8981.jpg', # 랜드로버#디스커버리_5(2017)
                    F'{get_project_root()}/examples/C-220715_17_CR18_02_A0658.jpg']  # 지프#랭글러_JL(2018)
    example_answer = ["마을버스", "BMW#1시리즈_F20(2012)", "현대#그랜저_XG(2002)", "랜드로버#디스커버리_5(2017)", "지프#랭글러_JL(2018)"]
        

    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            # 예제 이미지를 위한 버튼 추가
            #st.write(f"샘플 이미지 {idx}")
            col.image(example_images[idx])
            if st.button(f"샘플 이미지 {idx+1} 테스트(클릭)", key=f"btn_{idx}"):
                # 선택된 이미지를 fix_image 함수로 전달
                example_image_path = example_images[idx]
                image = Image.open(example_image_path)
                col1.write(f"이미지 추론 결과: {example_answer[idx]}")
                col1.image(image, use_column_width=True)
            