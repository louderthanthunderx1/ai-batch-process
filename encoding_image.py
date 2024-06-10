import numpy as np
import pandas as pd
import io, os, json
import cv2, pickle, datetime, faiss, psycopg2

from datetime import date
import time
from tqdm import tqdm

from obs import ObsClient
from obs import PutObjectHeader

from PIL import Image, ExifTags
import concurrent.futures

import torch
from torch import nn
from torchvision import transforms

import pickle
# ตัวนี้ระวัง Path ผิด
# ProductEncoder.network
from ProductEncoder.network import Network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("START initial")
print("base directory: {}".format(os.getcwd()))
# fixed random seed
seed = 1150

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

base_dir = os.getcwd()
encoder_model_path = '{}/models/acc97_class20up_augment_statedict.pth'.format(base_dir)
reference_image_folder_path = '{}/src/segment_flavor'.format(base_dir)
pickle_file_path = '{}/src/segment_flavor.pkl'.format(base_dir)
category_ignore_list = []

### ฟังก์ชั่นทั่วไป ###
# แปลง pillow -> torch tensor
def pil_to_tensor(image_pill, cfg):
    """
    Convert pill image into tensor and resize to fit the encoder model input shape.
    """
    image_size = (cfg.INPUT_SIZE, cfg.INPUT_SIZE)

    image_pill_resize = image_pill.resize(image_size).convert('RGB')
    image_numpy = np.array(image_pill_resize)
    image_tensor = transforms.ToTensor()(image_numpy)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

# ฟังก์ชั่นสำหรับโหลด encoder
def get_resnet_encoder(cfg):
    """
    Load pretrained resnet for encoding.
    ต้องใส่ cfg เข้ามาเป็น config ของ model
    """
    # สร้างตัวแปร model
    model = Network(cfg).cuda()

    # ปิดพวกค่าต่างๆที่เอาไว้สำหรับตอน train
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # โหลด Model จาก savepoint
    checkpoint = torch.load(cfg.RESUME_MODEL, map_location='cuda')
    model.load_model(cfg.RESUME_MODEL)

    return model

def get_transfer_model():
    # model = torch.load(encoder_model_path, map_location='cuda')
    model = Network(cfg).cuda()
    print(model)
    number_features = model.classifier.in_features
    print(number_features)
    # features = list(model.classifier.children())[:-1]  # Remove last layer
    # model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=2048, out_features=2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.25),
        nn.Linear(number_features, 124),
    )
    print(model)

    # model = torch.load(encoder_model_path).cuda()
    model.load_state_dict(torch.load(encoder_model_path))
    print(model)
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # ปรับโมเดลให้เป็น encoder
    # features = list(model.classifier.children())[:-1]  # Remove last layer
    model.classifier = nn.Identity() # remove classifier layers
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    print(model)
    
    model.eval()

    return model
  

# path ของ encoder และ prediction threshold
prediction_threshold = 0.8

# ประกาศ class เพื่อสร้าง object สำหรับ config encoder
# บางค่าไม่ได้มีผลอะไร แค่มันบังคับใส่เฉยๆไม่งั้นสร้าง object ไม่ได้
# ที่ต้อง config จริงๆ คือ encoder path, input size
class Args:
  RESUME_MODEL = encoder_model_path
  gpu_ids = '0'
  BACKBONE_FREEZE = False
  BASE_LR = 0.5
  WEIGHT_DECAY = 1e-4
  LR_SCHEDULER = 'multistep'
  BATCH_SIZEt = 512
  NUM_WORKERS = 16
  MAX_EPOCH = 120
  SAMPLER_TYPE = 'instance_balance'
  INPUT_SIZE = 112
  COLOR_SPACE = 'BGR'
  DATASET_ROOT = './dataset'
  DATASET_TRAIN_JSON = './dataset/AiProducts/converted_train.json'
  DATASET_VALID_JSON = './dataset/AiProducts/converted_val.json'
  DATASET_TEST_JSON = './dataset/AiProducts/converted_test.json'
  PRETRAINED_BACKBONE = ''
  SHOW_STEP = 500
  SAVE_STEP = 5
  num_classes = 50030
  LOSS_TYPE = 'CrossEntropy'
  OPTIMIZER = 'SGD'
  CLASSIFIER = 'FC'
  BACKBONE = 'resnest50'
  DECAY_STEP = [10,20,30]
  CHANGE_SAMPLER_EPOCH = 40
  p = 0

# สร้าง object config
cfg = Args()

# เรียกฟังก์ชั่นเพื่อ load encoder
print('Loading Encoder Model')
# encoder_model = get_resnet_encoder(cfg)
encoder_model = get_transfer_model()



### เริ่มส่วน โหลด reference vector ###

# ประกาศ list ของ แบรนด์ ที่เราจะ classify เก็บไว้ใน dict
reference_image_dict_image_ver = {}

print('Loading Reference Image Dictionary')
# Change this path if not run in Colab
reference_image_folder_list = os.listdir(reference_image_folder_path)
print(reference_image_folder_list)

# กรองและจัดเก็บภาพอ้างอิงใน dictionary
for folder in reference_image_folder_list:
  folder_name_split = folder.split('_')
  if len(folder_name_split) != 4 or folder_name_split[1] in category_ignore_list:
    continue
  reference_image_dict_image_ver[folder] = []

# โหลดและปรับขนาดภาพอ้างอิง
# ลูปนี้ read ภาพขึ้นมา แล้ว append ใส่ reference_image_dict_image_ver เก็บไว้
n_classes = len(reference_image_folder_list)
start_time = time.time()
for brand_folder in tqdm(reference_image_folder_list, desc=f'process preprocessing images in {n_classes} classes...'):
  folder_name_split = brand_folder.split('_')
  if len(folder_name_split) != 4 or folder_name_split[1] in category_ignore_list:
    continue
  brand_folder_path = os.path.join(reference_image_folder_path, brand_folder)
  brand_folder_list = os.listdir(brand_folder_path)
  # print(brand_folder)

  for brand_image_filename in brand_folder_list:

    if brand_image_filename == '.ipynb_checkpoints':
      continue
    brand_image_path = os.path.join(brand_folder_path, brand_image_filename)
    # print(brand_image_path)

    # read ภาพมาแล้ว resize
    try:
      brand_image = Image.open(brand_image_path)
    except:
      continue
    image_size = (cfg.INPUT_SIZE, cfg.INPUT_SIZE)

    image_pill_resize = brand_image.resize(image_size)
    # append เข้า list (ที่อยู่ใน dict)
    reference_image_dict_image_ver[brand_folder].append(image_pill_resize)

finished_time = time.time() - start_time
print(finished_time) 
# 20 minutes


# def process_image(brand_image_path, image_size):
#     try:
#         brand_image = Image.open(brand_image_path)
#         image_pill_resize = brand_image.resize(image_size)
#         return image_pill_resize
#     except:
#         return None

# def load_reference_images(reference_image_folder_path, cfg):
#     start_time = time.time()
    
#     reference_image_dict_image_ver = {}
#     reference_image_folder_list = os.listdir(reference_image_folder_path)
    
#     folder_filter_time = time.time()
#     for folder in reference_image_folder_list:
#         folder_name_split = folder.split('_')
#         if len(folder_name_split) != 4 or folder_name_split[1] in category_ignore_list:
#             continue
#         reference_image_dict_image_ver[folder] = []
#     folder_filter_end_time = time.time()

#     image_size = (cfg.INPUT_SIZE, cfg.INPUT_SIZE)

#     brand_folder_process_time = time.time()
#     for brand_folder in reference_image_folder_list:
#         folder_name_split = brand_folder.split('_')
#         if len(folder_name_split) != 4 or folder_name_split[1] in category_ignore_list:
#             continue
#         brand_folder_path = os.path.join(reference_image_folder_path, brand_folder)
#         brand_folder_list = os.listdir(brand_folder_path)

#         image_paths = [os.path.join(brand_folder_path, img) for img in brand_folder_list if img != '.ipynb_checkpoints']

#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             resized_images = list(executor.map(lambda img_path: process_image(img_path, image_size), image_paths))

#         # กรอง None ออก
#         resized_images = [img for img in resized_images if img is not None]
#         reference_image_dict_image_ver[brand_folder].extend(resized_images)
#     brand_folder_process_end_time = time.time()

#     total_time = brand_folder_process_end_time - start_time
#     folder_filter_time_taken = folder_filter_end_time - folder_filter_time
#     brand_folder_process_time_taken = brand_folder_process_end_time - brand_folder_process_time

#     print(f"Total time taken: {total_time:.2f} seconds")
#     print(f"Time taken to filter folders: {folder_filter_time_taken:.2f} seconds")
#     print(f"Time taken to process brand folders: {brand_folder_process_time_taken:.2f} seconds")

#     return reference_image_dict_image_ver
  
# # เรียกใช้ฟังก์ชัน
# reference_image_dict_image_ver = load_reference_images(reference_image_folder_path, cfg)

# เริ่ม vectorize image
print('Encoding Reference Image Dictionary')
pakage_brand_list = []
reference_image_list = []
reference_image_tensor_list = []

# Convert reference image dict into tensor list for FAISS.
# วนลูป read ภาพจากแต่ละ brand มา encode
start_time = time.time()
for pakage_brand, image_list in tqdm(reference_image_dict_image_ver.items(), desc="Process Vectorizing..."):

    for image in image_list:
        pakage_brand_list.append(pakage_brand)

        # reference_image_list.append(image)

        image_tensor = pil_to_tensor(image, cfg).cuda()

        encoder_model.eval()
        
        with torch.no_grad():
          image_encoded = np.array(encoder_model(image_tensor)[0].view(1, -1).cpu()).reshape(-1)

        reference_image_tensor_list.append(image_encoded)
vectorize_end_time = time.time()

# preprocess เพื่อนำ list of vector ที่ได้มาไปเข้า FAISS
pakage_brand_list = np.array(pakage_brand_list)
# reference_image_list = reference_image_list
reference_image_tensor_list = np.array(reference_image_tensor_list)

print('Fit Reference Image Dictionary into FAISS')
# ตั้ง 2048 เพราะ encoder เราใช้ 2048
faiss_start_time = time.time()
feature_dimension = 2048
index_flat = faiss.IndexFlatL2(feature_dimension)
index_flat.add(reference_image_tensor_list)
faiss_end_time = time.time()
print('Complete Load All Service')

total_time = vectorize_end_time - start_time
vectorize_time = vectorize_end_time - start_time
faiss_time = faiss_end_time - faiss_start_time
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Time taken to vectorize images: {vectorize_time:.2f} seconds")
print(f"Time taken to fit into FAISS: {faiss_time:.2f} seconds")

save_list = []
save_list.append(pakage_brand_list)
save_list.append(reference_image_tensor_list)
# open a file for writing
with open(pickle_file_path, 'wb') as f:
    # use pickle to dump the object to the file
    pickle.dump(save_list, f)
    
    
# check output dimession from model
# สร้าง input tensor ตัวอย่าง
# dummy_input = torch.randn(1, 3, cfg.INPUT_SIZE, cfg.INPUT_SIZE).cuda()  # [batch size 1, 3 channels, input_size, input_size]

# # ทำการ forward pass และดูขนาดของ output
# with torch.no_grad():
#     dummy_output = encoder_model(dummy_input)
# # print(dummy_output)
# if isinstance(dummy_output, tuple):
#     dummy_output = dummy_output[0]  # เลือกค่าแรกจาก tuple หากมีมากกว่าหนึ่งค่า
# print(f"Output shape: {dummy_output.shape}")