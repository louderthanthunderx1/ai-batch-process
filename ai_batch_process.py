### QUERY SCRIPT'S ###
import pandas as pd
from sqlalchemy import text, create_engine
from urllib.parse import quote
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")


### create database engine
def create_engine_dataverse():
    dbschema = 'dataverse'
    engine = create_engine(
        'postgresql+psycopg2://service_mystery:%s@43.225.141.184:5432/Mystery' % quote('Tspace@2021'),
        connect_args={'options': '-csearch_path={}'.format(dbschema)})
    return engine

### create dataframe from query
def create_dataframe_from_query(engine, start_date, end_date):
    # initialize dataframe
    df = pd.DataFrame()
    
    ### select view raw data log
    # query = f"""
    # select * from dataverse.runtime_raw_result_log
    # where create_datetime between '{start_date}' and '{end_date}'
    # """
    
    ### select photo_id for run batch
    select_query = f"""
                    select distinct fs2.photo_id 
                    from warehouse_prod.fact_stock fs2 
                    join warehouse_prod.fact_task ft 
                        on ft.task_id = fs2.task_id 
                    join warehouse_prod.fact_photo fp 
                        on fp.photo_id = fs2.photo_id 
                    join warehouse_prod.dim_sku_ai dsa 
                        on fs2.sku_id = dsa.sku_id 
                    where ft.task_status = 'สำเร็จ'
                        and ft.update_datetime between '{start_date}' and '{end_date}'
                        and fs2."source" = 'stock_photo'
                        and fs2.stock_location = 'main_shelf'
                """
    try:
        df = pd.DataFrame(engine.connect().execute(text(select_query)))
    except Exception as e:
        print(f"Error executing query: {str(e)}")
    return df

engine = create_engine_dataverse()

### select data from datetime
def select_data(engine):
    start_date = '2024-05-01' # change date
    end_date = '2024-05-31' # change date
    print(f"Select data between date {start_date} to {end_date}")
    
    df =  create_dataframe_from_query(engine=engine,
                                    start_date=start_date,
                                    end_date=end_date)
    return df

df = select_data(engine=engine)
df.head()


table_name = 'ai_performance_bot_csd_raw_result_log'
schema_name = 'dataverse'
# Insert the DataFrame into the SQL table
count_df.to_sql(table_name, engine, schema=schema_name, if_exists='append', index=False)




import io, os, json
import time
import uuid
import threading
import pika
import pickle
import base64
import gc
from io import BytesIO

import cv2
from PIL.ExifTags import GPSTAGS, TAGS
from PIL import Image, ExifTags
from IPython.display import display

import torch
from torch import nn
from torchvision import transforms

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from tqdm.notebook import tqdm

import faiss
from obs import ObsClient, PutObjectHeader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

# ตัวนี้ระวัง Path ผิด
# ProductEncoder.network
from ProductEncoder.network import Network

from ai_model import get_resnet_encoder, get_transfer_model, get_resnet_trasforms, load_bot_beer_model, load_can_beer_model, load_bot_csd_model, load_bot_tea_model, load_bot_water_model
from ai_utils import sku_detection_faiss, get_prediction_df_faiss
from ai_helper import get_floor_level, show_result_qdvp # ,upload_result_image, convert_to_degrees

# Database Config
DB_HOST = os.environ.get("DB_HOST", "43.225.141.184")
DB_PORT = "5432"
DB_USERNAME = "service_mystery"
DB_PASSWORD = "Tspace@2021"
DB_NAME = "Mystery"

LOG_TABLE = "dataverse.ai_saleint_result_log_dev"
RAW_RESULT_LOG_TABLE = "dataverse.ai_saleint_raw_result_log_dev"
IMAGE_DETAIL_LOG_TABLE = "dataverse.ai_saleint_image_detail_log_dev"
BEER_LOG = 'dataverse.ai_performance_beer_size_log'
ENERGY_LOG = 'dataverse.ai_performance_enegydrink_log'
BOT_CSD_LOG = 'dataverse.ai_performance_bot_csd_log'
BOT_CSD_RAW_RESULT_LOG = 'dataverse.ai_performance_bot_csd_raw_result_log'


# OBS Config
AK = '5IXPDAUIKYP9IBMVVAP2'
SK = 'OoyfSoov4qjNQxTk6aclwvjqesOO5Z2WPzZTn1s7'
server = 'https://obs.ap-southeast-2.myhuaweicloud.com'
bucket_name = 'mystery-shopping'
obs_client = ObsClient(access_key_id = AK, secret_access_key = SK, server = server)
#UTC config
utc_shift = timedelta(hours=7)

######################

# RABBITMQ PARAMETER #

######################

MQ_HOST = os.environ.get("MQ_HOST", "110.238.115.174")
MQ_PORT = os.environ.get("MQ_PORT", "5672")
MQ_USER = os.environ.get("MQ_USER", "tspace")
MQ_PASS = os.environ.get("MQ_PASS", "TSpace@cw2022")
MQ_V_HOST = os.environ.get("MQ_V_HOST", "/")
QUEUE_AI_RESULT = os.environ.get("QUEUE_AI_RESULT", "ai-product.image")

n_neighbor = 5
category_ignore_list = []

# list สำหรับกรองผลเฉพาะ category ที่อยู่ในตู้แช่ (refig)
# shelf_category_list = ["Gin", "Rum", "Vodka", "Whisky", "Brandy", "Tequila"]
# list สำหรับ category ที่อยู่ได้ทั้ง ตู้แช่ (refig) และ ชั้นวางเหล้า (shelf)
# special_category_list = ["White Spirits", "White Spirits Clear Bottle", "White Spirits Green Bot", "Herbal Spirits", 'Energy Drink']
sale_int_category_list = ['Beer']

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
detector_model_path = '{}/models/best 40 epoch march.pt'.format(base_dir)
reference_image_folder_path = '{}/models/segment_folder/segment_flavor'.format(base_dir)
pickle_file_path = '{}/models/segment_flavor.pkl'.format(base_dir)
bot_beer_size_model_path = '{}/models/size_model/bot_beer_resnet50_model09.pth'.format(base_dir)
can_beer_size_model_path = '{}/models/size_model/can_beer_resnet50_model09.pth'.format(base_dir)
bot_csd_size_model_path = '{}/models/size_model/bot_csd_9_classes_model09.pth'.format(base_dir)
bot_tea_size_model_path = '{}/models/size_model/bot_tea_4_classes_model05.pth'.format(base_dir)
bot_water_size_model_path = '{}/models/size_model/bot_water_3_classes_model06.pth'.format(base_dir)


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
encoder_model = get_transfer_model(device=device, encoder_model_path=encoder_model_path, cfg=cfg)

### เริ่มส่วน โหลด reference vector ###
# อ่านไฟล์ ref img vector ที่ถูก encode ไว้แล้ว
# ถ้าต้องการให้ตัว model encode ใหม่ หลังจาก uncomment ส่วนข้างบนแล้ว ให้ comment ส่วนด้านล่างนี้ ตั้งแต่ Start Read จนถึง Finish Read

# Start Read Pickle file
# Open the file for reading
with open(pickle_file_path, 'rb') as f:
    # use pickle to load the object from the file
    my_large_object = pickle.load(f)
pakage_brand_list, reference_image_tensor_list = my_large_object
# Finish Read Pickle file
print('Fit Reference Image Dictionary into FAISS')

# ตั้ง 2048 เพราะ encoder เราใช้ 2048
feature_dimension = 2048
# index_flat = faiss.IndexFlatL2(feature_dimension)
# index_flat.add(reference_image_tensor_list)

# faiss gpu
gpu_resource = faiss.StandardGpuResources()
# gpu_resource

index_flat = faiss.IndexFlatL2(feature_dimension)
# index_flat.add(reference_image_tensor_list)
nlist = 750 #จำนวน centroids
m = 8
index_ivf = faiss.IndexIVFFlat(index_flat, feature_dimension, nlist, faiss.METRIC_L2)
# index_ivf = faiss.IndexIVFPQ(index_flat, feature_dimension, nlist, m, 8)
index_ivf = faiss.index_cpu_to_gpu(gpu_resource, 0, index_ivf)
assert not index_ivf.is_trained
index_ivf.train(reference_image_tensor_list)
assert index_ivf.is_trained
index_ivf.add(reference_image_tensor_list)
index_ivf.nprobe = 10  # ตั้งค่า nprobe


# ถ้าให้ model encode เอง โดยไม่อ่านไฟล์จาก pickle ให้ uncomment 2 บรรทัดข้างล่างนี้ด้วย เพื่อประหยัดพื้นที่ RAM เพราะตัวแปรไม่ได้ใช้แล้วหลังจาก encode
# del reference_image_dict_image_ver
# del reference_image_tensor_list
### จบส่วน โหลด reference vector ###

### เริ่มส่วน โหลด detector ###
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path = detector_model_path, force_reload = True, device = torch.cuda.current_device())

# ค่า confidence
yolov5_model.conf = 0.6
# จำนวน object ที่จะให้ detect (ไม่เกิน)
yolov5_model.max_det = 35

# load size model
bot_beer_model = load_bot_beer_model(model_path=bot_beer_size_model_path, device=device)
can_beer_model = load_can_beer_model(model_path=can_beer_size_model_path, device=device)
bot_csd_model = load_bot_csd_model(model_path=bot_csd_size_model_path, device=device)
bot_tea_model = load_bot_tea_model(model_path=bot_tea_size_model_path, device=device)
bot_water_model = load_bot_water_model(model_path=bot_water_size_model_path, device=device)


print('Complete Load All Service')
gc.collect()


### OBS SCRIPT'S & IMAGE OPERATION'S ###
def obs_ops():
    AK = 'AEFBJRJ1MOY1SWVLOULZ'
    SK = 'BwRN5g6fSkUusFCWwDS7WRASBOdZy5XYYoldChMt'
    SERVER = 'https://obs.ap-southeast-2.myhuaweicloud.com'
    obs_client = ObsClient(access_key_id = AK, secret_access_key = SK, server = SERVER)
    return obs_client

def download_image_to_bytes(download_path, obs_client, bucket_name):
    try:
        resp = obs_client.getObject(bucket_name, download_path, loadStreamInMemory=True)
        if resp.status < 300 and resp.body:
            return BytesIO(resp.body.buffer)
        else:
            print('Error: Unable to download image from OBS')
            return None
    except Exception as e:
        print(f'Error downloading image from OBS: {e}')
        return None

# def process_image(image_bytes_io):
#     print(image_bytes_io)
#     print(type(image_bytes_io))
#     try:
#         image_pil = Image.open(image_bytes_io)
#         display(image_pil)
#         tmp_pil = image_pil.copy()
        
#         for orientation in ExifTags.TAGS.keys():
#             if ExifTags.TAGS[orientation] == 'Orientation':
#                 break

#         exif = dict(image_pil._getexif().items())
#         try:
#             if exif.get(orientation) == 3:
#                 tmp_pil = tmp_pil.rotate(180, expand=True)
#             elif exif.get(orientation) == 6:
#                 tmp_pil = tmp_pil.rotate(270, expand=True)
#             elif exif.get(orientation) == 8:
#                 tmp_pil = tmp_pil.rotate(90, expand=True)
#         except:
#             print('No EXIF')
#         # image_np = np.array(image_pil)

#         # return image_np
#         return image_pil, tmp_pil
    
#     except Exception as e:
#         print(f'Error processing image: {e}')
#         return None, None
def process_image(image_bytes_io):
    # print(image_bytes_io)
    # print(type(image_bytes_io))
    try:
        image_pil = Image.open(image_bytes_io)
        tmp_pil = image_pil.copy()

        exif = None
        if hasattr(image_pil, '_getexif'):  # Check if _getexif method exists
            exif = image_pil._getexif()

        if exif is not None:  # Check if EXIF data is present
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break

            exif_dict = dict(exif.items())
            try:
                if exif_dict.get(orientation) == 3:
                    tmp_pil = tmp_pil.rotate(180, expand=True)
                elif exif_dict.get(orientation) == 6:
                    tmp_pil = tmp_pil.rotate(270, expand=True)
                elif exif_dict.get(orientation) == 8:
                    tmp_pil = tmp_pil.rotate(90, expand=True)
            except:
                print('Error processing EXIF orientation')
        else:
            print('No EXIF data')

        return image_pil, tmp_pil

    except Exception as e:
        print(f'Error processing image: {e}')
        return None, None
    
def get_image_array(photo_id):

    bucket_image = 'prod'
    image_filename = photo_id + '.jpg'
    image_bucket_path = f'{bucket_image}/{image_filename}'
    # print(image_bucket_path)
    bucket_name = 'mystery-shopping'
    image_bytes_io = download_image_to_bytes(image_bucket_path, obs_ops(), bucket_name)

    if image_bytes_io:
        image_pil, tmp_pil = process_image(image_bytes_io)
        return image_pil, tmp_pil
    else:
        return None, None
    
    
def get_photo_data(filter_df):
    photo_id = filter_df['photo_id'].values.any()
    # print(photo_id)
    # print(filter_df['photo_datetime'].values.any())
    # photo_datetime = filter_df['create_datetime'].values.any()
    # photo_datetime = filter_df['photo_datetime'].dt.strftime('%Y-%m-%d').values.any()
    # print(photo_datetime)
    # return photo_id, photo_datetime
    return photo_id

# Function to safely get the first element from a DataFrame column
def get_first_element(df, column_name):
    if not df.empty and column_name in df.columns and len(df[column_name].values) > 0:
        return str(df[column_name].values[0])
    else:
        return ""

def upload_result_image(ref_id, base64, MQ_USER, MQ_PASS, MQ_HOST, MQ_PORT, MQ_V_HOST, QUEUE_AI_RESULT):
    credentials = pika.PlainCredentials(MQ_USER, MQ_PASS)
    parameters = pika.ConnectionParameters(host=MQ_HOST, port=MQ_PORT, virtual_host=MQ_V_HOST, credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue=QUEUE_AI_RESULT)
    # channel.basic_publish(routing_key=QUEUE_AI_RESULT, exchange='', properties=pika.BasicProperties(headers={"ref_id": ref_id}),
                        #   body=base64)
    channel.basic_publish(routing_key=QUEUE_AI_RESULT, exchange='', properties=pika.BasicProperties(headers={"ref_id": ref_id, "path": "run_batch_photo_ai_20240405"}), # แก้ path ที่เก็บรูปตรงนี้
                          body=base64)                   
    connection.close()
    

def main():
    
    
    # start processing
    print(f"[INFO] start processing")
    data_df = select_data()
    data_df


    # get unique photo_id
    data_ids = data_df.photo_id.unique()
    # len(data_ids)
    print(f"[INFO] Total Image: {len(data_ids)}")

    # data_ids
    times = []
    i = 0
    for data_id in tqdm(data_ids[:1]):
        iteration_start_time = time.time()
        # for data_id in tqdm(data_ids[:300]):
        print(f"current photo_id: {data_id}")
        # get filter data from unique ids
        filter_df = data_df[data_df['photo_id'] == data_id].copy()
        print(filter_df)
        
        
        # photo_id, photo_datetime = get_photo_data(filter_df)
        photo_id = get_photo_data(filter_df)
        # print(photo_id)
        # print(photo_datetime)
        
        image_pil, tmp_pil = get_image_array(photo_id)
        display(image_pil)
        # display(tmp_pil)
        
        # connect database
        PGSQL = PostgreSQLOperator(DB_HOST, DB_PORT, DB_NAME, DB_USERNAME, DB_PASSWORD)
        
        # นำรูปภาพเข้าไปทำ prediction
        yolo_predictions_df, pill_crop_image_list, knn_list_label, knn_list_index, knn_list_distance = get_prediction_df_faiss(inference_image_path=image_pil, 
                                                                                                                        rotate_pil=tmp_pil, 
                                                                                                                        is_loaded=True,
                                                                                                                        yolov5_model=yolov5_model,
                                                                                                                        encoder_model=encoder_model,
                                                                                                                        cfg=cfg,
                                                                                                                        index_flat=index_ivf,
                                                                                                                        pakage_brand_list=pakage_brand_list)
        
        yolo_predictions_df
        
        if len(yolo_predictions_df) != 0:

            # เตรียมข้อมูลสำหรับปั้นผล ก่อนส่งไปให้ทาง survey app
            yolo_predictions_df['Floor'] = 0
            yolo_predictions_df['prediction_label'] = knn_list_label
            yolo_predictions_df['prediction_label'] = yolo_predictions_df['prediction_label'].apply(lambda x : ",".join(map(str,x)))
            yolo_predictions_df['prediction_dist'] = knn_list_distance
            yolo_predictions_df['prediction_dist'] = yolo_predictions_df['prediction_dist'].apply(lambda x : ",".join(map(str,x)))
            # yolo_predictions_df
            yolo_predictions_df['size'] = None
            
            ### start doing beer size ###
            filter_can_beer_df = yolo_predictions_df[yolo_predictions_df['SKU_Name'].str.startswith('Can_Beer')]
            
            if len(filter_can_beer_df) != 0:
                for index, rows in filter_can_beer_df.iterrows():
                    
                    image = pill_crop_image_list[index]
                    
                    transform_crop_image = get_resnet_trasforms(image, device=device)  # Unsqueeze to add a batch dimension

                    # Get model predictions
                    with torch.no_grad():
                        outputs = can_beer_model(transform_crop_image)
                        _, predicted = torch.max(outputs, 1)

                    # Print the prediction
                    class_names = ['S', 'L']  # Class names
                    predicted_class = class_names[predicted[0].cpu().numpy()]
                    # print(f"Index: {index}, Predicted class: {predicted_class}")
                    filter_can_beer_df.loc[index, 'size'] = predicted_class
                    
                yolo_predictions_df[yolo_predictions_df['SKU_Name'].str.startswith('Can_Beer')] = filter_can_beer_df
            else:
                print('No can beer')
                
            filter_bot_beer_df = yolo_predictions_df[yolo_predictions_df['SKU_Name'].str.startswith('Bot_Beer')]
            if len(filter_bot_beer_df) != 0:
                for index, rows in filter_bot_beer_df.iterrows():
                    image = pill_crop_image_list[index]
                    # Apply the transforms
                    transform_crop_image = get_resnet_trasforms(image, device=device)  # Unsqueeze to add a batch dimension

                    # Get model predictions
                    with torch.no_grad():
                        outputs = bot_beer_model(transform_crop_image)
                        _, predicted = torch.max(outputs, 1)

                    # Print the prediction
                    class_names = ['S', 'L']  # Class names
                    predicted_class = class_names[predicted[0].cpu().numpy()]
                    filter_bot_beer_df.loc[index, 'size'] = predicted_class
                    
                yolo_predictions_df[yolo_predictions_df['SKU_Name'].str.startswith('Bot_Beer')] = filter_bot_beer_df
                
            else:
                print('No bot beer')
                
            # display(yolo_predictions_df)    
            filter_bot_csd_df = yolo_predictions_df[(yolo_predictions_df['SKU_Name'].str.startswith('Bot_CSD')) | (yolo_predictions_df['SKU_Name'].str.startswith('Bot_CSD COLA'))]
            # display(filter_bot_csd_df)
            # display(filter_bot_beer_df)
            
            if len(filter_bot_csd_df) != 0:
                # bot_beer_threshold = 0.94
                for index, rows in filter_bot_csd_df.iterrows():
                    # print(rows['index'])
                    # print(index)
                    # idx = int(rows['index'])
                    # idx = index
                    image = pill_crop_image_list[index]
                    # display(image)
                    # Apply the transforms
                    transform_crop_image = get_resnet_trasforms(image, device=device)  # Unsqueeze to add a batch dimension

                    # Get model predictions
                    with torch.no_grad():
                        outputs = bot_csd_model(transform_crop_image)
                        # print(outputs)
                        _, predicted = torch.max(outputs, 1)
                        # probabilities = F.softmax(outputs, dim=1)
                        # print(probabilities.cpu().numpy())
                        # predicted = torch.max(probabilities, 1)[1]
                        # print(predicted)
                        
                        # Convert probabilities to numpy array and round them for better readability
                        # prob_array = probabilities[0].cpu().numpy()
                        # print(prob_array)

                    # Print the prediction
                    # class_names = ['320 ML', '620 ML']  # Class names
                    class_names = ['XXS', 'XS', 'S', 'M', 'L', 'LX', 'XL', 'XXL', '2XL']  # Class names
                    # class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8']  # Class names
                    
                    predicted_class = class_names[predicted[0].cpu().numpy()]
                    # predicted_class = class_names[predicted]
                    # print(f"Index: {index}, Predicted class: {predicted_class}, prob: {prob_array[predicted[0].cpu().numpy()]}")
                    filter_bot_csd_df.loc[index, 'size'] = predicted_class
                    # filter_bot_csd_df.loc[index, 'prob_S'] = prob_array[0] # Probability of 'S'
                    # filter_bot_csd_df.loc[index, 'prob_L'] = prob_array[1] # Probability of 'L'
                    # display(filter_bot_csd_df)
                    
                yolo_predictions_df[(yolo_predictions_df['SKU_Name'].str.startswith('Bot_CSD')) | (yolo_predictions_df['SKU_Name'].str.startswith('Bot_CSD COLA'))] = filter_bot_csd_df
                
            else:
                print('No bot CSD')
                
            filter_bot_tea_df = yolo_predictions_df[yolo_predictions_df['SKU_Name'].str.startswith('Bot_Tea')]
            if len(filter_bot_tea_df) != 0:
                for index, rows in filter_bot_tea_df.iterrows():
                    image = pill_crop_image_list[index]
                    # Apply the transforms
                    transform_crop_image = get_resnet_trasforms(image, device=device)  # Unsqueeze to add a batch dimension

                    # Get model predictions
                    with torch.no_grad():
                        outputs = bot_tea_model(transform_crop_image)
                        _, predicted = torch.max(outputs, 1)

                    # Print the prediction
                    class_names = ['S', 'M', 'L', 'XL']  # Class names
                    predicted_class = class_names[predicted[0].cpu().numpy()]
                    filter_bot_tea_df.loc[index, 'size'] = predicted_class
                    
                yolo_predictions_df[yolo_predictions_df['SKU_Name'].str.startswith('Bot_Tea')] = filter_bot_tea_df
                
            else:
                print('No bot Tea')
                
            filter_bot_water_df = yolo_predictions_df[yolo_predictions_df['SKU_Name'].str.startswith('Bot_Water')]
            if len(filter_bot_water_df) != 0:
                for index, rows in filter_bot_water_df.iterrows():
                    image = pill_crop_image_list[index]
                    # Apply the transforms
                    transform_crop_image = get_resnet_trasforms(image, device=device)  # Unsqueeze to add a batch dimension

                    # Get model predictions
                    with torch.no_grad():
                        outputs = bot_water_model(transform_crop_image)
                        _, predicted = torch.max(outputs, 1)

                    # Print the prediction
                    class_names = ['S', 'M', 'L']  # Class names
                    predicted_class = class_names[predicted[0].cpu().numpy()]
                    filter_bot_water_df.loc[index, 'size'] = predicted_class
                    
                yolo_predictions_df[yolo_predictions_df['SKU_Name'].str.startswith('Bot_Water')] = filter_bot_water_df
                
            else:
                print('No bot water')
        
            yolo_predictions_merge_df = yolo_predictions_df.copy()
            # yolo_predictions_merge_df
            
            # เรียง product ตามแกน y ก่อนให้ product ที่อยู่บนสุดของรุูปขึ้นก่อน
            yolo_predictions_merge_df = yolo_predictions_merge_df.sort_values(['ymax'], ascending=[True]).reset_index(drop=True)
            # yolo_predictions_merge_df
            # หาว่า product แต่ละชิ้นอยู่บนชั้นไหนบ้าง
            yolo_predictions_merge_df = get_floor_level(yolo_predictions_merge_df)
            # yolo_predictions_merge_df
            # พอหาว่า product ไหนอยู่ชั้นไหนเสร็จ ก็เอามา sort ด้วย ชั้นที่อยู่ กับ แกน x เพื่อที่จะทำให้ตัว product เรียงลำดับจาก ซ้ายไปขวา บนลงล่าง
            yolo_predictions_merge_df = yolo_predictions_merge_df.sort_values(['Floor', 'xmin'], ascending=[True, True]).reset_index(drop=True).reset_index()
            # yolo_predictions_merge_df
            
            
            ## new query step ###
            # add size to SKU_Name
            if len(yolo_predictions_merge_df['size'].value_counts()) != 0:
                print('Have Size')
                yolo_predictions_merge_df
                not_empty = yolo_predictions_merge_df['size'].notna()
                df = yolo_predictions_merge_df.loc[not_empty]
                df['SKU_Name'] = df['SKU_Name'] + '_' + df['size']
                yolo_predictions_merge_df.loc[not_empty] = df
                yolo_predictions_merge_df = yolo_predictions_merge_df.fillna(value = "")   
            else:
                print('None Size')
                yolo_predictions_merge_df = yolo_predictions_merge_df.fillna(value = "")
                
            # yolo_predictions_merge_df
            
            
        
            
            # query ตัว lookup สำหรับ lookup จากผลของ ai ให้เป็นชื่อที่ทาง survey app รู้จัก เพื่อนนำไปทำ prelist
            # โดยตัว lookup จะถูกเก็บไว้ในรูปแบบของ dataframe ชื่อ label_lookup_df
            # lookup_list = yolo_predictions_merge_df['SKU_Name'].to_list()
            # lookup_list
            
            lookup_list = yolo_predictions_merge_df['SKU_Name'].to_list()

            if lookup_list:
                # sku_condition = ','.join([f"'{sku.replace("'", "''")}'" for sku in lookup_list])  # Escaping single quotes in SKUs
                sku_condition = ''
                for sku in lookup_list:
                    escaped_sku = sku.replace("'", "''")
                    if sku_condition:
                        sku_condition += ','
                    sku_condition += f"'{escaped_sku}'"
                print(sku_condition)
                try:
                    select_query = f"""
                                SELECT sku_id
                                ,product_type AS pack_type
                                ,sub_category AS category 
                                ,sub_brand_th AS sub_brand
                                ,flavor_th AS flavor_label
                                ,pack_size AS pack_size
                                ,ai_label AS ai_label
                                FROM dataverse.dim_sku_ai_bak20240404 dsa 
                                WHERE ((sub_category IN ('Beer', 'CSD', 'CSD Cola') AND is_represent = true) OR sub_category NOT IN ('Beer', 'CSD', 'CSD Cola'))
                                AND ai_label IN ({sku_condition})
                                GROUP BY sku_id, product_type, sub_category, sub_brand_th, flavor_th, pack_size, ai_label
                    """
                    label_result_list = PGSQL.selectDatabase(select_query)
                    label_lookup_df = pd.DataFrame.from_dict(label_result_list)
                except Exception as e:
                    print(f"Error executing SQL query: {e}")
            else:
                print("lookup_list is empty, skipping query execution.")
                        
            # print(display(label_lookup_df))
            # try:
            #     # select_query = f"""
            #     #                 select pack_type_th as pack_type
            #     #                 ,category
            #     #                 ,sub_brand_th as sub_brand
            #     #                 ,flavor_th as flavor_label
            #     #                 ,ai_label as ai_label
            #     #                 from dataverse.m_flavor_lookup_bak20240229
            #     #                 where ai_label in ({','.join([f"'{sku}'" for sku in lookup_list])})
            #     #                 group by pack_type_th, category, sub_brand_th, flavor_th, ai_label
            #     #                 """       
            #     select_query = f"""
            #                 select sku_id
            #                 ,product_type as pack_type
            #                 ,sub_category as category 
            #                 ,sub_brand_th as sub_brand
            #                 ,flavor_th as flavor_label
            #                 ,pack_size as pack_size
            #                 ,ai_label as ai_label
            #                 --,is_represent 
            #                 from dataverse.dim_sku_ai_bak20240404 dsa 
            #                 where (sub_category in ('Beer', 'CSD', 'CSD Cola') and is_represent = true) or sub_category not in ('Beer', 'CSD', 'CSD Cola')
            #                 and ai_label in ({','.join([f"'{sku}'" for sku in lookup_list])})
            #                 group by sku_id, product_type, sub_category, sub_brand_th, flavor_th, pack_size, ai_label
            #     """        
            #     label_result_list = PGSQL.selectDatabase(select_query)
            #     label_lookup_df = pd.DataFrame.from_dict(label_result_list)
            # except Exception as e:
            #     print(e)
                
            # label_lookup_df
            ### clean size with haven't size ###
            size_category_list = ['Beer', 'CSD', 'CSD Cola', 'Tea', 'Water']
            # label_lookup_df.loc[~label_lookup_df['category'].isin(size_category_list), 'pack_size'] = ""
            if 'category' in label_lookup_df.columns:
                label_lookup_df.loc[~label_lookup_df['category'].isin(size_category_list), 'pack_size'] = ""
            else:
                print("Column 'category' does not exist in the DataFrame")
            label_lookup_df
            # yolo_predictions_merge_df.info()
            # yolo_predictions_merge_df.SKU_Name
            
            json_result = []
            index = 1
            result_dict = {}
            filter_result_category_list = set()
            # yolo_predictions_merge_df
            # for df_index, row in yolo_predictions_merge_df.iterrows():
            #     # print(row['SKU_Name'])
            #     tmp_split = row["SKU_Name"].split('_')
            #     if tmp_split[1].lower() != "tea" and tmp_split[0].lower() == "box":
            #         tmp_split[0] = "Bot"
            #     filter_name = "_".join(tmp_split)
            #     tmp_split = filter_name.split('_')
            #     # print(tmp_split)
            #     if tmp_split[1].lower() != "other":
                    
            #         try:
            #             # print("filter name",filter_name)
            #             filter_df = label_lookup_df[label_lookup_df['ai_label'] == filter_name]
            #             display(filter_df)
            #             # break
            #             ai_label = filter_df['ai_label'].values[0].split('_')
            #             # ai_label[0]
                        
            #             if len(filter_df) != 0:
            #                 # pack_type = get_first_element(filter_df, 'pack_type')
            #                 # category = get_first_element(filter_df, 'category')
            #                 # sub_brand = get_first_element(filter_df, 'sub_brand')
            #                 # flavor = get_first_element(filter_df, 'flavor_label')
            #                 pack_type = ai_label[0]
            #                 category = ai_label[1]
            #                 sub_brand = ai_label[2]
            #                 flavor = ai_label[3]
            #             else:
            #                 pack_type = tmp_split[0]
            #                 category = tmp_split[1]
            #                 sub_brand = tmp_split[2]
            #                 flavor = tmp_split[3]
                        
            #             name_key = "_".join([pack_type, category, sub_brand, flavor])
            #             # print(name_key)
                        
            #             if name_key not in result_dict.keys():
                                
            #                 result_dict[name_key] = {
            #                     # index * 10 เพราะจะได้สร้าง gap ให้ app สามารถ insert บน ล่าง ได้
            #                     'index': index*10,
            #                     'pack_type': pack_type,
            #                     'category': category,
            #                     'brand': sub_brand,
            #                     'flavor': flavor,
            #                     'facing': 1
            #                 }
                            
            #                 filter_result_category_list.add(result_dict[name_key]['category'])
            #                 index += 1
            #                 json_result.append(result_dict[name_key])
                        
            #             else:
            #                 result_dict[name_key]['facing'] += 1
                            
            #         except Exception as e:
            #             pack_type = tmp_split[0]
            #             category = tmp_split[1]
            #             sub_brand = tmp_split[2]
            #             flavor = tmp_split[3]
                    
            #     yolo_predictions_merge_df.loc[df_index, ['pack_type']] = pack_type
            #     yolo_predictions_merge_df.loc[df_index, ['category']] = category
            #     yolo_predictions_merge_df.loc[df_index, ['brand']] = sub_brand
            #     yolo_predictions_merge_df.loc[df_index, ['flavor']] = flavor
            # process ในการ lookup ชื่อและปั้นผลออกไปให้ survey app
            for df_index, row in yolo_predictions_merge_df.iterrows():
                sku_id = ""
                # เมื่อ split จะทำให้ได้ชื่อ 4 index โดย index 0 คือ packtype, 1 : category, 2 : brand, 3 : flavor 4: pack_size
                tmp_split = row["SKU_Name"].split('_')
                # print(tmp_split)
                # ดักเงื่อนไขเพื่อแปลงชื่อสำหรับ lookup
                if tmp_split[1].lower() != "tea" and tmp_split[0].lower() == "box":
                    tmp_split[0] = "Bot"
                filter_name = "_".join(tmp_split)
                # filter_size = "_".join(tmp_split[-1:])
                # print(filter_name)
                tmp_split = filter_name.split('_')
                # print(tmp_split)
                # ดักไม่ให้ other ถูกส่งไปหน้าบ้าน
                if tmp_split[1].lower() != "other":
                    try:
                        # lookup หาชื่อที่ตรงกัน เพื่อส่งไปให้ทาง app
                        # print(label_lookup_df['ai_label'])
                        # print(label_lookup_df[label_lookup_df['ai_label'] == filter_name])
                        filter_df = label_lookup_df[label_lookup_df['ai_label'] == filter_name]
                        # display(filter_df)
                        sku_id = filter_df['sku_id'].values[0]
                        pack_type = filter_df['pack_type'].values[0]
                        category = filter_df['category'].values[0]
                        sub_brand = filter_df['sub_brand'].values[0]
                        flavor = filter_df['flavor_label'].values[0]
                        updated_sizes = filter_df['pack_size'].values[0] if 'pack_size' in filter_df else "" # handling the error
                        # sizes = filter_size
                        # if sizes in size_list:
                        #     updated_sizes = size_list[sizes]
                        # updated_sizes = [size_list[size] if size in size_list else size for size in sizes]
                        # updated_sizes = updated_sizes[0]
                        # print(updated_sizes)
                            
                        name_key = "_".join([pack_type, category, sub_brand, flavor, updated_sizes])
                        # print(name_key)
                        if name_key not in result_dict.keys():
                            
                            result_dict[name_key] = {
                                # index * 10 เพราะจะได้สร้าง gap ให้ app สามารถ insert บน ล่าง ได้
                                'index': index*10,
                                'pack_type': pack_type,
                                'category': category,
                                'brand': sub_brand,
                                'flavor': flavor,
                                'size': updated_sizes,
                                'facing': 1,
                            }
                            
                            # เช็คว่า category ที่จะส่งไป ตรงกับชนิดของตู้ๆ  ที่ user ถ่ายมาหรือไม่
                            # if (shelf_type == 'refig' and category not in shelf_category_list) or (shelf_type == 'shelf' and category in shelf_category_list) or (category in special_category_list):
                            filter_result_category_list.add(result_dict[name_key]['category'])
                            index += 1
                            json_result.append(result_dict[name_key])
                            # index += 1
                            # json_result.append(result_dict[filter_name])

                        else:
                            result_dict[name_key]['facing'] += 1
                                
                    except Exception as e:
                        print(e)
                        # print(filter_name)
                        # print(tmp_split)
                        sku_name = ""
                        pack_type = tmp_split[0]
                        category = tmp_split[1]
                        sub_brand = tmp_split[2]
                        flavor = tmp_split[3]
                        # print("Check error")
                        # print(tmp_split[4])
                        updated_sizes = tmp_split[4] if len(tmp_split) > 4 else "" # handling the error
                else:
                    other_label = row["SKU_Name"].split('_')
                    sku_id = ""
                    pack_type = other_label[0]
                    category = other_label[1]
                    sub_brand = other_label[2]
                    flavor = other_label[3]
                    updated_sizes = other_label[4] if len(other_label) > 4 else "" # handling the error

                yolo_predictions_merge_df.loc[df_index, ['sku_id']] = sku_id
                yolo_predictions_merge_df.loc[df_index, ['pack_type']] = pack_type
                yolo_predictions_merge_df.loc[df_index, ['category']] = category
                yolo_predictions_merge_df.loc[df_index, ['brand']] = sub_brand
                yolo_predictions_merge_df.loc[df_index, ['flavor']] = flavor
                yolo_predictions_merge_df.loc[df_index, ['pack_size']] = updated_sizes
            
            # json_result
            # yolo_predictions_merge_df.info()
            # display(yolo_predictions_merge_df.info())
            # display(json_result)
            
            result_image = show_result_qdvp(tmp_pil, yolo_predictions_merge_df, False, True)
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            result_image_pill = Image.fromarray(result_image)
            display(result_image_pill)
    
            filter_condition = yolo_predictions_merge_df.category.isin(list(filter_result_category_list))
            # display(filter_condition)
            # display(yolo_predictions_merge_df)
            # display(result_df[filter_condition])
            # display(result_df[filter_condition][['pack_type', 'category', 'brand', 'flavor', 'size', 'real_facing']].rename(columns={'real_facing': 'facing_number'}))
            # display(yolo_predictions_merge_df[filter_condition][['pack_type', 'category', 'brand', 'flavor', 'size_th']].value_counts().reset_index().rename(columns={'brand': 'brand', 0: 'facing_number'}))
            now = datetime.utcnow() + utc_shift
            current_date = now.strftime("%Y-%m-%d")

            # เตรียมไว้สำหรับทำ timestamp ใน log
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
            
            # filter_other = ['Other']
            # if len(yolo_predictions_merge_df[~yolo_predictions_merge_df['category'].isin(filter_other)]) > 0:
            #     count_df = yolo_predictions_merge_df[~yolo_predictions_merge_df['category'].isin(filter_other)]
            #     count_df = count_df[['pack_type', 'category', 'brand', 'flavor', 'size']].value_counts().reset_index(name='facing_number')
            #     count_df['id'] = photo_id
            #     count_df['create_datetime'] = dt_string
            #     # display(count_df)
            #     PGSQL.insertDatabaseList(ENERGY_LOG, count_df.to_dict('records'))
            
            if len(yolo_predictions_merge_df[filter_condition]) > 0:
                # count_df = yolo_predictions_merge_df[filter_condition][['pack_type', 'category', 'brand', 'flavor', 'size']].value_counts().reset_index(name='facing_number')
                count_df = yolo_predictions_merge_df[filter_condition][['pack_type', 'category', 'brand', 'flavor', 'pack_size', 'size']].value_counts().reset_index(name='facing_number')
                count_df = count_df.rename(columns={'size': 'group_size'})
                # count_df
                count_df['photo_id'] = photo_id
                count_df['create_datetime'] = dt_string
                # count_df.to_dict('records')
                # count_df
                PGSQL.insertDatabaseList(BOT_CSD_LOG, count_df.to_dict('records'))
            #     # count_df = yolo_predictions_merge_df[filter_condition][['pack_type', 'category', 'brand', 'flavor', 'size', 'real_facing']].rename(columns={'real_facing': 'facing_number'})
            #     # count_df = yolo_predictions_merge_df[filter_condition][['pack_type', 'category', 'brand', 'flavor', 'size']].value_counts().reset_index().rename(columns={0: 'facing_number'})
            #     count_df = yolo_predictions_merge_df[filter_condition][['pack_type', 'category', 'brand', 'flavor', 'size']].value_counts().reset_index(name='facing_number')
            #     count_df['id'] = photo_id
            #     count_df['create_datetime'] = dt_string
            #     # display(count_df)
            #     PGSQL.insertDatabaseList(BEER_LOG, count_df.to_dict('records'))
            
        # save raw image to obs and null result to postgresql
        else:
            # ในกรณีที่ไม่เจอ product เลยใน log จะเก็บเป็น unknown ให้กับรูปภาพนั้น ๆ
            image_rgb = np.array(tmp_pil)
            img = image_rgb[:, :, ::-1].copy()
            result_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result_image_pill = Image.fromarray(result_image)
            # threading.Thread(target=upload_image_obs, args=(result_image_pill, obs_client, bucket_name,
            #                                                 'runtime_photo_ai/' + current_date + '_no_result/' + ref_uuid)).start()

            tmp_json_list = [
                {
                    "photo_id": photo_id,
                    "pack_type": "unknown",
                    "category": "unknown",
                    "brand": "unknown",
                    "flavor": "unknown",
                    "pack_size": "unknown",
                    "group_size": "unknown",
                    "facing_number": 0,
                    "create_datetime": dt_string
                }
            ]
            # PGSQL = PostgreSQLOperator(DB_HOST, DB_PORT, DB_NAME, DB_USERNAME, DB_PASSWORD)
            PGSQL.insertDatabaseList(BOT_CSD_LOG, tmp_json_list)
            
        
        ### เก็บ log raw results ###   
        yolo_predictions_merge_df
        yolo_predictions_merge_df["photo_id"] = photo_id
        yolo_predictions_merge_df['create_datetime'] = dt_string
        
        # เก็บ ผลดิบ ที่ถูก predict ออกมาจาก AI เพื่อเอาไปใช้วิเคราะห์ต่อ
        yolo_predictions_merge_df = yolo_predictions_merge_df.rename(columns={'xmin': 'x_min', 'ymin': 'y_min', 'xmax': 'x_max', 'ymax': 'y_max', 'Floor': 'floor_level'})
        
        if 'SKU_Name' in yolo_predictions_merge_df.columns:
            yolo_predictions_merge_df['SKU_Name'] = yolo_predictions_merge_df['SKU_Name'].str.replace("'", "\'")
        else:
            print("Column 'SKU_Name' does not exist in the DataFrame")
        # yolo_predictions_merge_df['SKU_Name'] = yolo_predictions_merge_df['SKU_Name'].str.replace("'", "\'")
        yolo_predictions_merge_df = yolo_predictions_merge_df.rename(columns={'SKU_Name': 'ai_label', 'size': 'group_size'})
        # yolo_predictions_merge_df.info()
        
        # save to log database raw results
        PGSQL.insertDatabaseList(BOT_CSD_RAW_RESULT_LOG, yolo_predictions_merge_df[['photo_id', 'index', 'x_min', 'y_min', 'x_max', 'y_max', 'BBox_Area', 'ai_label','pack_type', 'category', 'brand', 'flavor','pack_size','group_size','prediction_label','prediction_dist', 'create_datetime', 'floor_level']].to_dict('records'))
        
        PGSQL.close()
        
        # torch allowcated memory
        torch.cuda.memory_allocated()
        
        buffered = BytesIO()
        result_image_pill.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        
        # อัพโหลดผลขึ้น obs
        try:
            # upload_result_image(ref_uuid, img_str)
            upload_result_image(photo_id, img_str, MQ_USER=MQ_USER, MQ_PASS=MQ_PASS, MQ_HOST=MQ_HOST, MQ_PORT=MQ_PORT, MQ_V_HOST=MQ_V_HOST, QUEUE_AI_RESULT=QUEUE_AI_RESULT)
            
        except Exception as e:
            print(f"Error during image upload: {e}")
            
        iteration_end_time = time.time()
        times.append(iteration_end_time - iteration_start_time)
         # Calculate average time per iteration
        avg_time_per_iteration = sum(times) / len(times)

        # Estimate time remaining
        time_remaining = (avg_time_per_iteration * (len(data_ids) - (i + 1)) / 60) / 60

        tqdm.write(f"Estimated time remaining: {time_remaining:.2f} hr")
        
        i += 1
        
    print(f"Total time taken: {sum(times):.2f} seconds")  
                
            
if __name__ == '__main__':

    main()
    print(f"[INFO] finished...")