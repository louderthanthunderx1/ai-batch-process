import io
import pika
import cv2
import torchvision.transforms as transforms
import numpy as np

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

# ฟังก์ชั่น ไว้ output ภาพที่ มีการตีกรอบแล้ว
def show_result_qdvp(image_path, yolo_predictions_df, show=False, is_loaded=False):
    # เช็คว่าภาพโหลดมารึยัง
    if is_loaded:
        image_rgb = np.array(image_path)
        img = image_rgb[:, :, ::-1].copy()
    else:
        # Define threshold and image path
        img = cv2.imread(image_path)

    # แกะพวกค่าต่างๆออกมาจาก df
    bbox_result = yolo_predictions_df['BBox']
    bboxes = np.vstack(bbox_result)

    # brand_list = list(reference_image_dict_image_ver.keys())
    # color_list = list(np.random.random(size=len(brand_list)) * 255)

    labels_list = list(yolo_predictions_df['brand'])
    labels_list_np = np.array(labels_list)

    labels_brand_list = list(yolo_predictions_df['SKU_Name'])
    labels_brand_list_np = np.array(labels_brand_list)

    index_np = sorted(np.array(yolo_predictions_df['index']))
    floor_level_np = sorted(np.array(yolo_predictions_df['floor_level']))

    # วนลูปเพื่อ ตีกรอบลงไปในรูป
    for bbox, label, labels_brand, index, floor_level in zip(bboxes, labels_list_np, labels_brand_list_np, index_np, floor_level_np):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])

        # brand_index = brand_list.index(labels_brand)
        # brand_color = color_list[brand_index]

        # ตีกรอบสีเหลี่ยม
        cv2.rectangle(
            img, left_top, right_bottom, color=(0, 255, 255), thickness=2)

        display_label = labels_brand.split('_')

        # เขียน class ของ object นั้นลงไป
        cv2.putText(img, f'[{str(index)}],f{str(floor_level)}', (bbox_int[0] + 5, bbox_int[1] + 15), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=(255, 255, 51))

        y0 = bbox_int[1] + 15 + 20
        dy = 20
        for i, line in enumerate(display_label):
            y = y0 + i * dy
            cv2.putText(img, line, (bbox_int[0] + 5, y), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=(255, 255, 51))

    if show:
        cv2.imshow(img)

    return img

# ฟังก์ชั่น ไว้ processing SKU name pattern ( ex. BOT_Beer_Chang_Original ) ให้แยกเป็น column
def split_row(row):

    name_split = row['SKU_Name'].split('_')
    row['pack_type'] = name_split[0]
    row['category'] = name_split[1]
    row['brand'] = name_split[2]
    row['flavor'] = name_split[3]

    return row

# def upload_image_obs(image_pil, obs_client, bucket_name, image_id, headers_function):
    
#     result_image_byte = io.BytesIO()
#     image_pil.save(result_image_byte, format='PNG')
#     result_image_byte = result_image_byte.getvalue()

#     object_key = image_id +  ".jpg"

#     # headers = PutObjectHeader()
#     headers = headers_function
#     headers.contentType = 'text/plain'

#     resp = obs_client.putContent(bucket_name, object_key, result_image_byte, metadata = {'meta1':'value1', 'meta2':'value2'}, headers = headers)

#     if resp.status < 300:
#         print('requestId:', resp.requestId)
#         print('etag:', resp.body.etag)
#         print('versionId:', resp.body.versionId)
#         print('storageClass:', resp.body.storageClass)
#     else:
#         print('errorCode:', resp.errorCode)
#         print('errorMessage:', resp.errorMessage)

def upload_result_image(ref_id, base64, MQ_USER, MQ_PASS, MQ_HOST, MQ_PORT, MQ_V_HOST, QUEUE_AI_RESULT):
    credentials = pika.PlainCredentials(MQ_USER, MQ_PASS)
    parameters = pika.ConnectionParameters(host=MQ_HOST, port=MQ_PORT, virtual_host=MQ_V_HOST, credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue=QUEUE_AI_RESULT)
    # channel.basic_publish(routing_key=QUEUE_AI_RESULT, exchange='', properties=pika.BasicProperties(headers={"ref_id": ref_id}),
                        #   body=base64)
    channel.basic_publish(routing_key=QUEUE_AI_RESULT, exchange='', properties=pika.BasicProperties(headers={"ref_id": ref_id, "path": "runtime_photo_ai"}), # แก้ path ที่เก็บรูปตรงนี้
                          body=base64)                   
    connection.close()

def convert_to_degrees(value):
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

# หาว่า object แต่ละอันอยู่ชั้นที่เท่าไหร่บ้าง
def get_floor_level(prediction_df):
   # Calculate the threshold based on a percentage of the maximum 'y_max' value
    threshold_percentage = 0.12
    threshold = threshold_percentage * prediction_df['ymax'].max()
    # print(prediction_df['ymax'].max())
    # # print(prediction_df['ymax'].min())
    # # print(prediction_df['ymax'].max() - prediction_df['ymin'].max())
    # print(threshold)

    # Initialize floor level and new_floor_level
    floor_level = 1

    # Calculate the difference between consecutive 'y_max' values
    y_max_diff = prediction_df['ymax'].diff()
    # print(y_max_diff)
    # print(y_max_diff.abs())
    # Create a mask for when the difference exceeds the threshold
    mask = y_max_diff.abs() > threshold
    # print(mask)
    # print(mask.cumsum())
    # print(floor_level+mask.cumsum())
    # Add 1 for each True value in the mask and cumulative sum
    prediction_df['Floor'] = floor_level + mask.cumsum()

    return prediction_df


# Function to modify a single SKU_Name
def transform_sku_name(sku_name):
    tmp_split = sku_name.split('_')

    # Modify based on conditions
    if len(tmp_split) > 1 and tmp_split[1].lower() != "tea" and tmp_split[0].lower() == "box":
        tmp_split[0] = "Bot"

    # Reassemble the modified name
    return "_".join(tmp_split)