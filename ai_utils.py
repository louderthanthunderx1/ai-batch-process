import torch
import torchvision.transforms as transforms
import cv2
import time
from PIL import Image
import numpy as np
import io
import pika
import pandas as pd

from ai_helper import pil_to_tensor

# เทียบ sim ของ object แต่ละตัว
def sku_detection_faiss(image_rgb, yolo_predictions_df, encoder_model, n_neighbor, cfg, index_flat, pakage_brand_list):
  """
  Loop throgh each detected packages. Use the crop image to find the most similar SKU.
  """
  knn_list_label = []
  knn_list_index = []
  knn_list_distance = []
  stat_label_list = []
  pill_crop_image_list = []
  segment_image_tensor_list = []

  bounding_box_list = list(yolo_predictions_df['BBox'])

  # Loop through each package bonding box.
  # Loop สำหรับ Crop
  for index, bounding_box in enumerate(bounding_box_list):

    # Crop each package from original image.
    segment_bounding_box = bounding_box
    segment_image = image_rgb.crop(segment_bounding_box)

    pill_crop_image_list.append(segment_image)
    segment_image_tensor = pil_to_tensor(segment_image, cfg).cuda()
    segment_image_tensor_list.append(segment_image_tensor[0])

  segment_image_tensor_list = torch.stack(segment_image_tensor_list).cuda()

  # Encode object ที่ crop มาทั้งหมด มา Encode
  encoder_model.eval()
  with torch.no_grad():
    semgnet_image_encoded_list = encoder_model(segment_image_tensor_list)

  # วนลูป เพื่อเทียบ sim กับ FAISS
  # for semgnet_image_encoded in semgnet_image_encoded_list[0]:
  #
  #   semgnet_image_encoded = np.array(semgnet_image_encoded.view(1, -1).cpu())
  #   # print(type(index_flat))
  #   FAISS_distance, FAISS_index = index_flat.search(x=semgnet_image_encoded, k=5)
  #
  #   knn_list_label.append(pakage_brand_list[FAISS_index[0]])
  #   knn_list_index.append(FAISS_index[0])
  #
  #   knn_list_distance.append(FAISS_distance[0])

  segment_image_encoded_np = semgnet_image_encoded_list[0].cpu().numpy()
  FAISS_distance, FAISS_index = index_flat.search(x=segment_image_encoded_np, k=n_neighbor)
  knn_list_label = list(pakage_brand_list[FAISS_index])
  knn_list_index = list(FAISS_index)
  knn_list_distance = list(FAISS_distance)

  # for label_list in knn_list_label:
  #     label_array = np.array(label_list)
  #     unique, counts = np.unique(label_array, return_counts=True)
  #     result = np.column_stack((unique, counts))
  #     sorted_result = np.flip(result[counts.argsort()], axis=0)
  #
  #     stat_label_list.append(sorted_result)

  return pill_crop_image_list, knn_list_label, knn_list_index, knn_list_distance


# ส่งรูปเข้ามาแล้วเพื่อหาผลออกเป็น Dataframe
def get_prediction_df_faiss(inference_image_path, rotate_pil, is_loaded=False, yolov5_model=None, encoder_model=None, cfg=None, index_flat=None, pakage_brand_list=None):
    # start = time.time()
    # เช็คว่าภาพโหลดมารึยัง
    if is_loaded:
        image_rgb = inference_image_path.convert('RGB')
    else:
        inference_image_path = inference_image_path
        # Read image into BGR format
        image_rgb = Image.open(inference_image_path)
    # print("1: {}".format(time.time() - start))
    # Predict packages bouding box
    # Predict product ด้วย yolo
    results = yolov5_model(image_rgb)
    # print("2: {}".format(time.time() - start))
    # แกะค่าต่างๆออกมา ถ้าเป็น model ตัวอื่น ก็แกะให้ได้ค่าตรงกับ format ถัดไปที่จะรับ
    yolo_predictions_df = results.pandas().xyxy[0]
    yolo_predictions_df['BBox'] = [[xmin, ymin, xmax, ymax] for xmin, ymin, xmax, ymax in
                                   zip(yolo_predictions_df['xmin'], yolo_predictions_df['ymin'],
                                       yolo_predictions_df['xmax'], yolo_predictions_df['ymax'])]

    bbox_area_list = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in list(yolo_predictions_df['BBox'])]
    yolo_predictions_df['BBox_Area'] = bbox_area_list
    # print("3: {}".format(time.time() - start))
    # ถ้ามัน predict ไม่เจออะไรก็จบ
    if len(yolo_predictions_df) == 0:
        prediction_df = pd.DataFrame(
            columns=['SKU_Index', 'SKU_Name', 'X1_Position', 'X2_Position', 'Y1_Position', 'Y2_Position', 'BBox',
                     'Bbox_Area', 'Image'])
        return [], [], [], [], []
    # print("4: {}".format(time.time() - start))
    # ส่งไปทำ sim เพื่อ predict brand
    n_neighbor = 5
    pill_crop_image_list, knn_list_label, knn_list_index, knn_list_distance = sku_detection_faiss(image_rgb=rotate_pil,
                                                                                                  yolo_predictions_df=yolo_predictions_df,
                                                                                                  encoder_model=encoder_model,
                                                                                                  n_neighbor=n_neighbor,
                                                                                                  cfg=cfg,
                                                                                                  index_flat=index_flat,
                                                                                                  pakage_brand_list=pakage_brand_list)
    # print("5: {}".format(time.time() - start))
    final_prediction = np.array(knn_list_label)[:, 0]
    yolo_predictions_df['SKU_Name'] = list(final_prediction)

    # process แยก pack_type, category, brand, flavor จาก sku name pattern
    # yolo_predictions_df = yolo_predictions_df.apply(split_row, axis=1)

    # print("6: {}".format(time.time() - start))

    return yolo_predictions_df, pill_crop_image_list, knn_list_label, knn_list_index, knn_list_distance


