# -------------------------------------------------------------
# 1. 导入所需库
# -------------------------------------------------------------
import os
import sys
import glob
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.notebook import display
from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')  # 忽略警告信息

# -------------------------------------------------------------
# 2. 基础数据加载与演示
# -------------------------------------------------------------
# 以下部分代码演示如何加载和查看某个JSON标注文件与对应的视频帧

# 2.1 加载一个JSON文件中的训练集标注数据：第45个视频的标注文件
train_anno_demo = json.load(open('训练集(有标注第一批)/标注/45.json', encoding='utf-8'))
print("演示：标注文件中的第一条记录：", train_anno_demo[0])
print("演示：该标注文件的长度：", len(train_anno_demo))

# 2.2 使用Pandas读取并显示JSON格式的标注数据
print("演示：Pandas显示标注文件：")
display(pd.read_json('训练集(有标注第一批)/标注/45.json'))

# 2.3 打开并读取指定视频文件的一帧（第45个视频）
video_path_demo = '训练集(有标注第一批)/视频/45.mp4'
cap_demo = cv2.VideoCapture(video_path_demo)

ret_demo, frame_demo = cap_demo.read()
if ret_demo:
    print("演示：读取的帧尺寸:", frame_demo.shape)

# 2.4 获取视频的总帧数
total_frames_demo = int(cap_demo.get(cv2.CAP_PROP_FRAME_COUNT))
print("演示：该视频总帧数:", total_frames_demo)

# 2.5 简单演示在帧上绘制一个矩形框并显示
bbox_demo = [746, 494, 988, 786]  # [x_min, y_min, x_max, y_max]
pt1_demo = (bbox_demo[0], bbox_demo[1])
pt2_demo = (bbox_demo[2], bbox_demo[3])
color_demo = (0, 255, 0)  # 绿色
thickness_demo = 2

# 在图像上绘制边界框（如果帧读取成功）
if ret_demo:
    cv2.rectangle(frame_demo, pt1_demo, pt2_demo, color_demo, thickness_demo)
    frame_demo = cv2.cvtColor(frame_demo, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_demo)
    plt.title("在图像上绘制矩形框示例")
    plt.axis('off')
    plt.show()

# -------------------------------------------------------------
# 3. 准备YOLO数据集目录与配置
# -------------------------------------------------------------
# 3.1 创建数据集所需目录
if not os.path.exists('yolo-dataset/'):
    os.mkdir('yolo-dataset/')
if not os.path.exists('yolo-dataset/train'):
    os.mkdir('yolo-dataset/train')
if not os.path.exists('yolo-dataset/val'):
    os.mkdir('yolo-dataset/val')

# 3.2 生成YOLO的配置文件yolo.yaml
dir_path = os.path.abspath('./') + '/'
yolo_config_path = 'yolo-dataset/yolo.yaml'
with open(yolo_config_path, 'w', encoding='utf-8') as up:
    up.write(f'''
path: {dir_path}/yolo-dataset/
train: train/
val: val/

names:
    0: 非机动车违停
    1: 机动车违停
    2: 垃圾桶满溢
    3: 违法经营
''')

print(f"已生成YOLO配置文件：{yolo_config_path}")

# -------------------------------------------------------------
# 4. 加载训练集标注并生成YOLO训练数据
# -------------------------------------------------------------
train_annos = glob.glob('训练集(有标注第一批)/标注/*.json')
train_videos = glob.glob('训练集(有标注第一批)/视频/*.mp4')

train_annos.sort()
train_videos.sort()

# 定义类别标签
category_labels = ["非机动车违停", "机动车违停", "垃圾桶满溢", "违法经营"]

# 4.1 将前5个视频及其标注文件用于训练集
for anno_path, video_path in zip(train_annos[:5], train_videos[:5]):
    print("[训练集] 处理视频:", video_path)
    anno_df = pd.read_json(anno_path)  # 读取JSON标注为DataFrame
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取结束

        img_height, img_width = frame.shape[:2]
        frame_anno = anno_df[anno_df['frame_id'] == frame_idx]

        # 保存当前帧为JPEG
        image_name = f"{anno_path.split('/')[-1][:-5]}_{frame_idx}.jpg"
        img_save_path = os.path.join('yolo-dataset/train', image_name)
        cv2.imwrite(img_save_path, frame)

        # 如果该帧有标注，则保存相应的YOLO格式标注
        if len(frame_anno) > 0:
            txt_name = f"{anno_path.split('/')[-1][:-5]}_{frame_idx}.txt"
            txt_save_path = os.path.join('yolo-dataset/train', txt_name)
            with open(txt_save_path, 'w') as up:
                for category, bbox in zip(frame_anno['category'].values, frame_anno['bbox'].values):
                    category_idx = category_labels.index(category)
                    x_min, y_min, x_max, y_max = bbox
                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    # 调试信息：若中心点大于1说明数值有问题
                    if x_center > 1:
                        print("异常边界框:", bbox)

                    up.write(f'{category_idx} {x_center} {y_center} {width} {height}\n')

        frame_idx += 1
    cap.release()

# 4.2 将最后3个视频及其标注文件用于验证集
for anno_path, video_path in zip(train_annos[-3:], train_videos[-3:]):
    print("[验证集] 处理视频:", video_path)
    anno_df = pd.read_json(anno_path)
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_height, img_width = frame.shape[:2]
        frame_anno = anno_df[anno_df['frame_id'] == frame_idx]

        # 保存当前帧为JPEG
        image_name = f"{anno_path.split('/')[-1][:-5]}_{frame_idx}.jpg"
        img_save_path = os.path.join('yolo-dataset/val', image_name)
        cv2.imwrite(img_save_path, frame)

        # 如果该帧有标注，则保存相应的YOLO格式标注
        if len(frame_anno) > 0:
            txt_name = f"{anno_path.split('/')[-1][:-5]}_{frame_idx}.txt"
            txt_save_path = os.path.join('yolo-dataset/val', txt_name)
            with open(txt_save_path, 'w') as up:
                for category, bbox in zip(frame_anno['category'].values, frame_anno['bbox'].values):
                    category_idx = category_labels.index(category)
                    x_min, y_min, x_max, y_max = bbox
                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    up.write(f'{category_idx} {x_center} {y_center} {width} {height}\n')

        frame_idx += 1
    cap.release()

print("YOLO数据集构建完成。")

# -------------------------------------------------------------
# 5. 模型训练
# -------------------------------------------------------------
# 若仅需数据集，后续训练可在shell或其他环境中执行；也可在此直接调用

# 设置环境变量，指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = YOLO("yolov8n.pt")  # 加载官方预训练模型
results = model.train(
    data="yolo-dataset/yolo.yaml",
    epochs=2,
    imgsz=1080,
    batch=16
)

print("训练完成。模型保存在: runs/detect/train/")

# -------------------------------------------------------------
# 6. 模型推理与提交文件生成
# -------------------------------------------------------------
# 6.1 加载最佳模型
model = YOLO("runs/detect/train/weights/best.pt")

# 6.2 处理测试集视频并生成推理结果JSON
if not os.path.exists('result/'):
    os.mkdir('result')

for path in glob.glob('测试集/*.mp4'):
    print("[推理] 处理测试视频:", path)
    submit_json = []
    # 设置推理置信度阈值为0.05，可根据实际情况调整
    results_infer = model(path, conf=0.05, imgsz=1080, verbose=False)

    for idx, result in enumerate(results_infer):
        boxes = result.boxes
        if len(boxes.cls) == 0:
            continue  # 如果没有检测到目标，跳过该帧

        # 解析边界框信息
        xyxy = boxes.xyxy.data.cpu().numpy().round()  # [x_min, y_min, x_max, y_max]
        cls_ids = boxes.cls.data.cpu().numpy().round()
        confs = boxes.conf.data.cpu().numpy()

        # 将检测结果写入submit_json列表
        for i, (ci, xy, confi) in enumerate(zip(cls_ids, xyxy, confs)):
            submit_json.append({
                'frame_id': idx,
                'event_id': i + 1,
                'category': category_labels[int(ci)],
                'bbox': list(map(int, xy)),
                'confidence': float(confi)
            })

    # 将检测结果写入 JSON 文件
    result_name = os.path.splitext(os.path.basename(path))[0] + '.json'
    result_save_path = os.path.join('result', result_name)
    with open(result_save_path, 'w', encoding='utf-8') as up:
        json.dump(submit_json, up, indent=4, ensure_ascii=False)

    print(f"[推理] 生成结果文件: {result_save_path}")

print("所有测试视频推理完成，结果已保存在 result/ 目录中。")
