import math
import os

from paddleocr import PaddleOCR

os.chdir('../mmpose')
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from mmdet.apis import inference_detector, init_detector

OCR_model = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, show_log=False)  # 初始化OCR模型


def pic_predic(img_path, detector, pose_estimator):
    class num_instance():
        img = None  # 当前完整图片，且需要为PIL图片
        bbox = None
        keypoint = None
        value = ''

        def __init__(self, bbox, keypoint, img_path):
            self.bbox = bbox
            self.keypoint = keypoint
            img = cv2.imread(img_path)
            self.img = img[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]  # (y, x)
            self.img = cv2.resize(self.img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        def num_value(self):
            cv2.imwrite('temp_sip.png', self.img)
            result = OCR_model.ocr('temp_sip.png', cls=True)
            str = ''
            if result[0]:
                txts = [line[1][0] for line in result[0]]
                for p in txts:
                    str += p
                result = str
            else:
                result = ''

            if result.replace(" ", "") == '':
                # print('OCR识别为空')
                pass
            self.value = result
            return result  # 为识别的字符串

        def return_path(self):
            # cv2.imwrite('temp_sip.png', self.img)
            return self.img, self.bbox

    class pin_instance():
        keypoints = []

    # 获取一条直线
    def calc_abc_from_line_2d(x0, y0, x1, y1):
        a = y0 - y1
        b = x1 - x0
        c = x0 * y1 - x1 * y0
        return a, b, c

    # 计算两直线交点。
    def get_line_cross_point(line1, line2):
        a0, b0, c0 = calc_abc_from_line_2d(*line1)
        a1, b1, c1 = calc_abc_from_line_2d(*line2)
        D = a0 * b1 - a1 * b0
        if D == 0:
            return None
        x = (b0 * c1 - b1 * c0) / D
        y = (a1 * c0 - a0 * c1) / D
        return x, y

    # 计算OCR识别内容是否为数字
    def first_digit(string):
        num_list_new = []  # 新建空列表，用以存储提取的数值
        a = ''  # 将空值赋值给a
        for i in string:  # 将字符串进行遍历
            if str.isdigit(i):  # 判断i是否为数字，如果“是”返回True，“不是”返回False
                a += i  # 如果i是数字格式，将i以字符串格式加到a上
            else:
                a += " "  # 如果i不是数字格式，将“ ”（空格）加到a上
        # 数字与数字之间存在许多空格，所以需要对字符串a按''进行分割。
        num_list = a.split(" ")  # 按''进行分割，此时a由字符串格式编程列表
        # print("num_list is \n", num_list)
        for i in num_list:  # 对列表a，进行遍历
            try:  # try 结构体，防止出错后直接退出程序
                if int(i) > 0:
                    num_list_new.append(int(i))  # 如果列表a的元素为数字，则赋值给num_list_new
                else:
                    pass  # 如果不是数字，pass
            except:
                pass
        # print("num_list is \n", num_list_new)
        # 打印出的结果[198, 4747, 12305, 15498915, 105, 386379, 177, 4217, 14645390, 21, 530, 853525]。是我们需要的数字。
        # print("len(num_list_new)", len(num_list_new))  # 作为验证，可以数一下列表元素个数
        # 打印结果为：12
        if len(num_list_new) != 0:
            return num_list_new[0]
        else:
            return 0

    # OPENCV中心点读数
    def component(src):
        src = cv2.GaussianBlur(src, (3, 3), 0)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # 使用开运算去掉外部的噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        output = cv2.connectedComponents(binary, connectivity=8, ltype=cv2.CV_32S)
        num_labels = output[0]
        # print(num_labels)  # output: 4
        labels = output[1]  # 对应的array矩阵，其值为对应的标签
        # print(labels)
        # 构造颜色
        colors = []
        for i in range(num_labels):
            b = np.random.randint(0, 256)
            g = np.random.randint(0, 256)
            r = np.random.randint(0, 256)
            colors.append((b, g, r))  # 创建四种不同颜色
        colors[0] = (0, 0, 0)  # 固定背景为黑色

        # 画出连通图
        h, w = gray.shape
        max_height_pixels = []  # 用于存储最长刻度
        max_height = 0
        min_height = 10000
        box_h = 0
        point = None
        image = np.zeros((h, w, 3), dtype=np.uint8)  # 绘制空白图像
        for row in range(h):
            for col in range(w):
                if labels[row, col] == 1:  # 指定像素值
                    # 此处经行刻度值像素值处理
                    image[row, col] = colors[labels[row, col]]  # 绘制对应的颜色
                    if row >= max_height:
                        max_height = row
                    if row <= min_height:
                        min_height = row
                else:
                    image[row, col] = (255, 255, 255)  # 底色为白色
        for row in range(h):
            for col in range(w):
                if row == max_height:
                    max_height_pixels.append(col)
        if len(max_height_pixels) > 2:
            average_col = int(sum(max_height_pixels) / len(max_height_pixels))
            point = (average_col, min_height)
        elif len(max_height_pixels) == 1:
            point = (max_height_pixels[0], min_height)
        else:
            point = (0, 0)
            print('未找到中心点')
        # 消除干的其它刻度值
        box_h = int(abs(max_height - min_height) / 3 * 2)
        y_thr = min_height + box_h
        # 消除所有y_thr以上的像素值
        min_x, min_y = 1000, 1000
        max_x, max_y = 0, 0
        for row in range(h):
            for col in range(w):
                if row <= y_thr:  # 指定像素值
                    # 此处经行刻度值像素值处理
                    continue
                else:
                    if labels[row, col] == 1:  # 指定像素值
                        if row <= min_y: min_y = row
                        if col <= min_x: min_x = col
                        if row >= max_y: max_y = row
                        if col >= max_x: max_x = col
        min_y = min_height
        x = min_x + int(abs(max_x - min_x) / 2)
        point = (x, min_height)
        return point

    # 预测-目标检测
    init_default_scope(detector.cfg.get('default_scope', 'mmdet'))
    detect_result = inference_detector(detector, img_path)  # 目标检测
    # print(detect_result.keys())  # 预测类别种类
    # 预测类别
    # print(detect_result.pred_instances.labels) # 为什么label只有0和1
    # 置信度
    # print(detect_result.pred_instances.scores)
    # 框坐标：左上角X坐标、左上角Y坐标、右下角X坐标、右下角Y坐标
    # print(detect_result.pred_instances.bboxes[11]) # 模型判断正确

    # 置信度阈值过滤，获得最终目标检测预测结果
    # 置信度阈值
    CONF_THRES_NUM = 0.5
    CONF_THRES_PIN = 0.5
    # 存储num和pin实例
    num_list = []
    pin = pin_instance()

    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes_num = bboxes[np.logical_and(pred_instance.labels == 1, pred_instance.scores > CONF_THRES_NUM)]
    bboxes_pin = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > CONF_THRES_PIN)]  #
    # print(bboxes_pin) # [[373.9966     10.209503  409.3842    285.35934     0.6912988]]
    # label==1 为num label == 0 为pin
    bboxes_num = bboxes_num[nms(bboxes_num, 0.3)][:, :4].astype('int')
    bboxes_pin = bboxes_pin[nms(bboxes_pin, 0.3)][:, :4].astype('int')  # 贪婪搜索高置信度bbox并且overlap<thr
    # print(len(bboxes_pin)) # [[ 989  334 1073  478]] # 此处获得所有出现的bbox
    # 获取每个 bbox 的关键点预测结果
    pose_results_num = inference_topdown(pose_estimator, img_path, bboxes_num)
    pose_results_pin = inference_topdown(pose_estimator, img_path, bboxes_pin)
    # print(len(pose_results))  # 12
    # 把多个bbox的pose结果打包到一起
    data_samples_num = merge_data_samples(pose_results_num)
    data_samples_pin = merge_data_samples(pose_results_pin)
    # print(data_samples.keys()) # key值 ['_pred_heatmaps', 'pred_fields', 'gt_instances', 'pred_instances']

    # 预测结果-关键点坐标
    # keypoints = data_samples.pred_instances.keypoints.astype('int')
    # print(keypoints)
    # print(keypoints.shape)  # (12, 3, 2) 表示一共有12个label，每个label有3个关键点，每个关键点有2个坐标（此处是否逻辑出现问题）
    # 索引为 0 的框，每个关键点的坐标
    # print(keypoints[0, :, :])
    # 每一类关键点的预测热力图
    # kpt_idx = 1
    # heatmap = data_samples.pred_fields.heatmaps[kpt_idx, :, :]
    # # 索引为 idx 的关键点，在全图上的预测热力图
    # plt.imshow(heatmap)
    # plt.show()

    # Opencv可视化
    img_bgr = cv2.imread(img_path)
    img_height = img_bgr.shape[0]  # 图片高 y: 3648
    img_weight = img_bgr.shape[1]  # 图片宽 x: 5472
    # print(f'height : {img_height}, weight: {img_weight}')
    # 检测框的颜色
    bbox_color_num = (150, 0, 0)
    bbox_color_pin = (0, 155, 0)
    # 检测框的线宽
    bbox_thickness = 5
    # 关键点半径
    kpt_radius = 20
    # 连接线宽
    skeleton_thickness = 5
    # pin关键信息（直接从config配置文件中粘贴）
    dataset_info = {
        'dataset_name': 'Mr_215_Keypoint_coco',
        'classes': 'measurement',
        'paper_info': {
            'author': 'Magia',
            'title': 'Mr measurement',
            'container': 'OpenMMLab',
            'year': '2024',
            'homepage': ''
        },
        'keypoint_info': {
            0: {'name': 'pin_start', 'id': 0, 'color': [255, 0, 0], 'type': '', 'swap': ''},
            1: {'name': 'pin_end', 'id': 1, 'color': [0, 255, 0], 'type': '', 'swap': ''},
            2: {'name': 'num_value', 'id': 2, 'color': [0, 0, 255], 'type': '', 'swap': ''}  # 颜色为BGR
            # 0: {'name': 'num_value', 'id': 0, 'color': [0, 0, 255], 'type': '', 'swap': ''}
        },
        'skeleton_info': {
            0: {'link': ('pin_start', 'pin_end'), 'id': 0, 'color': [100, 150, 200]},
            # 1: {'link': ('pin_start', 'num_value'), 'id': 1, 'color': [100, 150, 0]},
        }
    }
    # 关键点类别和关键点ID的映射字典
    label2id = {}
    for each in dataset_info['keypoint_info'].items():
        label2id[each[1]['name']] = each[0]
    # # num的绘画
    if len(bboxes_num) == 0:
        print('No numbers')
        cv2.putText(img_bgr, 'No numbers', (0, 2000), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 7)
        return img_bgr
    for bbox_idx, bbox in enumerate(bboxes_num):  # 遍历每个检测框
        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color_num, bbox_thickness)
        # 此处为提取出的bbox

        # 索引为 0 的框，每个关键点的坐标
        keypoints = data_samples_num.pred_instances.keypoints[bbox_idx, :, :].astype('int')
        # print(keypoints)

        # 画连线
        # for skeleton_id, skeleton in dataset_info['skeleton_info'].items():  # 遍历每一种连接
        #     skeleton_color = skeleton['color']
        #     srt_kpt_id = label2id[skeleton['link'][0]]  # 起始点的类别 ID
        #     srt_kpt_xy = keypoints[srt_kpt_id]  # 起始点的 XY 坐标
        #     dst_kpt_id = label2id[skeleton['link'][1]]  # 终止点的类别 ID
        #     dst_kpt_xy = keypoints[dst_kpt_id]  # 终止点的 XY 坐标
        #     img_bgr = cv2.line(img_bgr, (srt_kpt_xy[0], srt_kpt_xy[1]), (dst_kpt_xy[0], dst_kpt_xy[1]),
        #                        color=skeleton_color, thickness=skeleton_thickness)

        # 画关键点
        for kpt_idx, kpt_xy in enumerate(keypoints):  # 遍历该检测框中的每一个关键点，
            if kpt_idx == label2id['num_value']:
                kpt_color = dataset_info['keypoint_info'][kpt_idx]['color']
                # img_bgr = cv2.circle(img_bgr, (kpt_xy[0], kpt_xy[1]), kpt_radius, kpt_color, -1)# 绘画关键点
                # TODO:此处记录kpt_xy的位置，做出以坐标点像素为核心的坐标轴，构建像素区间
                # print(bbox)
                num_temp = num_instance(bbox, kpt_xy, img_path)
                num_list.append(num_temp)
                break

    # pin 的绘画
    # if len(bboxes_pin) == 0:
    #     print('No pin')
    #     cv2.putText(img_bgr, 'No pin', (0, 2000), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 7)
    #     return img_bgr
    #
    # for bbox_idx, bbox in enumerate(bboxes_pin):  # 遍历每个检测框
    #
    #     # 画框
    #     img_bgr = cv2.rectangle(img_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color_pin, bbox_thickness)
    #
    #     # 索引为 0 的框，每个关键点的坐标
    #     keypoints = data_samples_pin.pred_instances.keypoints[bbox_idx, :, :].astype('int')
    #     # print(keypoints)
    #
    #     # 画连线
    #     for skeleton_id, skeleton in dataset_info['skeleton_info'].items():  # 遍历每一种连接
    #         skeleton_color = skeleton['color']
    #         srt_kpt_id = label2id[skeleton['link'][0]]  # 起始点的类别 ID
    #         srt_kpt_xy = keypoints[srt_kpt_id]  # 起始点的 XY 坐标
    #         dst_kpt_id = label2id[skeleton['link'][1]]  # 终止点的类别 ID
    #         dst_kpt_xy = keypoints[dst_kpt_id]  # 终止点的 XY 坐标
    #         img_bgr = cv2.line(img_bgr, (srt_kpt_xy[0], srt_kpt_xy[1]), (dst_kpt_xy[0], dst_kpt_xy[1]),
    #                            color=skeleton_color, thickness=skeleton_thickness)
    #
    #     # 画关键点
    #     for kpt_idx, kpt_xy in enumerate(keypoints):  # 遍历该检测框中的每一个关键点
    #         if kpt_idx == label2id['pin_start'] or kpt_idx == label2id['pin_end']:
    #             kpt_color = dataset_info['keypoint_info'][kpt_idx]['color']
    #             img_bgr = cv2.circle(img_bgr, (kpt_xy[0], kpt_xy[1]), kpt_radius, kpt_color, -1)
    #             pin.keypoints.append(kpt_xy)
    # 画出位图中心的两个关键点作为pin针
    point1 = (img_weight // 2, 0)
    point2 = (img_weight // 2, img_height // 3)
    pin.keypoints.append(point1)
    pin.keypoints.append(point2)
    # print(pin.keypoints[0])
    kpt_color = [255, 0, 0]
    img_bgr = cv2.circle(img_bgr, (point1[0], point1[1]), kpt_radius, kpt_color, -1)
    img_bgr = cv2.circle(img_bgr, (point2[0], point2[1] * 2), kpt_radius, kpt_color, -1)
    img_bgr = cv2.line(img_bgr, (point1[0], point1[1]), (point2[0], point2[1] * 2),
                       color=[100, 150, 200], thickness=skeleton_thickness)
    #
    # cv2.imwrite('data/outputs_1/G3_opencv.jpg', img_bgr)  # 此处需要自重写自己的
    # TODO: 此处经行以下步骤：
    # 1. 判断pin关键点落入的刻度区间
    point_left = (0, 0)
    point_right = (8000, 0)
    center = int(abs((pin.keypoints[0][0] + pin.keypoints[1][0]) / 2))  # 获取pin的横坐标中心
    for idx, num in enumerate(num_list):
        if center > num.keypoint[0]:  # 如果点在num右边
            if point_left[0] == 0:
                point_left = num.keypoint  # 初始化
            else:
                if num.keypoint[0] >= point_left[0]:
                    point_left = num.keypoint  # 如果num比目前记录point大，则更新最大点
        elif center < num.keypoint[0]:  # 如果点在num左边
            if point_right[0] == 8000:
                point_right = num.keypoint  # 右num初始化
            else:
                if num.keypoint[0] <= point_right[0]:
                    point_right = num.keypoint  # 如果num比目前记录point小，则更新最小点
        else:
            point_left = num.keypoint  # 在中心的情况
    # 2. 对左刻度图片进行OCR，识别出对应的刻度值,用于数值对比。
    model_value = 0
    actual_value = 0
    num_value_1 = 0
    for num in num_list:
        if num.keypoint[0] == point_left[0]:
            num_value_1 = first_digit(num.num_value())  # 此处需要对OCR识别内容做清洗
            print(f'OCR 识别结果:{num.num_value()}')  # 查看OCR识别的数字内容
    line_pin = [pin.keypoints[0][0], pin.keypoints[0][1], pin.keypoints[1][0], pin.keypoints[1][1]]
    line_num = [point_left[0], point_left[1], point_right[0], point_right[1]]
    x, y = get_line_cross_point(line_pin, line_num)
    # print(f'交点坐标为（{x}, {y}）')
    if point_left[0] <= x <= point_right[0]:  # 如果交点在区间内
        # 4. 计算交点得到结果，并计算其在坐标点上的占比，得出结果
        len_1 = math.sqrt((x - point_left[0]) ** 2 + (y - point_left[1]) ** 2)
        len_2 = math.sqrt((point_left[0] - point_right[0]) ** 2 + (point_left[1] - point_right[1]) ** 2)
        percent = len_1 / len_2
        # print(percent)
        num_value_1 = num_value_1 + percent
        model_value = num_value_1
        num_str = '%.2f' % num_value_1
    else:
        num_str = 'Pls set pin vertical'




    # 2. 对左刻度图片进行OCR，识别出对应的刻度值。
    num_value_1 = 0
    for num in num_list:
        if num.keypoint[0] == point_left[0]:
            num_value_1 = first_digit(num.num_value())  # 此处需要对OCR识别内容做清洗
            # print(f'OCR 识别结果：{num.num_value()}')  # 查看OCR识别的数字内容
            path, bbox = num.return_path()
            point_left = component(path)  # OPENCV读数
            point_left = (point_left[0] * 2 + bbox[0], point_left[1] * 2 + bbox[1])
        elif num.keypoint[0] == point_right[0]:
            point_right = component(num.return_path()[0])  # OPENCV读数
            path, bbox = num.return_path()
            point_right = component(path)  # OPENCV读数
            point_right = (point_right[0]*2 + bbox[0], point_right[1]*2 + bbox[1])

    # 3. 利用pin的两关键点和区间左右坐标的关键点构建分别构建两个截距式方程（注意判断0坐标）
    line_pin = [pin.keypoints[0][0], pin.keypoints[0][1], pin.keypoints[1][0], pin.keypoints[1][1]]
    line_num = [point_left[0], point_left[1], point_right[0], point_right[1]]
    x, y = get_line_cross_point(line_pin, line_num)
    # print(f'交点坐标为（{x}, {y}）')
    if point_left[0] <= x <= point_right[0]:  # 如果交点在区间内
        # 4. 计算交点得到结果，并计算其在坐标点上的占比，得出结果
        len_1 = math.sqrt((x - point_left[0]) ** 2 + (y - point_left[1]) ** 2)
        len_2 = math.sqrt((point_left[0] - point_right[0]) ** 2 + (point_left[1] - point_right[1]) ** 2)
        percent = len_1 / len_2
        # print(percent)
        num_value_1 = num_value_1 + percent
        actual_value = num_value_1
        num_str = '%.2f' % num_value_1
    else:
        num_str = 'Pls set pin vertical'
    # print('读数结果为： {}'.format(num_str))
    # 4.5 两个插值对比
    error = '%.3f' % abs(model_value-actual_value)
    temp = '%.3f' % model_value
    # 5. 将结果显示并绘制在图片上
    cv2.putText(img_bgr, f'value:{temp},error:{error}', (0, 1500), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 7)
    cv2.circle(img_bgr, point_left, kpt_radius, kpt_color, -1)
    cv2.circle(img_bgr, point_right, kpt_radius, kpt_color, -1)
    return img_bgr


def generate_video(input_path='demo/Demo.mkv'):
    filehead = input_path.split('/')[-1]
    output_path = "out-" + filehead

    print('视频开始处理', input_path)

    # 获取视频总帧数
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while (cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)

    # cv2.namedWindow('Crack Detection and Measurement Video Processing')
    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))
    # 有 GPU 就用 GPU，没有就用 CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('device', device)
    detector = init_detector(
        'data/rtmdet_tiny_triangle.py',
        'checkpoint/rtmdet_tiny_measurement_epoch_191_resize.pth',
        device=device
    )
    pose_estimator = init_pose_estimator(
        'data/rtmpose-s-triangle.py',
        'checkpoint/rtmpose_m_measurement_epoch_50_resize.pth',
        device=device,
        cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}}
    )
    # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while (cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break

                # 处理帧
                frame_path = './temp_frame.png'
                cv2.imwrite(frame_path, frame)
                try:
                    frame = pic_predic(frame_path, detector, pose_estimator)
                except Exception as error:
                    print('报错！', error)
                    pass

                if success == True:
                    # cv2.imshow('Video Processing', frame)
                    out.write(frame)

                    # 进度条更新一帧
                    pbar.update(1)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                # break
        except:
            print('中途中断')
            pass

    # cv2.destroyAllWindows()
    out.release()
    cap.release()
    print('视频已保存', output_path)


# 无pin针长交点脚本
if __name__ == '__main__':
    #生成图像
    # video_path = 'demo/Demo.mkv'
    # generate_video(video_path)

    # 生成图片
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    detector = init_detector(
        'data/rtmdet_tiny_triangle.py',
        'checkpoint/rtmdet_tiny_measurement_epoch_191_resize.pth',
        device=device
    )
    pose_estimator = init_pose_estimator(
        'data/rtmpose-s-triangle.py',
        'checkpoint/rtmpose_m_measurement_epoch_50_resize.pth',
        device=device,
        cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}}
    )

    frame_path = r'D:\AI_fileCollected\Datasets\Measurement\test_Image_2_CV/test_500_20240904134740995.jpg'  # 图片路径
    frame = pic_predic(frame_path, detector, pose_estimator)
    cv2.imwrite('temp_frame.png', frame)

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
