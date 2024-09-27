
import cv2
import numpy as np
import time
from tqdm import tqdm
# 该whl不太支持GPU运行，用CPU对单张图片的推理同样也比较快
from mmdeploy_runtime import Detector
from mmdeploy_runtime import PoseDetector

import matplotlib.pyplot as plt


def process_frame(img_bgr, bbox_detector, pose_detector):
    # 框（rectangle）可视化配置
    bbox_label_num = 'num'  # 框的类别名称
    bbox_label_pin = 'pin'
    bbox_color_num = (255, 129, 0)  # 框的 BGR 颜色
    bbox_color_pin = (255, 10, 0)
    bbox_thickness = 5  # 框的线宽

    # 框类别文字
    bbox_labelstr = {
        'font_size': 1,  # 字体大小
        'font_thickness': 1,  # 字体粗细
        'offset_x': 0,  # X 方向，文字偏移距离，向右为正
        'offset_y': 0,  # Y 方向，文字偏移距离，向下为正
    }

    # 关键点 BGR 配色
    kpt_color_map = {
        0: {'name': 'pin_start', 'color': [0, 255, 0], 'radius': 5},  # 60度角点
        1: {'name': 'pin_end', 'color': [0, 0, 255], 'radius': 5},  # 90度角点
        2: {'name': 'num', 'color': [255, 0, 0], 'radius': 5},  # 30度角点
    }

    # 关键点类别文字
    kpt_labelstr = {
        'font_size': 1,  # 字体大小
        'font_thickness': 1,  # 字体粗细
        'offset_x': 0,  # X 方向，文字偏移距离，向右为正
        'offset_y': 0,  # Y 方向，文字偏移距离，向下为正
    }

    # 骨架连接 BGR 配色
    skeleton_map = [
        {'srt_kpt_id': 0, 'dst_kpt_id': 1, 'color': [47, 255, 173], 'thickness': 2},  # 60度角点-90度角点
    ]
    # 目标检测推理预测
    bboxes, labels, _ = bbox_detector(img_bgr)
    bboxes_num = []
    bboxes_pin = []
    for idx, label in enumerate(labels):
        if label == 1:
            bboxes_num.append(bboxes[idx])
        else:
            bboxes_pin.append(bboxes[idx])

    # 置信度阈值过滤
    bboxes_num = np.array(bboxes_num)
    bboxes_pin = np.array(bboxes_pin) # 此处需要把输出结果转换成numpy经行计算
    bboxes_num = bboxes_num[bboxes_num[:, -1] > 0.55]
    bboxes_pin = bboxes_pin[bboxes_pin[:, -1] > 0.55]
    # 获取整数坐标
    bboxes_num = bboxes_num[:, :4].astype(np.int32)
    bboxes_pin = bboxes_pin[:, :4].astype(np.int32)

    keypoints_num = pose_detector(img_bgr, bboxes_num)[:, :, :2].astype(np.int32)
    keypoints_pin = pose_detector(img_bgr, bboxes_pin)[:, :, :2].astype(np.int32)
    # print(keypoints)
    num_bbox_num = len(bboxes_num)
    num_bbox_pin = len(bboxes_pin)

    for idx in range(num_bbox_num):  # 遍历每个框

        # 获取该框坐标
        bbox_xyxy = bboxes_num[idx]

        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color_num,
                                bbox_thickness)

        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label_num,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color_num,
                              bbox_labelstr['font_thickness'])

        bbox_keypoints = keypoints_num[idx]  # 该框所有关键点坐标和置信度

        # 画该框的关键点
        for kpt_id in kpt_color_map:
            # 获取该关键点的颜色、半径、XY坐标
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]
            if kpt_id == 2:
                # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
                img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

                # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
                kpt_label = ''  # 写关键点类别 ID（二选一）
                # kpt_label = str(kpt_color_map[kpt_id]['name']) # 写关键点类别名称（二选一）
                img_bgr = cv2.putText(img_bgr, kpt_label,
                                      (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
                                      cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color,
                                      kpt_labelstr['font_thickness'])

    for idx in range(num_bbox_pin):  # 遍历每个框

        # 获取该框坐标
        bbox_xyxy = bboxes_pin[idx]

        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color_pin,
                                bbox_thickness)

        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label_pin,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color_pin,
                              bbox_labelstr['font_thickness'])

        bbox_keypoints = keypoints_pin[idx]  # 该框所有关键点坐标和置信度

        # 画该框的骨架连接
        for skeleton in skeleton_map:
            # 获取起始点坐标
            srt_kpt_id = skeleton['srt_kpt_id']
            srt_kpt_x = bbox_keypoints[srt_kpt_id][0]
            srt_kpt_y = bbox_keypoints[srt_kpt_id][1]

            # 获取终止点坐标
            dst_kpt_id = skeleton['dst_kpt_id']
            dst_kpt_x = bbox_keypoints[dst_kpt_id][0]
            dst_kpt_y = bbox_keypoints[dst_kpt_id][1]

            # 获取骨架连接颜色
            skeleton_color = skeleton['color']

            # 获取骨架连接线宽
            skeleton_thickness = skeleton['thickness']

            # 画骨架连接
            img_bgr = cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color,
                               thickness=skeleton_thickness)

        # 画该框的关键点
        for kpt_id in kpt_color_map:
            # 获取该关键点的颜色、半径、XY坐标
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]
            if not kpt_id == 2:
                # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
                img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

                # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
                kpt_label = ''  # 写关键点类别 ID（二选一）
                # kpt_label = str(kpt_color_map[kpt_id]['name']) # 写关键点类别名称（二选一）
                img_bgr = cv2.putText(img_bgr, kpt_label,
                                      (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
                                      cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color,
                                      kpt_labelstr['font_thickness'])

    # plt.imshow(img_bgr[:, :, ::-1])
    # plt.show()
    return img_bgr


# 视频逐帧处理代码模板
# 不需修改任何代码，只需定义process_frame函数即可
# 同济子豪兄 2021-7-10

def generate_video(input_path='Demo.mkv'):
    filehead = input_path.split('/')[-1]
    output_path = "out-1" + filehead
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

    # img_path = '../mmdetection/data/pictures/1_Color.png'
    # img_bgr = cv2.imread(img_path)
    # 目标检测模型目录
    detect = '../rtmdet2onnx'

    # 关键点检测模型目录
    pose = '../rtmpose2onnx'

    # 计算设备
    device = 'cpu'
    # device = 'cuda'

    bbox_detector = Detector(detect, device)
    pose_detector = PoseDetector(pose, device)

    # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while (cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break
                # 处理帧
                # frame_path = './temp_frame.png'
                # cv2.imwrite(frame_path, frame)
                try:
                    frame = process_frame(frame, bbox_detector, pose_detector)
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

    out.release()
    cap.release()
    print('视频已保存', output_path)


if __name__ == '__main__':
    video_path = 'Demo.mkv'
    generate_video(video_path)
