# 划分训练集和测试集
# _2表示适用与钢尺版本的转换文件
import os
import shutil
import random

from tqdm import tqdm

class_list_1 = {
    'supercategory': 'single_digit',
    'id': 0,
    'name': 'single_digit',
    'keypoints': ['num_value'],  # 大小写敏感
}
class_list_2 = {
    'supercategory': 'ten_digit',
    'id': 1,
    'name': 'ten_digit',
    'keypoints': ['num_value']  # 大小写敏感
}
coco = {}

coco['categories'] = []
coco['categories'].append(class_list_1)
coco['categories'].append(class_list_2)
coco['images'] = []
coco['annotations'] = []

IMG_ID = 0
ANN_ID = 0


def dataSeperate(Dataset_root = r'D:\AI_fileCollected\code\Measurement\Measurement\Datasets\Measurement'):
    os.chdir(os.path.join(Dataset_root, 'annotations'))
    print('共有 {} 个 labelme 格式的 json 文件'.format(len(os.listdir())))
    test_frac = 0.1  # 测试集比例
    random.seed(123)  # 随机数种子，便于复现
    folder = '.'
    img_paths = os.listdir(folder)
    random.shuffle(img_paths)  # 随机打乱

    val_number = int(len(img_paths) * test_frac)  # 测试集文件个数
    train_files = img_paths[val_number:]  # 训练集文件名列表
    val_files = img_paths[:val_number]  # 测试集文件名列表

    print('数据集文件总数', len(img_paths))
    print('训练集文件个数', len(train_files))
    print('测试集文件个数', len(val_files))
    # 创建文件夹，存放训练集的 labelme格式的 json 标注文件
    train_labelme_jsons_folder = 'train_labelme_jsons'
    os.mkdir(train_labelme_jsons_folder)
    for each in tqdm(train_files):
        src_path = os.path.join(folder, each)
        dst_path = os.path.join(train_labelme_jsons_folder, each)
        shutil.move(src_path, dst_path)
    # 创建文件夹，存放训练集的 labelme格式的 json 标注文件
    val_labelme_jsons_folder = 'val_labelme_jsons'
    os.mkdir(val_labelme_jsons_folder)
    for each in tqdm(val_files):
        src_path = os.path.join(folder, each)
        dst_path = os.path.join(val_labelme_jsons_folder, each)
        shutil.move(src_path, dst_path)
    print(len(os.listdir(train_labelme_jsons_folder)) + len(os.listdir(val_labelme_jsons_folder)))


def label2coco(Dataset_root = r'D:\AI_fileCollected\code\Measurement\Measurement\mmdetection\data\Measurement', indx=1):
    def process_single_json(labelme, image_id=1):
        '''
        输入labelme的json数据，输出coco格式的每个框的关键点标注信息
        '''

        global ANN_ID, coco_path

        coco_annotations = []
        for each_ann in labelme['shapes']:  # 遍历该json文件中的所有标注

            if each_ann['shape_type'] == 'rectangle':  # 筛选出个体框

                # 个体框元数据
                bbox_dict = {}
                if each_ann['label'] == 'single_digit':
                    bbox_dict['category_id'] = 0  # 此处需要更改ID
                elif each_ann['label'] == 'ten_digit':
                    bbox_dict['category_id'] = 1 # 此处需要更改ID
                bbox_dict['segmentation'] = []

                bbox_dict['iscrowd'] = 0
                bbox_dict['segmentation'] = []

                bbox_dict['image_id'] = image_id
                # print(bbox_dict['image_id'])
                bbox_dict['id'] = ANN_ID
                # print(ANN_ID)
                ANN_ID += 1

                # 获取个体框坐标
                bbox_left_top_x = min(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
                bbox_left_top_y = min(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
                bbox_right_bottom_x = max(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
                bbox_right_bottom_y = max(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
                bbox_w = bbox_right_bottom_x - bbox_left_top_x
                bbox_h = bbox_right_bottom_y - bbox_left_top_y
                bbox_dict['bbox'] = [bbox_left_top_x, bbox_left_top_y, bbox_w, bbox_h]  # 左上角x、y、框的w、h
                bbox_dict['area'] = bbox_w * bbox_h

                # 筛选出分割多段线
                for each_ann in labelme['shapes']:  # 遍历所有标注
                    if each_ann['shape_type'] == 'polygon':  # 筛选出分割多段线标注
                        # 第一个点的坐标
                        first_x = each_ann['points'][0][0]
                        first_y = each_ann['points'][0][1]
                        if (first_x > bbox_left_top_x) & (first_x < bbox_right_bottom_x) & (
                                first_y < bbox_right_bottom_y) & (first_y > bbox_left_top_y):  # 筛选出在该个体框中的关键点
                            bbox_dict['segmentation'] = list(
                                map(lambda x: list(map(lambda y: round(y, 2), x)), each_ann['points']))  # 坐标保留两位小数
                            # bbox_dict['segmentation'] = each_ann['points']

                # 筛选出该个体框中的所有关键点
                bbox_keypoints_dict = {}
                for each_ann in labelme['shapes']:  # 遍历所有标注

                    if each_ann['shape_type'] == 'point':  # 筛选出关键点标注
                        # 关键点横纵坐标
                        x = int(each_ann['points'][0][0])
                        y = int(each_ann['points'][0][1])
                        label = each_ann['label']

                        if (x > bbox_left_top_x) & (x < bbox_right_bottom_x) & (y < bbox_right_bottom_y) & (
                                y > bbox_left_top_y):  # 筛选出在该个体框中的关键点
                            bbox_keypoints_dict[label] = [x, y]  # 该值有3 中lable

                bbox_dict['num_keypoints'] = len(bbox_keypoints_dict)
                # print(len(bbox_keypoints_dict))

                # 把关键点按照类别顺序排好，注意此类需要自己定义,重复添加了1此
                bbox_dict['keypoints'] = []
                for each_class in class_list_1['keypoints']:
                    if each_class in bbox_keypoints_dict: # 确实在
                        bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0])
                        bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1])
                        bbox_dict['keypoints'].append(2)  # 2-可见不遮挡 1-遮挡 0-没有点

                    else:  # 不存在的点，一律为0
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)
                        bbox_dict['keypoints'].append(0)


                coco_annotations.append(bbox_dict)
                # print(bbox_dict['image_id'])
        # print(coco_annotations[0]['image_id'])
        return coco_annotations

    def process_folder():
        image_id = 0

        # 遍历所有 labelme 格式的 json 文件
        for labelme_json in os.listdir():

            if labelme_json.split('.')[-1] == 'json':

                with open(labelme_json, 'r', encoding='utf-8') as f:

                    labelme = json.load(f)

                    ## 提取图像元数据
                    img_dict = {}
                    img_dict['file_name'] = labelme['imagePath']
                    img_dict['height'] = labelme['imageHeight']
                    img_dict['width'] = labelme['imageWidth']
                    img_dict['id'] = image_id
                    coco['images'].append(img_dict)
                    ## 提取框和关键点信息
                    # print(image_id)
                    coco_annotations = process_single_json(labelme, image_id=image_id)
                    coco['annotations'] += coco_annotations

                    image_id += 1

                    print(labelme_json, '已处理完毕')

            else:
                pass

    import os
    import json
    import numpy as np

    coco['info'] = {}
    coco['info']['description'] = 'This is a MR measurement_test_ver'
    if indx == 1:
        path = os.path.join(Dataset_root, 'annotations', 'train_labelme_jsons')
        os.chdir(path)
        process_folder()
        # 保存coco标注文件
        coco_path = '../../train_coco.json'
        with open(coco_path, 'w') as f:
            json.dump(coco, f, indent=2)
        os.chdir('../../')
    elif indx == 2:
        path = os.path.join(Dataset_root, 'annotations', 'val_labelme_jsons')
        os.chdir(path)
        process_folder()
        # 保存coco标注文件
        coco_path = '../../val_coco.json'
        with open(coco_path, 'w') as f:
            json.dump(coco, f, indent=2)
        os.chdir('../../')


# 批量转换成coco数据集
if __name__ == '__main__':
    # dataSeperate(r'D:\AI_fileCollected\Datasets\Measurement\formal_Images')
    label2coco(r'D:\AI_fileCollected\Datasets\Measurement\formal_Images',2) # 1 为训练集 2 为测试机
    # # 得到完整的MS COCO格式的数据集
    from pycocotools.coco import COCO
    # my_coco = COCO(r'D:\AI_fileCollected\Datasets\Measurement\formal_Images\train_coco.json')
    # my_coco = COCO(r'D:\AI_fileCollected\Datasets\Measurement\formal_Images\val_coco.json')