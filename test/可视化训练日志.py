import os

os.chdir(r'D:\AI_fileCollected\code\Measurement\Measurement\mmdetection')
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 载入训练日志
# 日志文件路径
log_path = 'work_dirs/rtmdet_tiny_triangle/20240813_091917/vis_data/scalars.json'

# log_path = 'work_dirs/rtmdet_tiny_triangle/20230511_234855/vis_data/scalars.json'
with open(log_path, "r") as f:
    json_list = f.readlines()
print(len(json_list))
print(eval(json_list[4]))  # 损失以及评估
df_train = pd.DataFrame()
df_test = pd.DataFrame()
for each in tqdm(json_list):
    if 'coco/bbox_mAP' in each:
        df_test = df_test.append(eval(each), ignore_index=True)
    else:
        df_train = df_train.append(eval(each), ignore_index=True)
# 构成PD表
print(df_train)  # 训练数据
print(df_test)  # 测试数据
# 导出训练日志格式表
df_train.to_csv('训练日志-训练集.csv', index=False)
df_test.to_csv('训练日志-测试集.csv', index=False)

# 设置Matplotlib中文字体

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.plot([1, 2, 3], [100, 500, 300])
plt.title('matplotlib中文字体测试', fontsize=25)
plt.xlabel('X轴', fontsize=15)
plt.ylabel('Y轴', fontsize=15)
plt.show()  # 测试是否为中文字体

# 可视化辅助函数
from matplotlib import colors as mcolors
import random

random.seed(124)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick',
          'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen',
          'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray',
          'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue',
          'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid',
          'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred',
          'deeppink', 'hotpink']
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D",
           "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
linestyle = ['--', '-.', '-']


def get_line_arg():
    '''
    随机产生一种绘图线型
    '''
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    # line_arg['marker'] = random.choice(markers)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = random.randint(1, 4)
    # line_arg['markersize'] = random.randint(3, 5)
    return line_arg


print(df_train.columns)
# output: Index(['base_lr', 'lr', 'data_time', 'loss', 'loss_cls', 'loss_bbox', 'time',
#        'epoch', 'iter', 'memory', 'step'],
#       dtype='object')

metrics = ['loss', 'loss_bbox', 'loss_cls', 'lr']  # 定义损失函数
plt.figure(figsize=(16, 8))

x = df_train['step']
for y in metrics:
    try:
        plt.plot(x, df_train[y], label=y, **get_line_arg())
    except:
        pass

plt.tick_params(labelsize=20)
plt.xlabel('step', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.title('训练集损失函数', fontsize=25)
plt.savefig('训练集损失函数.pdf', dpi=120, bbox_inches='tight')

plt.legend(fontsize=20)

plt.show()
# 研究训练准确率
# metrics = ['acc']
# plt.figure(figsize=(16, 8))
#
# x = df_train['step']
# for y in metrics:
#     try:
#         plt.plot(x, df_train[y], label=y, **get_line_arg())
#     except:
#         pass
#
# plt.tick_params(labelsize=20)
# plt.xlabel('step', fontsize=20)
# plt.ylabel('loss', fontsize=20)
# plt.title('训练集准确率', fontsize=25)
# plt.savefig('训练集准确率.pdf', dpi=120, bbox_inches='tight')
#
# plt.legend(fontsize=20)
#
# plt.show()

# 测试机评估指标-Ms coco mETRIC
print(df_test.columns) # 查看测试指标
metrics = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75', 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l'] # 设置需要的coco
plt.figure(figsize=(16, 8))

x = df_test['step']
for y in metrics:
    try:
        plt.plot(x, df_test[y], label=y, **get_line_arg())
    except:
        pass

plt.tick_params(labelsize=20)
# plt.ylim([0, 100])
plt.xlabel('Epoch', fontsize=20)
plt.ylabel(y, fontsize=20)
plt.title('测试集评估指标', fontsize=25)
plt.savefig('测试集分类评估指标.pdf', dpi=120, bbox_inches='tight')

plt.legend(fontsize=20)
plt.show()

# # 测试集评估指标-PASCAL VOC Metric（如果生成的图是空的，说明没有pascal voc指标，跳过本图即可）
# metrics = ['pascal_voc/mAP', 'pascal_voc/AP50']
# plt.figure(figsize=(16, 8))
#
# x = df_test['step']
# for y in metrics:
#     try:
#         plt.plot(x, df_test[y], label=y, **get_line_arg())
#     except:
#         pass
#
# plt.tick_params(labelsize=20)
# # plt.ylim([0, 100])
# plt.xlabel('Epoch', fontsize=20)
# plt.ylabel(y, fontsize=20)
# plt.title('测试集评估指标', fontsize=25)
# plt.savefig('测试集分类评估指标.pdf', dpi=120, bbox_inches='tight')
#
# plt.legend(fontsize=20)
#
# plt.show()
