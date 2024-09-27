import os

import cv2
import numpy as np
import pandas as pd
import math

from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import matplotlib

# 创建 图表 文件夹，用于存放图表
if not os.path.exists('图表'):
    os.mkdir('图表')
    print('创建空文件夹 图表')

matplotlib.rc("font",family='SimHei') # 中文字体
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

# plt.plot([1,2,3], [100,500,300])
# plt.title('matplotlib中文字体测试', fontsize=25)
# plt.xlabel('X轴', fontsize=15)
# plt.ylabel('Y轴', fontsize=15)
# plt.show()
folder_path = r'D:\AI_fileCollected\code\Measurement\Measurement\Datasets\Measurement\Image'

N = 16 # 可视化图像的个数
# # n 行 n 列
# n = math.floor(np.sqrt(N))
#
# # 读取文件夹中的所有图像
# images = []
# for each_img in os.listdir(folder_path)[:N]:
#     img_path = os.path.join(folder_path, each_img)
#     img_bgr = cv2.imread(img_path)
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     images.append(img_rgb)
#
# # 画图
# fig = plt.figure(figsize=(10, 10))
# grid = ImageGrid(fig, 111,  # 类似绘制子图 subplot(111)
#                  nrows_ncols=(n, n),  # 创建 n 行 m 列的 axes 网格
#                  axes_pad=0.02,  # 网格间距
#                  share_all=True
#                  )
#
# # 遍历每张图像
# for ax, im in zip(grid, images):
#     ax.imshow(im)
#     ax.axis('off')
#
# plt.tight_layout()
#
# plt.savefig('图表/图像-一些图像.pdf', dpi=120, bbox_inches='tight')
#
# plt.show()

df = pd.read_csv(r'D:\AI_fileCollected\code\Measurement\Measurement\Datasets\Measurement\kpt_dataset_eda.csv')
# 图像个数
print('图像个数：{}'.format(len(df['imagePath'].unique())))
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

x = df['imageWidth']
y = df['imageHeight']

xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

plt.figure(figsize=(10,10))
plt.scatter(x, y, c=z,  s=5, cmap='Spectral_r')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])

plt.tick_params(labelsize=15)

xy_max = max(max(df['imageWidth']), max(df['imageHeight']))
plt.xlim(xmin=0, xmax=xy_max)
plt.ylim(ymin=0, ymax=xy_max)

plt.ylabel('height', fontsize=25)
plt.xlabel('width', fontsize=25)

plt.savefig('图表/图像-图像宽高尺寸分布.pdf', dpi=120, bbox_inches='tight')

plt.show()

df_num = pd.DataFrame()
label_type_list = []
num_list = []
for each in df['label_type'].unique():
    label_type_list.append(each)
    num_list.append(len(df[df['label_type'] == each]))

df_num['label_type'] = label_type_list
df_num['num'] = num_list

df_num = df_num.sort_values(by='num', ascending=False)
plt.figure(figsize=(22, 7))

x = df_num['label_type']
y = df_num['num']

plt.bar(x, y, facecolor='#1f77b4', edgecolor='k')

plt.xticks(rotation=90)
plt.tick_params(labelsize=15)
plt.xlabel('标注类别', fontsize=20)
plt.ylabel('图像数量', fontsize=20)

plt.savefig('图表/图像-各标注种类个数.pdf', dpi=120, bbox_inches='tight')

plt.show()
df_num = pd.DataFrame()
label_type_list = []
num_list = []
for each in df['imagePath'].unique():
    label_type_list.append(each)
    num_list.append(len(df[df['imagePath'] == each]))

df_num['label_type'] = label_type_list
df_num['num'] = num_list

df_num = df_num.sort_values(by='num', ascending=False)
plt.figure(figsize=(22, 10))

x = df_num['label_type']
y = df_num['num']

plt.bar(x, y, facecolor='#1f77b4', edgecolor='k')

plt.xticks(rotation=90)
plt.tick_params(labelsize=8)
plt.xlabel('图像路径', fontsize=20)
plt.ylabel('标注个数', fontsize=20)

plt.savefig('图表/图像-不同图像的标注个数.pdf', dpi=120, bbox_inches='tight')

plt.show()
df_box = df[df['label_type']=='rectangle']
df_box = df_box.reset_index(drop=True)
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

x = df_box['bbox_center_x_norm']
y = df_box['bbox_center_y_norm']

xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

plt.figure(figsize=(7,7))
plt.scatter(x, y,c=z,  s=3,cmap='Spectral_r')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
plt.tick_params(labelsize=15)
plt.xlim(0,1)
plt.ylim(0,1)

plt.xlabel('x_center', fontsize=25)
plt.ylabel('y_center', fontsize=25)

plt.savefig('图表/框标注-框中心点位置分布.pdf', dpi=120, bbox_inches='tight')

plt.show()
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

x = df_box['bbox_width_norm']
y = df_box['bbox_height_norm']

xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

plt.figure(figsize=(7,7))
# plt.figure(figsize=(12,12))
plt.scatter(x, y,c=z,  s=1,cmap='Spectral_r')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])

plt.tick_params(labelsize=15)
plt.xlim(0,1.02)
plt.ylim(0,1.015)

plt.xlabel('width', fontsize=25)
plt.ylabel('height', fontsize=25)

plt.savefig('图表/框标注-框宽高分布.pdf', dpi=120, bbox_inches='tight')

plt.show()
# 二选一运行
df_point = df[df['label_type']=='point']    # 所有关键点
# df_point = df[df['label']=='angle_30']      # 指定一类关键点
df_point = df_point.reset_index(drop=True)
df_num = pd.DataFrame()
label_type_list = []
num_list = []
for each in df_point['label'].unique():
    label_type_list.append(each)
    num_list.append(len(df_point['label'] == each))

df_num['label_type'] = label_type_list
df_num['num'] = num_list

df_num = df_num.sort_values(by='num', ascending=False)
plt.figure(figsize=(22, 7))

x = df_num['label_type']
y = df_num['num']

plt.bar(x, y, facecolor='#1f77b4', edgecolor='k')

plt.xticks(rotation=90)
plt.tick_params(labelsize=15)
plt.xlabel('标注类别', fontsize=20)
plt.ylabel('图像数量', fontsize=20)

plt.savefig('图表/关键点标注-不同类别点的标注个数.pdf', dpi=120, bbox_inches='tight')

plt.show()
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

# 所有关键点
x = df_point['kpt_x_norm']
y = df_point['kpt_y_norm']

xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

plt.figure(figsize=(7,7))
plt.scatter(x, y,c=z,  s=3,cmap='Spectral_r')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
plt.tick_params(labelsize=15)
plt.xlim(0,1)
plt.ylim(0,1)

plt.xlabel('kpt_x', fontsize=25)
plt.ylabel('kpt_y', fontsize=25)

plt.savefig('图表/关键点标注-关键点位置分布.pdf', dpi=120, bbox_inches='tight')

plt.show()