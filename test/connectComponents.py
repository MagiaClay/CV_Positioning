import cv2 as cv
import numpy as np
from paddleocr import PaddleOCR


def OCRtest(path):
    def first_digit(string):
        num_list_new = []  # 新建空列表，用以存储提取的数值
        a = ''  # 将空值赋值给a
        # print(string)
        for i in string:  # 将字符串进行遍历
            if i.isdigit():  # 判断i是否为数字，如果“是”返回True，“不是”返回False
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
    OCR_model = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, show_log=False)  # 初始化OCR模型，全局变量
    result = OCR_model.ocr(path, cls=True)
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
        return '0'
        pass
    else:
        return first_digit(result)  # 为识别的字符串


# 返回刻度读数
def remove_glare(img):
    # 读取图像
    img = img

    height, width = img.shape[:2]
    # print("height: ", height)
    # print("width: ", width)

    # 转换为灰度图像
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 使用自适应阈值分割图像
    _, thresh = cv.threshold(img, 160, 255, 0)

    # 定义矩形的顶点坐标
    # start_point = (2, height // 2 - 100)
    # end_point = (width - 2, height // 2 + 100)
    start_point = (0, 0)
    end_point = (width, height)
    # rect = cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
    # cv2.imshow('rectangle', rect)
    # cv2.waitKey(0)

    # 使用 rectangle() 函数在图像上绘制矩形
    img_rectangle = cv.rectangle(thresh.copy(), start_point, end_point, (0, 0, 0), -1)
    # img_rectangle = cv2.rectangle(img, start_point, end_point, (0, 255, 0), 3)

    # 使用 bitwise_xor() 函数保留指定区域内的原始图像
    thresh = cv.bitwise_xor(thresh, img_rectangle)

    # cv2.imshow('threshold Image', thresh)
    # cv2.waitKey(0)

    cnts, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(img, cnts, -1, (0, 0, 0), -1)
    # cv2.imshow('find reflection', img)
    # cv2.waitKey(0)
    # cv.imshow("colored labels", img)
    # cv.imwrite("labels.png", image)
    # print("total componets : ", num_labels - 1)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img


def component(path='../mmpose/temp_sip.png'):
    src = cv.imread(path)
    # src = cv.resize(src, dsize=None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
    src = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = remove_glare(gray) # 在降噪处理之后
    cv.imwrite('temp_sip.png', gray)
    result = OCRtest('temp_sip.png')
    print(f'OCR的识别结果为：{result}')
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 去掉噪声，例如很小或者很大的噪声点
    # cv.imshow("binary", binary)
    # cv.imwrite('binary.png', binary)
    # 使用开运算去掉外部的噪声 # 不需要去噪
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    # cv.imshow("binary", binary)

    output = cv.connectedComponents(binary, connectivity=8, ltype=cv.CV_32S)
    # num_labels, labels, stats, centers = cv.connectedComponentsWithStats(binary, connectivity=8, ltype=cv.CV_32S)
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
            elif labels[row, col] == 0:
                image[row, col] = (255, 255, 255)  # 绘制对应的颜色
    for row in range(h):
        for col in range(w):
            if row == max_height:
                max_height_pixels.append(col)
    if len(max_height_pixels) > 2:
        average_col = int(sum(max_height_pixels) / len(max_height_pixels))
        point = (average_col, min_height)
        # print(f'目前的关键点为{point}')
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
    max_y = max_height
    x = min_x + int(abs(max_x - min_x) / 2)
    # point = (x, min_height)
    point = (x, max_y)

    image = cv.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 1)
    image = cv.circle(image, point, 1, (0, 0, 255), 2, 8, 0)

    # image = cv.circle(image, (point[0], max_height), 1, (0, 0, 255), 2, 8, 0)
    # image = cv.circle(image, point, 1, (0, 0, 255), 2, 8, 0)

    # # 绘制矩形和几何中心
    # for t in range(1, num_labels, 1):
    #     x, y, w, h, area = stats[t]
    #     cx, cy = centers[t]
    #     # 标出中心位置
    #     cv.circle(image, (np.int32(cx), np.int32(cy)), 2, (0, 255, 0), 2, 8, 0)
    #     # 画出外接矩形
    #     cv.rectangle(image, (x, y), (x + w, y + h), colors[t], 1, 8, 0)
    #     # cv.putText(image, "No." + str(t), (x, y), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1);
    #     print("label index %d, area of the label : %d" % (t, area))
    #
    #     # cv.imshow("colored labels", image)
    #     # cv.imwrite("labels.png", image)
    #     print("total number : ", num_labels - 1)

    # 需要知道矩形的长和宽来构建矩形

    cv.imshow("colored labels", image)
    # cv.imwrite("labels.png", image)
    # print("total componets : ", num_labels - 1)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # print(point)
    return point


if __name__ == '__main__':
    component(r'D:\AI_fileCollected\code\Measurement\Measurement\mmpose\temp_sip.png')
    # remove_glare(r'C:\Users\z004xzbw\Pictures/number_3.png')
    # print(OCRtest(r'C:\Users\z004xzbw\Pictures/number_10.png'))

