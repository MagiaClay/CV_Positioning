import os
import cv2
import numpy as np
from tqdm import tqdm


def generate_video(path=r'D:\AI_fileCollected\Datasets\Helmet_evalue\Images_56/'):
    output_path = r"D:\AI_fileCollected\Datasets\Measurement\formal_Images\images/measurement.mkv"
    filelist = os.listdir(path)
    fps = 20.0  # 视频每秒24帧
    # 可以使用cv2.resize()进行修改
    frame_count = 0
    for item in filelist:
        if item.endswith('.jpg'):
            # 找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
            frame_count += 1

    frame_size = (5472, 3648) # 要与实际frame大小一致

    fourcc = cv2.VideoWriter.fourcc(*'mp4v') # 此处函数遭到废弃
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))
    # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:
        try:
            for item in filelist:
                if item.endswith('.jpg'):
                    # 找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
                    item = path + item
                    img = cv2.imread(item)
                    out.write(img)
                    pbar.update(1)

        except:
            print('中途中断')
            pass

    # cv2.destroyAllWindows()
    out.release()
    print('视频已保存', output_path)


if __name__ == '__main__':
    generate_video(r'D:\AI_fileCollected\Datasets\Measurement\formal_Images\images/')
