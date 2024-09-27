#!/usr/bin/env python
# coding=utf-8

# from PyCameraList.camera_device import test_list_cameras, list_video_devices, list_audio_devices
#
# cameras = list_video_devices()
# print(dict(cameras))
# # return: {0: 'Intel(R) RealSense(TM) 3D Camera (Front F200) RGB', 1: 'NewTek NDI Video', 2: 'Intel(R) RealSense(TM) 3D Camera Virtual Driver', 3: 'Intel(R) RealSense(TM) 3D Camera (Front F200) Depth', 4: 'OBS-Camera', 5: 'OBS-Camera2', 6: 'OBS-Camera3', 7: 'OBS-Camera4', 8: 'OBS Virtual Camera'}
#
# audios = list_audio_devices()
# print(dict(audios))
# # return:  {0: '麦克风阵列 (Creative VF0800)', 1: 'OBS-Audio', 2: '线路 (NewTek NDI Audio)'}


from PyCameraList.camera_device import test_list_cameras, list_video_devices, list_audio_devices
import cv2 as cv


def show_in_cv(camera_id):
    cap = cv.VideoCapture(camera_id, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv.CAP_PROP_FPS, 30)
    i = 0
    while True:
        suc, frame = cap.read()
        if not suc:
            break
        cv.imshow("preview camera", frame)
        cv.waitKey(30)
        i += 1
    return


def main():
    cameras = list_video_devices()
    print(f'\n\n===========================\ncamera_list: {cameras}')
    idx = 0
    camera_id = cameras[idx][0]
    camera_name = cameras[idx][1]
    print(f'\n\n===========================\npreview camera: camera_id={camera_id} camera_name={camera_name}')
    show_in_cv(camera_id)


if __name__ == '__main__':
    main()
