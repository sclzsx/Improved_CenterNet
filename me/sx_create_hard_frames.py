from pathlib import Path
import cv2
import os
import random

vid_root = '/workspace/DATA/zhatu/videos/0825'
save_dir = vid_root + '/hard_frames'
if save_dir is not None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

fff = []
for path in Path(vid_root).glob('*.*'):
    cap = cv2.VideoCapture(str(path))
    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f = []
    for idx in range(0, total_frame_num - 1):
        _, frame = cap.read()
        f.append(frame)
    random.shuffle(f)
    for i, ff in enumerate(f):
        fff.append(ff)
        if i > 9:
            break
random.shuffle(fff)

hard_frames = fff[:10]
for i, ffff in enumerate(hard_frames):
    cv2.imwrite(save_dir + '/' + str(i) + '.jpg', ffff)
