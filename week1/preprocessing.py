import numpy as np
import cv2
from PIL import Image
import torch
# 데이터 정규화
def data_normalize(train_ds,val_ds):
    train_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in train_ds]  # 이미지의 높이와 너비를 기준으로 평균을 구한다 -> RGB채널별로 이미지의 각 픽셀에 대한 평균이 계산된다.
    train_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in train_ds]
    train_meanR=np.mean([m[0] for m in train_meanRGB])
    train_meanG=np.mean([m[1] for m in train_meanRGB])
    train_meanB=np.mean([m[2] for m in train_meanRGB])

    train_stdR=np.std([m[0] for m in train_stdRGB])
    train_stdG=np.std([m[1] for m in train_stdRGB])
    train_stdB=np.std([m[2] for m in train_stdRGB])

    val_meanRGB=[np.mean(x.numpy(), axis=(1,2)) for x, _ in val_ds]
    #이미지의 높이와 너비를 기준으로 평균을 구한다 -> RGB채널별로 이미지의 각 픽셀에 대한 평균이 계산된다.
    val_stdRGB=[np.std(x.numpy(), axis=(1,2)) for x, _ in val_ds]

    val_meanR=np.mean([m[0] for m in val_meanRGB])
    val_meanG=np.mean([m[1] for m in val_meanRGB])
    val_meanB=np.mean([m[2] for m in val_meanRGB])

    val_stdR=np.std([m[0] for m in val_stdRGB])
    val_stdG=np.std([m[1] for m in val_stdRGB])
    val_stdB=np.std([m[2] for m in val_stdRGB])

    return train_meanR,train_meanG,train_meanB,train_stdR,train_stdG,train_stdB,val_meanR,val_meanG,val_meanB,val_stdR,val_stdG,val_stdB

def resize_img(img, img_size=480):
    img_np = np.array(img)

    if img_np.shape[1] > img_np.shape[0]:
        ratio = img_size / img_np.shape[1]
    else:
        ratio = img_size / img_np.shape[0]

    img_np = cv2.resize(img_np, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)

    w, h = img_np.shape[1], img_np.shape[0]
    dw = (img_size - w) / 2
    dh = (img_size - h) / 2

    M = np.float32([[1, 0, dw], [0, 1, dh]])
    img_re = cv2.warpAffine(img_np, M, (img_size, img_size))

    return Image.fromarray(img_re)


def rand_bbox(size, lam):  # size : [B, C, W, H]
    W = size[2]  # 이미지의 width
    H = size[3]  # 이미지의 height
    cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기
    cut_w = np.int(W * cut_rat)  # 패치의 너비
    cut_h = np.int(H * cut_rat)  # 패치의 높이

    # uniform
    # 이미지 내에서 랜덤한 중심 좌표 추출
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치의 좌표값 계산 및 이미지 범위 조
    bbx1 = np.clip(cx - cut_w // 2, 0, W)  # cx: 이미지 내엣 랜덤하게 선택된 중심 x좌표, 'cut_x'는 패치의 너비의 절반
    # 왼쪽 모서리의 x 좌표 (bbx1)를 계산
    # cx - cut_w // 2는 중심 좌표에서 패치의 왼쪽 끝으로 이동한 좌표
    # cx - cut_w // 2 이 값이 -20이 나오면 값이 0 , 이미지를 벗어나지 않게끔 하기 위해서 계산 값이 - 면 0, 이미지 너비보다 크면 W
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def p_cutmix(img, label, device='cuda:0'):
    lam = np.random.beta(1.0, 1.0)
    rand_index = torch.randperm(img.size()[0]).to(device)
    target_a = label
    target_b = label[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))

    return target_a, target_b, lam