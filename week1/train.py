import os
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import DataLoader
import time
import wandb
from asgmt_1.dataset import Test_dataset,Train_dataset
from asgmt_1.model import ResnetN # ResnetN(num_block,input_channel,num_classes)
from asgmt_1.preprocessing import data_normalize, p_cutmix

import warnings
warnings.filterwarnings('ignore')

#GPU 번호 설정
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '7,' # gpu 번호를 선택

import torch
device='cuda:0'

# 데이터 불러오기
train_ds=Train_dataset(path='/home/sinssinej7/shared/hdd_ext/nvme1/public/vision/classification/cifar-100-python',transform=T.ToTensor())
val_ds=Test_dataset(path='/home/sinssinej7/shared/hdd_ext/nvme1/public/vision/classification/cifar-100-python',transform=T.ToTensor())

train_meanR,train_meanG,train_meanB,train_stdR,train_stdG,train_stdB,val_meanR,val_meanG,val_meanB,val_stdR,val_stdG,val_stdB=data_normalize(train_ds,val_ds)

transform_train = T.Compose([  # T.Lambda(lambda img: resize_img(img,48)),  # 이미지 리사이즈
    T.ToPILImage(),
    T.RandomCrop(32, padding=4),
    # T.RandomResizedCrop(32,scale=(0.5,1.0)),
    T.AutoAugment(policy=T.autoaugment.AutoAugmentPolicy.CIFAR10, interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),  # p확률로 좌우반전
    # T.ColorJitter(brightness=0.2, contrast=0.2,hue=0.2,saturation=0.2),  # 이미지 지터링
    # T.RandomVerticalFlip(p=0.5),  # p확률로 상하반전
    T.ToTensor(),
    T.Normalize(mean=[train_meanR, train_meanG, train_meanB], std=[train_stdR, train_stdG, train_stdB])])

transform_val = T.Compose([  # T.Lambda(lambda img: resize_img(img,48)),  # 이미지 resize
    T.ToPILImage(),
    T.CenterCrop(32),
    T.ToTensor(),
    T.Normalize(mean=[train_meanR, train_meanG, train_meanB], std=[train_stdR, train_stdG, train_stdB])])

train_ds.transform = transform_train
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True)
val_ds.transform = transform_val
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)


wandb.init(project='assignment1',
           name='se- resnet50_adam_weight_decay0.5,cutmix=1',
           config={
               'learning_rate': 1e-3,
               'momentum': 0.9,
               'arch': 'resnet50',
               'epochs':150

           })

model=ResnetN([3,4,6,3],3,100)
model.to(device)
loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.1)
# optimizer=torch.optim.Adam(model.parameters(),lr=wandb.config.learning_rate,weight_decay=1e-3)
optimizer=torch.optim.SGD(model.parameters(),lr=0.25,momentum=0.9,nesterov=True)


def train(train_loader, model, loss_fn, optimizer, device):
    global acc, avg_cost
    model.train()  # 모델을 학습 모드로 설정
    r_loss = 0
    corr = 0
    r_size = 0

    progress_bar = tqdm(train_loader)
    for idx, (img,label) in enumerate(progress_bar):
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()

        if  np.random.random() < 0.5:
            target_a,target_b,lam=p_cutmix(img,label)
            output = model(img)
            loss = loss_fn(output, target_a) * lam + loss_fn(output, target_b) * (1 - lam)

        else:
            output = model(img)
            loss = loss_fn(output, label)

        loss.backward()
        optimizer.step()

        _, pred = output.max(dim=1)
        corr += pred.eq(label).sum().item()
        r_loss += loss.item()
        r_size += img.size(0)

        acc = corr * 100 / r_size

        avg_cost = r_loss / (idx + 1)

        progress_bar.set_description(f'[Training] loss: {r_loss/(idx+1):.4f}, accuracy: {acc:.4f}')


    return avg_cost, acc


def accuracy(output, target, topk=(1,5)): # top1, top5 구하는 함수
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdims=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def val(val_loader, model, loss_fn, device):
    model.eval()
    with torch.no_grad():
        r_loss = 0
        top1=0
        top5=0

        for idx, (img, label) in enumerate(val_loader):
            img, label = img.to(device), label.to(device)

            output = model(img)
            acc1,acc5=accuracy(output,label)
            top1+=acc1
            top5+=acc5

            r_loss += loss_fn(output, label).item() * img.size(0)

        top1_accuracy = top1 / len(val_loader)
        top5_accuracy = top5 / len(val_loader)

        return r_loss / len(val_loader.dataset), top1_accuracy, top5_accuracy

def get_lr(opt):
  for param_group in opt.param_groups:
    return param_group['lr']


best=float('inf')
start_time = time.time()  # 코드 실행 시작 시간 기록
warmup_epochs=5
initial_lr=0.15
warmup_lr=optimizer.param_groups[0]['lr']
current_step=0
total_steps=warmup_epochs * len(train_loader)

from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer,T_max=wandb.config.epochs-warmup_epochs, eta_min=0.000001)

start_time = time.time()  # 코드 실행 종료 시간 기록

for epoch in range(wandb.config.epochs + 1):

    current_lr = get_lr(optimizer)

    print('Epoch={}/{},currnet lr={}'.format(epoch, wandb.config.epochs- 1, current_lr))

    if epoch <= warmup_epochs:
        lr = initial_lr + (warmup_lr - initial_lr) * current_step / total_steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        current_step += len(train_loader)
    else:
        scheduler.step()

    avg_cost, accuracy1 = train(train_loader, model, loss_fn, optimizer, device)
    val_loss,top1_accuracy,top5_accuracy = val(val_loader, model, loss_fn, device)

    print(val_loss, top1_accuracy,top5_accuracy)

    end_time = time.time()  # 코드 실행 종료 시간 기록
    execution_time = end_time - start_time  # 실행 시간 계산

    wandb.log({
          "epoch": epoch + 1,
          "train_loss": avg_cost,
          "train_accuracy":accuracy1,
          "test_loss": val_loss,
          "top1_accuracy": top1_accuracy,
          "top5_accuracy": top5_accuracy,
          "learning_rate": current_lr
    },step=epoch)


    if top1_accuracy > best:
        best = top1_accuracy
        torch.save(model.state_dict(), '/home/sinssinej7/private/my_first_pycharm/models3/weights2.pth')
        print('Copied best model weights!')
        wandb.run.summary["best_accuracy"] = best

    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)

    print("Execution Time:", hours, "시간", minutes, "분")

if __name__== "__main__":
    print(warmup_lr)
    print(total_steps)