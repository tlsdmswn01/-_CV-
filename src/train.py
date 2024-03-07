# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import RandomSampler,SequentialSampler

from torch import distributed as dist
import torch.utils.data.distributed
from torchvision.datasets.samplers import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import resnet as RN
import utils
import numpy as np

from thop.profile import profile

import warnings
import wandb

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.25, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=1.0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')
parser.add_argument('--wandb',default=False,help='use_wandb')
parser.add_argument('--gpu', type=str, default='0,',help='use_gpu')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100


def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    #GPU 번호 설정
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # gpu 번호를 선택
    args.local_rank = int(os.environ.get('LOCAL_RANK',0))

    args.device = 'cuda:%d' % args.local_rank
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(args.local_rank) #해당 gpu를 사용하겠다
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    print(args.world_size)
    args.rank = torch.distributed.get_rank() #현재 프로세스의 랭크 확인
    print('local_rank:',args.local_rank)
    print(args.rank)
    print(args.rank, args.device, args.local_rank) #출력해보면 동일한 것을 확인할 수 있음

    args.distributed=True


    if args.wandb:
        wandb.init(project='assignment3',
            name='ddp,wandb',
            config={
                'learning_rate': 1e-3,
                'momentum': 0.9,
                'arch': 'resnet50',
                'epochs':200
            })
    
    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        #데이터 dist
        if args.dataset == 'cifar100':
            train_dataset=datasets.CIFAR100('../data', train=True, download=True, transform=transform_train)
            val_dataset= datasets.CIFAR100('../data', train=False, transform=transform_test)
            numberofclass = 100

            if args.distributed:
                train_sampler = DistributedSampler(train_dataset, shuffle=True)
                val_sampler = DistributedSampler(val_dataset, shuffle=False)
                print(1)
            else:
                train_sampler = RandomSampler(train_dataset)
                val_sampler = SequentialSampler(val_dataset)

            train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=int(args.batch_size/args.world_size), shuffle=False, sampler=train_sampler,num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=int(args.batch_size/args.world_size), shuffle=False, sampler=val_sampler,num_workers=args.workers, pin_memory=True)


        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':
        traindir = os.path.join('/home/data/ILSVRC/train')
        valdir = os.path.join('/home/data/ILSVRC/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        numberofclass = 1000

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))
    

    model = RN.resnet50_tutorial()
    model.cuda()
    dsize = (1, 3, 224, 224)
    inputs = torch.randn(dsize).cuda()
    start_time=time.time()
    torch_ops,_=profile(model,(inputs,),verbose=False)
    end_time=time.time()
    torch_ops=torch_ops/(1000**3)
    params=count_params(model)/(1000**2)
    throughput=args.batch_size/(end_time-start_time)

    
    # 모델 dist
    if args.distributed:
        model = RN.resnet50_tutorial()
        model.cuda(args.local_rank)
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=[args.local_rank])
    else:
        model = RN.resnet50_tutorial()
        model.cuda()

    # define loss function (criterion) and optimizer
    if args.distributed:
        criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    else:
        criterion = nn.CrossEntropyLoss().cuda()


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True



    for epoch in range(0, args.epochs):

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        err1, err5, val_loss= validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            #'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_acc1': best_err1,
            'best_acc5': best_err5,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        if args.wandb:
            if args.local_rank == 0:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "test_loss": val_loss,
                    "top1_accuracy": err1,
                    "top5_accuracy": err5,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    'number_of_params': params,
                    'throughput': throughput,
                    'FLOPS' : torch_ops
                },step=epoch)

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    start_time = time.time()
    end=time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.distributed:
            input = input.cuda(args.local_rank)
            target = target.cuda(args.local_rank)
        else:
            input = input.cuda()
            target = target.cuda()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            if args.distributed:
                loss = reduce_mean(loss, args.world_size)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)
            if args.distributed:
                loss = reduce_mean(loss, args.world_size)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    num_batches = len(val_loader)  # 검증 데이터셋의 총 배치 수

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():

        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            if args.distributed:
                input = input.cuda(args.local_rank)
                target = target.cuda(args.local_rank)
            else:
                input = input.cuda()
                target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            err1, err5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))

            top1.update(err1.item(), input.size(0))
            top5.update(err5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.verbose == True:
                print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                    'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                    epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
            epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def count_params(model: nn.Module):
    return sum([m.numel() for m in model.parameters()])


def reduce_mean(val, world_size):
    """Collect value to local zero gpu
    :arg
        val(tensor): target
        world_size(int): the number of process in each group
    """
    val = val.clone()
    dist.reduce(val, 0, dist.ReduceOp.SUM)
    val = val / world_size
    return val


if __name__ == '__main__':
    main()