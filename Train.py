import argparse
import os
from time import time
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from kdutils import seed_all, GradualWarmupScheduler
#from torchvision.models.resnet import resnet18
from torchvision import transforms
from models import *
from data.CIFAR10 import GetCifar10

seed_all(1000)

parser = argparse.ArgumentParser(description="CIFAR10_SNN_Training")

parser.add_argument("--model", type=str, default='resnet19')

parser.add_argument("--datapath", type=str, default='./dataset/cifar10/')
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--epoch", type=int, default=320)
parser.add_argument("--warm_up", action='store_true', default=False)
parser.add_argument("--load_weight", action='store_true', default=False)
parser.add_argument('--spike', action='store_true', help='use spiking network')
parser.add_argument('--lr', default=1e-1, type=float, help='initial learning rate')
parser.add_argument('--step', default=4, type=int, help='snn step')
args = parser.parse_args()

best_acc = 0.3
sta = time()

######## input model #######
model = resnet19_cifar()
# should be changed

######## save model #######
model_save_name = 'raw/snn-' + args.model + 'VirtualSpike.pth'
model_record_name = 'raw/' + args.model + 'VirtualSpike.txt'

######## change to snn #######
if args.spike is True:
    model = SpikeModel(model, args.step)
    model.set_spike_state(True)

######## load weight #######
#model.load_state_dict(torch.load('raw/snn-resnet19.pth', map_location=torch.device('cpu')))
    
SNN = model.cuda()

######## show parameters #######
n_parameters = sum(p.numel() for p in SNN.parameters() if p.requires_grad)
print('number of params:', n_parameters)
print(SNN)

######## amp #######
loss_fun = torch.nn.CrossEntropyLoss().cuda()
scaler = torch.cuda.amp.GradScaler()

######## split BN #######

optimer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


scheduler = CosineAnnealingLR(optimer, T_max=args.epoch, eta_min=0)
writer = None

traindir = args.datapath + 'train'
valdir = args.datapath + 'val'

###data

train_data,test_data= GetCifar10(args.batch)

if __name__ == '__main__':
    for i in range(args.epoch):
        loss_ce_all = 0
        start_time = time()
        total = 0
        right = 0
        SNN.train()
        print("epoch:{}".format(i))
        for step, (imgs, target) in enumerate(train_data):
            imgs, target = imgs.cuda(non_blocking=True), target.cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                output = SNN(imgs,is_drop=False)

                loss = loss_fun(output,target)
            right = (output.argmax(1) == target).sum() + right
            loss_ce_all += loss.item()

            optimer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimer)
            scaler.update()

            total += float(target.size(0))

            if step % 100 == 0:
                print("step:{:.2f} loss_ce:{:.2f}".format(step / len(train_data), loss.item()))
        accuracy1 = 100 * right / total
        
        scheduler.step()
        
        SNN.eval()
        right = 0
        total = 0

        with torch.no_grad():
            for (imgs, target) in test_data:
                imgs, target = imgs.cuda(non_blocking=True), target.cuda(non_blocking=True)
                output  = SNN(imgs,is_drop=False)
                right = (output.argmax(1) == target).sum() + right
                total += float(target.size(0))

            accuracy = 100 * right / total
            end_time = time()
            print("epoch:{} time:{:.0f}  loss:{:.4f} train_acc:{:.4f} test_acc:{:.4f} eta:{:.2f}".format(i,end_time - start_time,loss_ce_all,accuracy1,accuracy, (end_time - start_time) * (args.epoch - i - 1) / 3600))
            if accuracy > best_acc:
                best_acc = accuracy
                print("best_acc:{:.4f}".format(best_acc))
                torch.save(model.state_dict(), model_save_name)
        filepoint=open(model_record_name,'a')
        filepoint.write("epoch:{} time:{:.0f}  loss:{:.4f} train_acc:{:.4f} test_acc:{:.4f} eta:{:.2f}\n".format(i,end_time - start_time,loss_ce_all,accuracy1,accuracy, (end_time - start_time) * (args.epoch - i - 1) / 3600))
        filepoint.write("best_acc:{:.4f}\n".format(best_acc))
        filepoint.close()
    
    end = time()
    print(end - sta)
    print("best_acc:{:.4f}".format(best_acc))