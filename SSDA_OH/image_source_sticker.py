import argparse
#from object.utils import AverageMeter
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
from torch.utils.data import DataLoader
from data_list import ImageList, StickerList, ShuffledImageList, mixup_batch
import random, pdb, math, copy
from tqdm import tqdm
from logger import get_logger
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

from utils import *

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = transforms.Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args):
    dset_loaders = {}
    train_bs = args.batch_size 

    with open(args.s_dset_path) as f_src:
        txt_src = f_src.readlines()
    
    with open(args.test_dset_path) as f_test:
        txt_test = f_test.readlines()


    with open(args.task_1_path) as f_t1:
        task_1_list = f_t1.readlines()
    

    dsize = len(txt_src)
    tr_size = int(0.9*dsize)
    tr_txt, val_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])

    train_dataset = ImageList(tr_txt, transform=image_train())
    dset_loaders["train_clean_images"] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,drop_last=False, pin_memory = False)
    
    test_dataset = ImageList(txt_test, transform=image_test())
    dset_loaders["test_images"] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker,drop_last=False, pin_memory = False)

    val_dataset = ImageList(val_txt, transform=image_test())
    dset_loaders["val_images"] = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker,drop_last=False, pin_memory = False)

    bg_dataset = ShuffledImageList(tr_txt, transform=image_train())
    dset_loaders["train_bg_images"] = DataLoader(bg_dataset, batch_size=args.batch_size_bg, shuffle=True, num_workers=args.worker,drop_last=True, pin_memory = False)

    train_sticker_dataset = StickerList(task_1_list, transform=sticker_train(), mode="RGBA", specified_len = len(train_dataset))
    dset_loaders["train_mixup_stickers"] = DataLoader(train_sticker_dataset, batch_size=args.batch_size+(args.batch_size_bg//2), shuffle=True, num_workers=0, drop_last=True, pin_memory = False)


    return dset_loaders

def cal_acc_oda(args, clean_loader, sticker_loader, bg_loader, netBB, netF, netB, netC, weights):
    start_test = True
    accumulated_loss = AverageMeter()

    with torch.no_grad():
        iter_clean_val = iter(clean_loader)
        iter_sticker_val = iter(sticker_loader)
        iter_bg_val = iter(bg_loader)
        data_bar = tqdm(range(len(clean_loader)))
        for i in data_bar:
            
            try:
                mixup_stickers, mixup_labels = iter_sticker_val.next()
            except:
                iter_sticker_val = iter(sticker_loader)
                mixup_stickers, mixup_labels = iter_sticker_val.next()
            try:
                clean_images, clean_labels = iter_clean_val.next()
            except:
                iter_clean_val = iter(clean_loader)
                clean_images, clean_labels = iter_clean_val.next()
            try:
                bg_images, bg_labels = iter_bg_val.next()
            except:
                iter_bg_val = iter(bg_loader)
                bg_images, bg_labels = iter_bg_val.next()

            clean_images = clean_images.cuda()
            
            mixed_up, mixed_up_labels = mixup_batch(clean_images, mixup_stickers).cuda(), mixup_labels.cuda()[:len(clean_images)]
            bg_images1, bg_labels1 = bg_images.cuda()[:args.batch_size_bg//2], bg_labels[:args.batch_size_bg//2].cuda()
            bg_mix_up ,bg_mix_labels = mixup_batch(bg_images[args.batch_size_bg//2:], mixup_stickers[args.batch_size:]).cuda(), bg_labels[args.batch_size_bg//2:].cuda()
            inputs = torch.cat((mixed_up, bg_images1, bg_mix_up),0)
            labels = torch.cat((mixed_up_labels, bg_labels1, bg_mix_labels))
            outputs = netC(netB(netF(netBB(inputs))))
            
            data_bar.set_description("{} : Step:{}".format(args.model, i))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False 
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    
    all_label = all_label.cpu()
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int),:]

    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    ood_acc = acc[-1:].item()

    return acc[:-1], np.mean(acc[:-1]), ood_acc, np.mean(acc) 

def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netBB = network.ResBase(res_name=args.net).cuda()
        netSF = network.ResBase_Layer4(res_name=args.net).cuda()
 
    netSB = network.feat_bootleneck(type=args.classifier, feature_dim=netSF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netSC = network.feat_classifier(type=args.layer, class_num = args.sticker_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netBB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netSF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netSB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netSC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    modelpath = args.stored_model_dir_src + '/model.pt'
    netBB.load_state_dict(torch.load(modelpath)['netBB'])
    netSF.load_state_dict(torch.load(modelpath)['netMF'])
    netSB.load_state_dict(torch.load(modelpath)['netMB'])

    netBB.eval()
    for k, v in netBB.named_parameters():
        v.requires_grad = False
    
    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["train_clean_images"])
    interval_iter = max_iter // 10
    start_test = True

    netSF.train()
    netSB.train()
    netSC.train()

    accumulated_train_acc = AverageMeter()
    accumulated_train_loss = AverageMeter()

    weights = torch.ones((1,11))
    weights[0, -1] = 0.01
    weights = weights.cuda()
    train_data_bar = tqdm(range(max_iter))
    iter_num = 0
    for step_i in train_data_bar:
        #if iter_num > max_iter:
        #    break
        try:
            clean_images, clean_labels = iter_clean.next()
        except:
            iter_clean = iter(dset_loaders["train_clean_images"])
            clean_images, clean_labels = iter_clean.next()
        try:
            bg_images, bg_labels = iter_bg.next()
        except:
            iter_bg = iter(dset_loaders["train_bg_images"])
            bg_images, bg_labels = iter_bg.next()
        try:
            mixup_stickers, mixup_labels = iter_sticker.next()
        except:
            iter_sticker = iter(dset_loaders["train_mixup_stickers"])
            mixup_stickers, mixup_labels = iter_sticker.next()

        if clean_images.size(0) == 1:
            continue

    
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        clean_images = clean_images.cuda()
        mixed_up_images, mixed_up_labels = mixup_batch(clean_images, mixup_stickers).cuda(), mixup_labels.cuda()[:len(clean_images)]

        bg_images1, bg_labels1 = bg_images.cuda()[:args.batch_size_bg//2], bg_labels.cuda()[:args.batch_size_bg//2]
        bg_mix, bg_mix_labels = mixup_batch(bg_images[args.batch_size_bg//2:],mixup_stickers[args.batch_size:]).cuda(), bg_labels[args.batch_size_bg//2:].cuda()
        
        images = torch.cat([mixed_up_images, bg_images1, bg_mix])
        labels = torch.cat([mixed_up_labels, bg_labels1, bg_mix_labels])

        outputs = netSC(netSB(netSF(netBB(images))))
        if start_test:
            all_output = outputs.float().cpu()
            all_label = labels.float()
            start_test = False
        else:
            all_output = torch.cat((all_output, outputs.float().cpu()), 0)
            all_label = torch.cat((all_label, labels.float()), 0)
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.sticker_num, weights = weights, epsilon=args.smooth, reduction=True)(outputs, labels)
               
        #mixed_up_classifier_loss = CrossEntropyLabelSmooth(num_classes=args.sticker_num, epsilon=args.smooth)(mixed_up_outputs, mixed_up_labels)
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        accumulated_train_loss.update(classifier_loss.item())
        
        train_data_bar.set_description("Train: Steps:{} , Loss:{:.4f}".format(iter_num, accumulated_train_loss.avg))
        if ((iter_num) % 500 == 0) or (iter_num == max_iter):
            start_test = True
            all_output = nn.Softmax(dim=1)(all_output)
            _, predict = torch.max(all_output, 1)

            all_label = all_label.cpu()
            matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
            matrix = matrix[np.unique(all_label).astype(int),:]

            acc = matrix.diagonal()/matrix.sum(axis=1) * 100
            train_ood_acc = acc[-1:].item()
            train_class_acc = acc[:-1]
            train_mean_acc = np.mean(acc)
            train_class_mean_acc = np.mean(acc[:-1])
            netSF.eval()
            netSB.eval()
            netSC.eval()
            
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(args, dset_loaders['val_images'], dset_loaders['train_mixup_stickers'], netBB, netSF, netSB, netSC)
                
            else:
                args.model = "Val"
                val_loss_acc = cal_acc_oda(args, dset_loaders['val_images'],
                dset_loaders['train_mixup_stickers'], dset_loaders['train_bg_images'], netBB, netSF, netSB, netSC, weights)

                args.model = "Test"
                test_loss_acc = cal_acc_oda(args, dset_loaders['test_images'],
                dset_loaders['train_mixup_stickers'], dset_loaders['train_bg_images'], netBB, netSF, netSB, netSC, weights)
            
            # Training Logs
            args.logger.info("Train : Step:[{}/{}] , Loss:{}, Per_Class_Acc:{}, Class_mean_acc:{}, OOD_Acc:{}, Mean_Acc:{}".format(iter_num , max_iter, 
            accumulated_train_loss.avg, train_class_acc, train_class_mean_acc, train_ood_acc, train_mean_acc))
            
            # Validation Logs
            args.logger.info("Val: Step:[{}/{}], Per_Class_Acc:{} , Class_mean_Acc:{}, OOD_Acc:{}, Mean_Acc: {}".format(iter_num , max_iter, val_loss_acc[0],
            val_loss_acc[1], val_loss_acc[2], val_loss_acc[3]))
            
            # Test Logs
            args.logger.info("Test: Per_Class_Acc:{} , Class_mean_Acc:{}, OOD_Acc:{}, Mean_Acc: {}".format(test_loss_acc[0],
            test_loss_acc[1], test_loss_acc[2], test_loss_acc[3]))
            
            
            if val_loss_acc[3] >= acc_init:
                acc_init = val_loss_acc[3]
                args.logger.info("Update Best Mean_Acc: {}".format(acc_init))
                best_netBB = netBB.state_dict()
                best_netSF = netSF.state_dict()
                best_netSB = netSB.state_dict()
                best_netSC = netSC.state_dict()

                torch.save({'netBB':best_netBB,
                            'netSF':best_netSF,
                            'netSB':best_netSB,
                            'netSC':best_netSC}, osp.join(args.output_dir_src, "model.pt"))
            accumulated_train_loss = AverageMeter()

            netSF.train()
            netSB.train()
            netSC.train()


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--s', type=int, default=3, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=1500, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size of clean images")
    parser.add_argument('--batch_size_bg', type=int, default=32, help="batch_size of background images")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office-31', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--task_1_path', type=str, default='../data/textured/Task_1_mixup.txt')

    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office-31':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    args.sticker_num = 11
    args.names = names
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    folder = '../data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'

    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    args.task_1_classes = "../data/textured/Task_1_classes.txt"
    args.task_3_classes = "../data/" + args.dset + "/classes.txt"

    if args.dset == 'office-home':

        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]

    args.output_dir_src = osp.join('Checkpoint_Sticker', args.dset, names[args.s][0].upper() + '2' + names[args.t][0].upper())
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    log_file = args.output_dir_src + '/log.txt'
    args.logger = get_logger('__train__', log_file)
    args.stored_model_dir_src = osp.join('Checkpoint_main', args.dset, names[args.s][0].upper() + '2' + names[args.t][0].upper())


    train_source(args)

