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
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, StickerList, mixup_batch
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


    with open(args.test_dset_path_0) as f_test0:
        txt_test0 = f_test0.readlines()
    with open(args.test_dset_path_1) as f_test1:
        txt_test1 = f_test1.readlines()
    with open(args.test_dset_path_2) as f_test2:
        txt_test2 = f_test2.readlines()
    

    with open(args.s_dset_path) as f_src:
        txt_src = f_src.readlines()


    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar0 = []
        for i in range(len(txt_test0)):
            rec = txt_test0[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar0.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar0.append(line)
        txt_test0 = new_tar0.copy()

        new_tar1 = []
        for i in range(len(txt_test1)):
            rec = txt_test1[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar1.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar1.append(line)
        txt_test1 = new_tar1.copy()

        new_tar2 = []
        for i in range(len(txt_test2)):
            rec = txt_test2[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar2.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar2.append(line)
        txt_test2 = new_tar2.copy()
    
    with open(args.task_1_path) as f_t1:
        task_1_list = f_t1.readlines()

    dsize = len(txt_src)
    tr_size = int(0.9*dsize)
    tr_txt, val_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])

    train_dataset = ImageList(tr_txt, transform=image_train())
    dset_loaders["train_images"] = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=args.worker,drop_last=False, pin_memory = False)
    
    test_dataset0 = ImageList(txt_test0, transform=image_test())
    dset_loaders["test_images0"] = DataLoader(test_dataset0, batch_size=train_bs, shuffle=False, num_workers=args.worker,drop_last=False, pin_memory = False)

    test_dataset1 = ImageList(txt_test1, transform=image_test())
    dset_loaders["test_images1"] = DataLoader(test_dataset1, batch_size=train_bs, shuffle=False, num_workers=args.worker,drop_last=False, pin_memory = False)

    test_dataset2 = ImageList(txt_test2, transform=image_test())
    dset_loaders["test_images2"] = DataLoader(test_dataset2, batch_size=train_bs, shuffle=False, num_workers=args.worker,drop_last=False, pin_memory = False)

    val_dataset = ImageList(val_txt, transform=image_test())
    dset_loaders["val_images"] = DataLoader(val_dataset, batch_size=train_bs, shuffle=False, num_workers=args.worker,drop_last=True, pin_memory = False)

    train_sticker_dataset = StickerList(task_1_list, transform=sticker_train(), mode="RGBA", specified_len = len(train_dataset))
    dset_loaders["train_mixup_stickers"] = DataLoader(train_sticker_dataset, batch_size=train_bs//2, shuffle=True, num_workers=0, drop_last=True, pin_memory = False)

    return dset_loaders


def cal_acc(args, loader, netBB, netMF, netMB, netMC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        data_bar = tqdm(range(len(loader)))
        for i in data_bar:
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netMC(netMB(netMF(netBB(inputs))))

            data_bar.set_description("{} : Step:{} ".format(args.model, i))

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1,1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent>threshold] = args.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int),:]

    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()
    known_acc = np.mean(acc[:-1])
    hos = (2 * known_acc * unknown_acc) / (known_acc + unknown_acc) #Harmonic Mean of known_acc and unknown_acc
    return known_acc, hos, unknown_acc
    # return np.mean(acc), np.mean(acc[:-1])

def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netBB = network.ResBase(res_name=args.net).cuda()
        netMF = network.ResBase_Layer4(res_name=args.net).cuda()
        netSF = network.ResBase_Layer4(res_name=args.net).cuda()

    netMB = network.feat_bootleneck(type=args.classifier, feature_dim=netMF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netSB = network.feat_bootleneck(type=args.classifier, feature_dim=netSF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netMC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netSC = network.feat_classifier(type=args.layer, class_num = args.sticker_class, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netBB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netMF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netMB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netMC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init0 = 0
    acc_init1 = 0
    acc_init2 = 0
    max_iter = args.max_epoch * len(dset_loaders["train_images"])
    interval_iter = max_iter // 10
    save_interval = len(dset_loaders["train_images"])

    netBB.train()
    netMF.train()
    netMB.train()
    netMC.train()

    accumulated_train_acc = AverageMeter()
    accumulated_train_loss = AverageMeter()

    accumulated_train_acc_mixed_up = AverageMeter()
    accumulated_train_loss_mixed_up = AverageMeter()

    train_data_bar = tqdm(range(max_iter))
    iter_num = 0
    for step_i in train_data_bar:
        #if iter_num > max_iter:
        #    break
        try:
            clean_images, clean_labels = iter_clean.next()
        except:
            iter_clean = iter(dset_loaders["train_images"])
            clean_images, clean_labels = iter_clean.next()
        
        try:
            mixup_stickers, mixup_labels = iter_sticker.next()
        except:
            iter_sticker = iter(dset_loaders["train_mixup_stickers"])
            mixup_stickers, mixup_labels = iter_sticker.next()

        if clean_images.size(0) == 1:
            continue

    
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        len_clean_imgs = len(clean_images)
        clean_images1, clean_labels1 = clean_images.cuda()[:len_clean_imgs//2], clean_labels.cuda()[:len_clean_imgs//2]
        mixed_up_images, mixed_up_labels = mixup_batch(clean_images[len_clean_imgs//2:], mixup_stickers).cuda(), clean_labels.cuda()[len_clean_imgs//2:]
        
        images = torch.cat([clean_images1, mixed_up_images])
        labels = torch.cat([clean_labels1, mixed_up_labels])
    
        outputs = netMC(netMB(netMF(netBB(images))))
        
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs, labels)            
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        accumulated_train_loss.update(classifier_loss.item())
        
        train_acc = torch.sum(torch.squeeze(torch.max(outputs, 1)[1]).float() == labels).item() / float(labels.size()[0]) * 100
        accumulated_train_acc.update(train_acc)

        train_data_bar.set_description("Train: Steps:{} , Loss:{:.4f} , Acc:{:.4f}".format(iter_num, accumulated_train_loss.avg, accumulated_train_acc.avg))
        if ((iter_num) % 500 == 0) or (iter_num == max_iter):
            netBB.eval()
            netMF.eval()
            netMB.eval()
            netMC.eval()
            # Training Logs
            args.logger.info("Train : Step:[{}/{}] , Loss:{} , Acc:{} ".format(iter_num , max_iter, 
            accumulated_train_loss.avg, accumulated_train_acc.avg))

            # Validation Logs
            args.model = "Val"
            val_acc, _ = cal_acc(args, dset_loaders['val_images'], netBB, netMF, netMB, netMC, False)
            args.logger.info("Val: Clean_Acc: {}".format(val_acc))

            # TEST1
            args.model = args.names[args.s][0].upper() + '2' + args.names[args.t[0]][0].upper()
            test_acc, _ = cal_acc(args, dset_loaders['test_images0'], netBB, netMF, netMB, netMC, False)
            args.logger.info("Test: {}, : Clean_Acc: {}".format(args.model, test_acc))            

            if val_acc >= acc_init0:
                acc_init0 = val_acc
                args.logger.info("Test: {}, Update Best Acc: {}".format(args.model, acc_init0))
                best_netBB = netBB.state_dict()
                best_netMF = netMF.state_dict()
                best_netMB = netMB.state_dict()
                best_netMC = netMC.state_dict()

                torch.save({'netBB':best_netBB,
                            'netMF':best_netMF,
                            'netMB':best_netMB,
                            'netMC':best_netMC}, osp.join(args.output_dir0, "model.pt"))

            # TEST2
            args.model = args.names[args.s][0].upper() + '2' + args.names[args.t[1]][0].upper()
            test_acc, _ = cal_acc(args, dset_loaders['test_images1'], netBB, netMF, netMB, netMC, False)
            args.logger.info("Test: {}, : Clean_Acc: {}".format(args.model, test_acc))

            if val_acc >= acc_init1:
                acc_init1 = val_acc
                args.logger.info("Test: {}, Update Best Acc: {}".format(args.model, acc_init1))
                best_netBB = netBB.state_dict()
                best_netMF = netMF.state_dict()
                best_netMB = netMB.state_dict()
                best_netMC = netMC.state_dict()

                torch.save({'netBB':best_netBB,
                            'netMF':best_netMF,
                            'netMB':best_netMB,
                            'netMC':best_netMC}, osp.join(args.output_dir1, "model.pt"))

            # TEST3
            args.model = args.names[args.s][0].upper() + '2' + args.names[args.t[2]][0].upper()
            test_acc, _ = cal_acc(args, dset_loaders['test_images2'], netBB, netMF, netMB, netMC, False)
            args.logger.info("Test: {}, : Clean_Acc: {}".format(args.model, test_acc))

            if val_acc >= acc_init2:
                acc_init2 = val_acc
                args.logger.info("Test: {}, Update Best Acc: {}".format(args.model, acc_init2))
                best_netBB = netBB.state_dict()
                best_netMF = netMF.state_dict()
                best_netMB = netMB.state_dict()
                best_netMC = netMC.state_dict()

                torch.save({'netBB':best_netBB,
                            'netMF':best_netMF,
                            'netMB':best_netMB,
                            'netMC':best_netMC}, osp.join(args.output_dir2, "model.pt"))

            accumulated_train_acc = AverageMeter()
            accumulated_train_loss = AverageMeter()
            netBB.train()
            netMF.train()
            netMB.train()
            netMC.train()
                

            torch.save({'netBB':netBB.state_dict(),
                        'netMF':netMF.state_dict(),
                        'netMB':netMB.state_dict(),
                        'netMC':netMC.state_dict()}, osp.join(args.output_dir_src,  str(iter_num) +'_'+ "model.pt"))


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=3, help="source")
    parser.add_argument('--t', type=int, default=[0,1,2], nargs = 3, help="target")
    parser.add_argument('--max_epoch', type=int, default=250, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
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
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
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
    args.sticker_class = 11
    args.names = names
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = '../data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'

    args.test_dset_path_0 = folder + args.dset + '/' + names[args.t[0]] + '_list.txt'
    args.test_dset_path_1 = folder + args.dset + '/' + names[args.t[1]] + '_list.txt'
    args.test_dset_path_2 = folder + args.dset + '/' + names[args.t[2]] + '_list.txt'

    args.task_1_classes = "../data/textured/Task_1_classes.txt"
    args.task_3_classes = "../data/" + args.dset + "/classes.txt"

    if args.dset == 'office-home':
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]

    args.output_dir_src = osp.join('Checkpoint_main', args.dset, names[args.s].upper())
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    log_file = args.output_dir_src + '/log.txt'
    args.logger = get_logger('__train__', log_file)

    args.output_dir0 = osp.join('Checkpoint_main', args.dset, args.names[args.s][0].upper() + '2' + args.names[args.t[0]][0].upper())
    if not osp.exists(args.output_dir0):
        os.system('mkdir -p ' + args.output_dir0)
    if not osp.exists(args.output_dir0):
        os.mkdir(args.output_dir0)

    args.output_dir1 = osp.join('Checkpoint_main', args.dset, args.names[args.s][0].upper() + '2' + args.names[args.t[1]][0].upper())
    if not osp.exists(args.output_dir1):
        os.system('mkdir -p ' + args.output_dir1)
    if not osp.exists(args.output_dir1):
        os.mkdir(args.output_dir1)

    args.output_dir2 = osp.join('Checkpoint_main', args.dset, args.names[args.s][0].upper() + '2' + args.names[args.t[2]][0].upper())
    if not osp.exists(args.output_dir2):
        os.system('mkdir -p ' + args.output_dir2)
    if not osp.exists(args.output_dir2):
        os.mkdir(args.output_dir2)

    train_source(args)

