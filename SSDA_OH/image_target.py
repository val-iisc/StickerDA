import argparse
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
from data_list import ImageList, ImageList_idx, StickerList, mixup_batch
import random, pdb, math, copy
from tqdm import tqdm
from logger import get_logger
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from loss import CrossEntropyLabelSmooth

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

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args):
    dset_loaders = {}
    train_bs = args.batch_size 
    
    with open(args.test_dset_path) as f_test:
        txt_test = f_test.readlines()
    with open(args.task_1_path) as f_t1:
        task_1_list = f_t1.readlines()

        
    target_dataset = ImageList_idx(txt_test, transform=image_train())
    dset_loaders["target_images"] = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker,drop_last=False, pin_memory = False)

    test_dataset = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker,drop_last=False, pin_memory = False)
    
    train_sticker_dataset = StickerList(task_1_list, transform=sticker_train(), mode="RGBA", specified_len = len(target_dataset))
    dset_loaders["train_mixup_stickers"] = DataLoader(train_sticker_dataset, batch_size=(args.batch_size//2), shuffle=True, num_workers=0, drop_last=True, pin_memory = False)

    return dset_loaders

def cal_acc(clean_loader, netBB, netMF, netMB, netMC, flag=False):
    start_test = True
    with torch.no_grad():
        data_bar = tqdm(range(len(clean_loader)))
        for i in data_bar:
            try:
                clean_images, clean_labels, _ = iter_clean.next()
            except:
                iter_clean = iter(clean_loader)
                clean_images, clean_labels, _ = iter_clean.next()
            
            data_bar.set_description("MainBranch : Step:{}".format(i))
            clean_images1, clean_labels1 = clean_images.cuda(), clean_labels.cuda()
            
            clean_outputs = netMC(netMB(netMF(netBB(clean_images1))))
            if start_test:
                all_output_clean = clean_outputs.float().cpu()
                all_label_clean = clean_labels1.float()
                
                start_test = False
            else:
                all_output_clean = torch.cat((all_output_clean, clean_outputs.float().cpu()), 0)
                all_label_clean = torch.cat((all_label_clean, clean_labels1.float()), 0)
    _, predict_clean = torch.max(all_output_clean, 1)
    predict_clean = predict_clean.cuda()
    accuracy_clean = torch.sum(torch.squeeze(predict_clean).float() == all_label_clean).item() / float(all_label_clean.size()[0])
    mean_ent_clean = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output_clean))).cpu().data.item()
    
    return accuracy_clean*100, mean_ent_clean


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netBB = network.ResBase(res_name=args.net).cuda()
        netMF = network.ResBase_Layer4(res_name=args.net).cuda()
        netSF = network.ResBase_Layer4(res_name=args.net).cuda()
      

    netMB = network.feat_bootleneck(type=args.classifier, feature_dim=netMF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netSB = network.feat_bootleneck(type=args.classifier, feature_dim=netSF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netMC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netSC = network.feat_classifier(type=args.layer, class_num = args.sticker_num, bottleneck_dim=args.bottleneck).cuda()
    
    modelpath = args.stored_model_dir_src_main + '/model.pt'   
    netBB.load_state_dict(torch.load(modelpath)['netBB'])
    netMB.load_state_dict(torch.load(modelpath)['netMB'])
    netMF.load_state_dict(torch.load(modelpath)['netMF'])
    netMC.load_state_dict(torch.load(modelpath)['netMC'])
    modelpath = args.stored_model_dir_src_st + '/model.pt'   
    netSB.load_state_dict(torch.load(modelpath)['netSB'])
    netSF.load_state_dict(torch.load(modelpath)['netSF'])
    netSC.load_state_dict(torch.load(modelpath)['netSC'])

    optimizer = optim.SGD(
        [
            {
                'params': netBB.parameters(),
                'lr': args.lr * .1  #1
            },
            {
                'params': netMF.parameters(),
                'lr': args.lr * 1  #10
            },
            {
                'params': netMB.bn.parameters(),
                'lr': args.lr * 1  #10
            },
            {
                'params': netMC.parameters(),
                'lr': args.lr * 1  #10
            },
            {
                'params': netSF.parameters(),
                'lr': args.lr * 1  #10
            },
            {
                'params': netSB.bn.parameters(),
                'lr': args.lr * 1  #10
            },
            {
                'params': netSC.parameters(),
                'lr': args.lr * 1  #10
            }
        ],
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target_images"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    acc_init = 0

    loader = dset_loaders["target_images"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    netBB.eval()
    netMF.eval()
    netMB.eval()
    netMC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx = data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            output = netMB(netMF(netBB(inputs)))  # a^t
            output_norm = F.normalize(output)
            outputs = netMC(output)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()
    
    netBB.train()
    netMF.train()
    netMB.train()
    netMC.train()
    train_data_bar = tqdm(range(max_iter))
    for step_i in train_data_bar:
        #if iter_num > max_iter:
        #    break
        if iter_num>0.5*max_iter:
            args.K = 5
            args.KK = 4

        netBB.train()
        netMF.train()
        netMB.train()
        netMC.train()
        netSF.train()
        netSB.train()
        netSC.train()

        try:
            test_images, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target_images"])
            test_images, _, tar_idx = iter_test.next()
        try:
            sticker_images, sticker_labels = iter_sticker.next()
        except:
            iter_sticker = iter(dset_loaders["train_mixup_stickers"])
            sticker_images, sticker_labels = iter_sticker.next()

        if test_images.size(0) == 1:
            continue


        iter_num += 1

        test_images1, tar_idx1 = test_images.cuda()[:len(test_images)//2], tar_idx[:len(test_images)//2]
        test_mixup_images  = mixup_batch(test_images[len(test_images)//2:], sticker_images).cuda()
        test_mixup_labels_SB, test_mixup_tar_MB = sticker_labels[:len(test_images)//2].cuda(), tar_idx[len(test_images)//2:]

        MB_images = torch.cat([test_images1, test_mixup_images])
        MB_idx = torch.cat([tar_idx1, test_mixup_tar_MB])

        SB_images = test_mixup_images
        SB_labels = test_mixup_labels_SB

        #lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netMB(netMF(netBB(MB_images)))
        output_test = netMC(features_test)
        softmax_out = nn.Softmax(dim=1)(output_test)
        output_re = softmax_out.unsqueeze(1) 

        SB_output = netSC(netSB(netSF(netBB(SB_images))))
    
        with torch.no_grad():
            SB_loss = CrossEntropyLabelSmooth(num_classes=args.sticker_num, epsilon=args.smooth, reduction=True)(SB_output, SB_labels).cuda()
            
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            
            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance,
                                     dim=-1,
                                     largest=True,
                                     k=args.K + 1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = score_bank[idx_near]  #batch x K x C
            #score_near=score_near.permute(0,2,1)

            fea_near = fea_bank[idx_near]  #batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1,
                                                       -1)  # batch x n x dim
            distance_ = torch.bmm(fea_near,
                                  fea_bank_re.permute(0, 2,
                                                      1))  # batch x K x n
            _, idx_near_near = torch.topk(
                distance_, dim=-1, largest=True,
                k=args.KK + 1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(
                match > 0., match,
                torch.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                    args.KK)  # batch x K x M

            
            #weight_kk[idx_near_near == tar_idx_] = 0

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            #print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM
            weight_kk = weight_kk.fill_(0.1)
            score_near_kk = score_near_kk.contiguous().view(
                score_near_kk.shape[0], -1, args.class_num)  # batch x KM x C

        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, args.K * args.KK,
                                                    -1)  # batch x KM x C
        const = torch.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
             weight_kk.cuda()).sum(1))
        loss = torch.mean(const)  #* 0.5

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K,
                                                         -1)  # batch x K x C
        
        loss += torch.mean(
            (F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) *
             weight.cuda()).sum(1))  #

        msoftmax = softmax_out.mean(dim=0)
        im_div = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        loss += im_div  

        loss += (args.st_wt * SB_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_data_bar.set_description("Train : Step:{}, MB_class_loss:{:.4f}, SB_class_loss:{:.4f}".format(iter_num, loss, SB_loss))
        if (iter_num % 100) == 0 or iter_num == max_iter:
            netBB.eval()
            netMF.eval()
            netMB.eval()
            netMC.eval()
            netSF.eval()
            netSB.eval()
            netSC.eval()
            
            acc, ent = cal_acc(dset_loaders['test'], netBB, netMF, netMB, netMC, False)
                
            args.logger.info('Step: [{}/{}], Clean_Adapt_Acc.:{:.4f}'.format(iter_num, max_iter, acc))

            if acc >= acc_init:
                acc_init = acc
                args.logger.info("Update Best Acc: {}".format(acc_init))
                best_netBB = netBB.state_dict()
                best_netMB = netMB.state_dict()
                best_netMF = netMF.state_dict()
                best_netMC = netMC.state_dict()
                best_netSF = netSF.state_dict()
                best_netSB = netSB.state_dict()
                best_netSC = netSC.state_dict()

                torch.save({'netBB':best_netBB,
                            'netMF':best_netMF,
                            'netMB':best_netMB,
                            'netMC':best_netMC,
                            'netSF':best_netSF,
                            'netSB':best_netSB,
                            'netSC':best_netSC}, osp.join(args.output_dir_src, "model.pt"))
        

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")
    parser.add_argument('--s', type=int, default=3, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=300, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--k', type=int, default=2, help="number of neighborhoods")
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--KK', type=int, default=3)
    parser.add_argument('--st_wt', type=int, default=1)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--task_1_path', type=str, default='../data/textured/Task_1_mixup.txt')
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    args.sticker_num = 11
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    folder = '../data/'  
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'


    args.task_1_classes = "../data/textured/Task_1_classes.txt"
    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]

    args.output_dir_src = osp.join('Ckpt_Adapt_unf', args.dset, names[args.s].upper()[0]+'2'+names[args.t].upper()[0])
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    log_file = args.output_dir_src + '/log.txt'
    args.logger = get_logger('__train__', log_file)

    args.stored_model_dir_src_main = osp.join('Checkpoint_main', args.dset, names[args.s].upper()[0]+'2'+names[args.t].upper()[0])
    args.stored_model_dir_src = osp.join('Checkpoint_Sticker', args.dset, names[args.s].upper()[0]+'2'+names[args.t].upper()[0])
    
    train_target(args)