# coding=gbk
import random
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import time
import shutil
from dataloader_isic import CancerSeT_CSV
from sklearn import metrics
from utils import Logger, AverageMeter, accuracy
from pytorch_pretrained_vit import ViT
from utils.metrics_function import ACC_2Clas_statistic, AUC_2Clas_statistic,  Confusion_Mat_2Clas_statistic
from torch.utils.data import Dataset
import pandas as pd
import argparse
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()
    print(f"Epoch {epoch}: Learning Rate = {optimizer.param_groups[0]['lr']:.6f}")

    for idx, data_img in enumerate(data_loader):
        inputs = data_img["img"].float()
        targets = data_img["labels"].float()
        if torch.cuda.is_available():
            data, target = inputs.cuda(), targets.long().cuda()
        criterion = torch.nn.CrossEntropyLoss()

        output = model(data)
        loss = criterion(output, target)
        acc = accuracy(output.data, target.data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 50 == 0:
            print('[' + '{:5}'.format(idx * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * idx / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()) + ' Acc:' + '{:6.4f}'.format(acc[0].item()))  # acc[0].item()
            loss_history.append(loss.item())


def evaluate(model, test_loader):
    model.eval()
    total_samples = len(test_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        people_id = []
        pred_list = []
        targets = []
        for idx, data in enumerate(test_loader):
            inputs = data["img"].float()
            labels = data["labels"].float()
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.long().cuda()

            outputs = model(inputs)
            pred = outputs.argmax(-1)
            correct_samples += pred.eq(labels).sum()

            people_id.extend(data['id'])
            pred_list.extend(outputs.detach().cpu().numpy())  # detel .tolist()
            targets.extend(labels.detach().cpu().numpy().tolist())

        acc = 100.0 * correct_samples / total_samples
        # df_error = pd.DataFrame({'people_id': people_id, 'preds': pred_list, 'labels': targets})
        # ACC_error_statistic(df_error)
        df = pd.DataFrame({'people_id': people_id, 'preds': pred_list, 'labels': targets})
        df = df.groupby('people_id')[['labels', 'preds']]
        person_preds, person_label, person_preds_label, acc_statistic = ACC_2Clas_statistic(df)
        auc_statistic = AUC_2Clas_statistic(person_preds, person_label)
        Confusion_Mat_2Clas_statistic(person_label, person_preds_label)
        print('  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
              '{:5}'.format(total_samples) + ' (' + '{:4.2f}'.format(acc) + '%)' + 'statis acc: ' + '{:4.2f}'.format(
            acc_statistic))

        return acc, acc_statistic, auc_statistic



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a ViT model on ISIC 2019 dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--experiment_name", type=str, default="ISIC_Classification", help="Experiment name.")
    parser.add_argument("--lr_init", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes.")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers for data loading.")
    parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--milestones", type=int, nargs="+", default=[50, 75], help="Milestones for LR scheduler.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Learning rate decay factor.")
    parser.add_argument("--train_path",
                        type=str,
                        default="/home/dkd/Data_4TDISK/ISIC_2019/ISIC_2019_Training_Input/",
                        help="Path to training data.")
    parser.add_argument("--val_path",
                        type=str,
                        default="/home/dkd/Data_4TDISK/ISIC_2019/ISIC_2019_Training_Input/",
                        help="Path to training data.")
    parser.add_argument("--test_path",
                        type=str,
                        default="/home/dkd/Data_4TDISK/ISIC_2019/ISIC_2019_Training_Input/",
                        help="Path to training data.")

    args = parser.parse_args()
    set_seed(args.seed)

    Liver_loader_train = CancerSeT_CSV(args.train_path, 'train')
    Liver_loader_val = CancerSeT_CSV(args.val_path, 'val')
    Liver_loader_test = CancerSeT_CSV(args.val_path, 'test')

    # Define data loaders for the unlabeled pool and test set
    train_loader = torch.utils.data.DataLoader(Liver_loader_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(Liver_loader_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)
    test_loader = torch.utils.data.DataLoader(Liver_loader_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                              pin_memory=True)

    #Model
    # model = make_model('resnet50', num_classes=args.num_classes, pretrained=True)
    # model = make_model('densenet121', num_classes=args.num_classes, pretrained=True)
    # model = make_model('shufflenet_v2_x1_0', num_classes=args.num_classes, pretrained=True)
    model = ViT('B_16_imagenet1k', pretrained=True, num_classes=args.num_classes, image_size=args.image_size)

    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()

    params_grad = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(params_grad, lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params1:', n_parameters)

    train_loss_history, test_loss_history, val_loss_history = [], [], []
    best_signal_auc = 0
    best_signal_acc = 0
    best_person_auc = 0
    best_person_acc = 0

    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch:', epoch)
        start_time = time.time()
        train(model, optimizer, train_loader, train_loss_history)
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        acc, acc_statistic, macro_auc = evaluate(model, val_loader)
        acc_test, acc_statistic_test, macro_auc_test = evaluate(model, test_loader)

        best_person_auc = max(best_person_auc, macro_auc)
        best_person_acc = max(best_person_acc, acc_statistic)
        acc_is_best = acc_statistic >= best_person_acc


        print(" best_person_auc: {}".format(best_person_auc) + " best_person_acc: {}\n".format(best_person_acc))

        lr_scheduler.step(epoch=epoch)