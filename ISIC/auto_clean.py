import os
import torch
import torch.nn as nn
import time
from einops import rearrange, repeat
# from compare.focal_loss import FocalLoss
# from compare.labelsmoothing_loss import LabelSmoothingCrossEntropy
# from compare.symmetric_loss import SymmetricCrossEntropyLoss
# from compare.asymmetric_loss import AsymmetricCrossEntropyLoss
# from compare.Curriculum_loss import CurriculumLoss
# from compare.active_passive_loss import ActivePassiveLoss
from utils import accuracy
import random
import argparse
from pytorch_pretrained_vit import ViT
from torch.utils.data import DataLoader
from dataloader_isic import CancerSeT_CSV
from utils.metrics_function import ACC_2Clas_statistic, AUC_2Clas_statistic,  Confusion_Mat_2Clas_statistic
import numpy as np
import pandas as pd
from NA_Loss import CustomLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = ViT('B_16_imagenet1k', pretrained=True, num_classes=256, image_size=224)


def parse_args():
    parser = argparse.ArgumentParser(description="Training a ViT model for cancer detection.")

    # Paths and Data
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
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for the optimizer.")

    # Model and Training Settings
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers for data loading.")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="Comma-separated list of GPU IDs to use.")
    parser.add_argument("--milestones", type=int, nargs='+', default=[25, 40],
                        help="Milestones for learning rate scheduler.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Learning rate decay factor.")

    return parser.parse_args()

def df_result_save(df):
    person_preds = []
    person_label = []
    name_list = []
    df_group = df.groupby('people_id')[['labels', 'preds']]
    for name, id_group in df_group:
        pred_pro = np.mean(id_group["preds"].values, axis=0)
        preds_pro = float(np.argmax([pred_pro]))
        person_preds.append(preds_pro)
        labels = float(id_group["labels"].mean())
        person_label.append(labels)
        name_list.append(name)
    assert len(name_list) == len(person_label) == len(person_preds)
    dataframe = pd.DataFrame({"patient": name_list, "pred": person_preds, "label": person_label})
    dataframe.to_csv("doctor_BM.csv", index=False, sep=",")


def ACC_error_statistic(df):
    error_name = []
    error_pred = []
    error_label = []
    person_label = []
    person_preds = []
    person_preds_label = []
    df_group = df.groupby('people_id')[['labels', 'preds']]
    for name, id_group in df_group:
        pred_pro = np.mean(id_group["preds"].values, axis=0)
        person_preds.append(list(pred_pro))
        preds_pro = float(np.argmax([pred_pro]))
        person_preds_label.append(preds_pro)
        labels = float(id_group["labels"].mean())
        person_label.append(labels)
        if int(preds_pro) != int(labels):
            if labels == 1 and preds_pro == 2:
                print(name)
            for index, row in id_group.iterrows():
                pred = float(np.argmax(row["preds"]))
                label = row["labels"]
                if int(pred) != int(label):
                    error_name.append(name)
                    error_pred.append(pred)
                    error_label.append(label)
    assert len(error_name) == len(error_label) == len(error_pred)
    # dataframe = pd.DataFrame({"patient": error_name, "pred": error_pred, "label": error_label})
    # dataframe.to_csv("ccrcc_error.csv", index=False, sep=",")




def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class CustomViTModel(nn.Module):
    def __init__(self, num_classes=2, image_size=224, patch_size=16, hiddden_size=256):
        super(CustomViTModel, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.vit = vit_model
        self.fc1 = nn.Linear(hiddden_size, num_patches + 1)  # FC layer 2
        self.fc2 = nn.Linear(num_patches + 1, num_classes)  # FC layer 1


    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        fc1_output = self.fc1(outputs)
        fc2_output = self.fc2(fc1_output)
        return fc1_output, fc2_output


masking_ratio = 0.15

def train(model, train_loader, criterion, criterion_patch, optimizer, epoch, epochs):
    model.train()
    for batch_idx, data_img in enumerate(train_loader):
        inputs = data_img["img"].float()
        labels = data_img["labels"].float()
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass for image classification
        _, outputs_classification = model(inputs)
        # outputs_classification = model(inputs)

        loss_classification = criterion(outputs_classification, labels.long(), epoch, epochs)
        # curriculum_weight = min(1.0, (epoch + 1) / epochs)
        # loss_classification = criterion(outputs_classification, labels.long())

        # mask patch position classification
        tokens = model.module.vit.patch_embedding(inputs)
        tokens = rearrange(tokens, 'b d w h -> b (w h) d')
        batch, num_patches, *_ = tokens.shape

        # # Masking
        num_masked = int(masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens_masked = tokens.clone()
        tokens_masked[batch_range, masked_indices] = 0.0
        tokens_masked = tokens_masked.to(device)
        attended_tokens = model.module.vit.transformer(tokens_masked)
        mlp_tokens = model.module.vit.fc(model.module.vit.norm(attended_tokens))
        patch_logits = rearrange(model.module.fc1(mlp_tokens), 'b n d -> (b n) d')
        # Define labels
        patch_labels = repeat(torch.arange(num_patches, device=device), 'n -> (b n)', b=batch)
        loss_position = criterion_patch(patch_logits, patch_labels)

        # # Combined loss
        # loss = loss_classification
        loss = loss_classification + loss_position

        # Backpropagation and optimization
        acc = accuracy(outputs_classification.data, labels.data, topk=(1,))
        acc = acc[0].item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # if batch_idx % 50 == 0:
        #     print('[' + '{:5}'.format(batch_idx * len(data_img)) + '/' + '{:5}'.format(len(train_loader.dataset)) +
        #           ' (' + '{:3.0f}'.format(100 * batch_idx / len(train_loader)) + '%)]  Loss: ' +
        #           '{:6.4f}'.format(loss) + ' class Loss: ' + '{:6.4f}'.format(loss_classification) +
        #            ' patch Loss: ' + '{:6.4f}'.format(loss_position) + ' Acc:' + '{:6.2f}'.format(acc))
        if batch_idx % 50 == 0:
            print('[' + '{:5}'.format(batch_idx * len(data_img)) + '/' + '{:5}'.format(len(train_loader.dataset)) +
                  ' (' + '{:3.0f}'.format(100 * batch_idx / len(train_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss) + ' class Loss: ' + '{:6.4f}'.format(loss_classification) + ' Acc:' + '{:6.2f}'.format(acc))


def test(model, test_loader):
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
            inputs, labels = inputs.to(device), labels.long().to(device)
            _, outputs = model(inputs)
            # outputs = model(inputs)
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



def Data_Cleaned(model, optimizer, lr_scheduler, criterion, criterion_patch, args):

    # Load dataset
    Liver_loader_train = CancerSeT_CSV(args.train_path, 'train')
    Liver_loader_val = CancerSeT_CSV(args.val_path, 'val')
    Liver_loader_test = CancerSeT_CSV(args.val_path, 'test')

    # Define data loaders for the unlabeled pool and test set
    train_loader = torch.utils.data.DataLoader(Liver_loader_train, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(Liver_loader_val, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True)
    test_loader = torch.utils.data.DataLoader(Liver_loader_test, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    for epoch in range(0, args.epochs):
        print("Epoch: ", epoch)
        train(model, train_loader, criterion, criterion_patch, optimizer, epoch, args.epochs)
        print("Val Result:")
        test(model, val_loader)
        print("Test Result:")
        test(model, test_loader)
        lr_scheduler.step()




def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    print(f"Using GPUs {args.gpu_ids}")


    # Initialize our model and optimizer
    model = CustomViTModel()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = CustomLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    # criterion = LabelSmoothingCrossEntropy()
    # criterion = SymmetricCrossEntropyLoss()
    # criterion = CurriculumLoss()
    # criterion = ActivePassiveLoss()
    criterion_patch = nn.CrossEntropyLoss()
    Data_Cleaned(model, optimizer, lr_scheduler, criterion, criterion_patch, args)



if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("time cost: ", end-start)









