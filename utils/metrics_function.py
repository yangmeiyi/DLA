from sklearn.preprocessing import label_binarize
from sklearn import metrics
from scipy import interp
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_curve


def list_onehot(actions: list, n: int):
    result = []
    for action in actions:
        result.append([int(k == action) for k in range(n)])
    return result

def ACC_2Clas_statistic(df):
    id_count = 0
    person_label = []
    person_preds = []
    person_preds_label = []
    for name, id_group in df:
        preds_pro = np.mean(id_group["preds"].values, axis=0)
        person_preds.append(list(preds_pro))
        preds_pro = float(np.argmax([preds_pro]))
        person_preds_label.append(preds_pro)
        labels = float(id_group["labels"].mean())
        person_label.append(labels)
        if int(preds_pro) == int(labels):
            id_count += 1
        # else:
        #     print("predicted error: ", name)
    acc_statistic = id_count / len(df)
    return person_preds, person_label, person_preds_label, acc_statistic


def ACC_3Clas_statistic(df):
    id_count = 0
    person_label = []
    person_preds = []
    person_preds_label = []
    for name, id_group in df:
        preds_pro = np.mean(id_group["preds"].values, axis=0)
        person_preds.append(list(preds_pro))
        preds_pro = float(np.argmax([preds_pro]))
        person_preds_label.append(preds_pro)
        labels = float(id_group["labels"].mean())
        person_label.append(labels)
        if int(preds_pro) == int(labels):
            id_count += 1
        # else:
        #     print("predicted error: ", name)
    acc_statistic = id_count / len(df)
    return person_preds, person_label, person_preds_label, acc_statistic




def ACC_8Clas_statistic(df):
    id_count = 0
    person_label = []
    person_preds = []
    person_preds_label = []
    for name, id_group in df:
        preds_pro = np.mean(id_group["preds"].values, axis=0)
        person_preds.append(list(preds_pro))
        preds_pro = float(np.argmax([preds_pro]))
        person_preds_label.append(preds_pro)
        labels = float(id_group["labels"].mean())
        person_label.append(labels)
        if int(preds_pro) == int(labels):
            id_count += 1
        # else:
        #     print("predicted error: ", name)
    acc_statistic = id_count / len(df)
    return person_preds, person_label, person_preds_label, acc_statistic

def macro_auc(y_true, person_preds, num):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num):
        roc_auc[i] = metrics.roc_auc_score(y_true[:, i], person_preds[:, i])
        print("class {} ".format(i) + 'statis auc ' + '{:.4f}'.format(roc_auc[i]))
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], person_preds[:, i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    # np.save('/home/dkd/Code/HCC_ICC/tu/result/B_our_fpr_hn.npy', fpr["macro"])
    # np.save('/home/dkd/Code/HCC_ICC/tu/result/B_our_tpr_hn.npy', tpr["macro"])
    roc_auc["macro"] = metrics.roc_auc_score(y_true, person_preds, average="macro")
    # plt.figure()
    # plt.plot(fpr["macro"], tpr["macro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    # plt.show()
    return roc_auc["macro"]


def AUC_2Clas_statistic(person_preds, person_label):
    # y_true = label_binarize(person_label, classes=[0, 1]) # using for three classification
    y_true = list_onehot(person_label, 2) # # using for thwo classification
    y_true = np.array(y_true)
    person_preds = np.array(person_preds)
    roc_auc = macro_auc(y_true, person_preds, num=2)
    # roc_auc = micro_auc(y_true, person_preds)
    print("macro_auc: ", roc_auc)
    return roc_auc

def Confusion_Mat_2Clas_statistic(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print("weighted_f1: ", f1_score(y_true, y_pred, average='weighted'))
    print("macro_f1: ", f1_score(y_true, y_pred, average='macro'))
    print("weighted_recall: ", recall_score(y_true, y_pred, labels=[0, 1], average='weighted'))
    print("macro_recall: ", recall_score(y_true, y_pred, labels=[0, 1], average='macro'))
    print("weighted_precision: ", precision_score(y_true, y_pred, labels=[0, 1], average='weighted'))
    print("macro_precision: ", precision_score(y_true, y_pred, labels=[0, 1], average='macro'))
    confusion_data = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("confusion matrix: \n", confusion_data)
    # plt.matshow(confusion_data, cmap=plt.cm.Reds)


def AUC_3Clas_statistic(person_preds, person_label):
    y_true = label_binarize(person_label, classes=[0, 1, 2]) # using for three classification
    # y_true = list_onehot(person_label, 2 ) # # using for thwo classification
    y_true = np.array(y_true)
    person_preds = np.array(person_preds)
    roc_auc = macro_auc(y_true, person_preds, num=3)
    # roc_auc = micro_auc(y_true, person_preds)
    print("macro_auc: ", roc_auc)
    return roc_auc

def Confusion_Mat_3Clas_statistic(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print("weighted_f1: ", f1_score(y_true, y_pred, average='weighted'))
    print("macro_f1: ", f1_score(y_true, y_pred, average='macro'))
    print("weighted_recall: ", recall_score(y_true, y_pred, labels=[0., 1., 2.], average='weighted'))
    print("macro_recall: ", recall_score(y_true, y_pred, labels=[0., 1., 2.], average='macro'))
    print("weighted_precision: ", precision_score(y_true, y_pred, labels=[0., 1., 2.], average='weighted'))
    print("macro_precision: ", precision_score(y_true, y_pred, labels=[0., 1., 2.], average='macro'))
    confusion_data = confusion_matrix(y_true, y_pred, labels=[0., 1., 2.])
    print("confusion matrix: \n", confusion_data)
    # plt.matshow(confusion_data, cmap=plt.cm.Reds)

def AUC_8Clas_statistic(person_preds, person_label):
    y_true = label_binarize(person_label, classes=[0, 1, 2, 3, 4, 5, 6, 7]) # using for three classification
    # y_true = list_onehot(person_label, 2 ) # # using for thwo classification
    y_true = np.array(y_true)
    person_preds = np.array(person_preds)
    roc_auc = macro_auc(y_true, person_preds, num=8)
    # roc_auc = micro_auc(y_true, person_preds)
    print("macro_auc: ", roc_auc)
    return roc_auc

def Confusion_Mat_8Clas_statistic(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print("weighted_f1: ", f1_score(y_true, y_pred, average='weighted'))
    print("macro_f1: ", f1_score(y_true, y_pred, average='macro'))
    print("weighted_recall: ", recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='weighted'))
    print("macro_recall: ", recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='macro'))
    print("weighted_precision: ", precision_score(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='weighted'))
    print("macro_precision: ", precision_score(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='macro'))
    confusion_data = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7])
    print("confusion matrix: \n", confusion_data)
    # plt.matshow(confusion_data, cmap=plt.cm.Reds)
