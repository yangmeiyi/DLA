# A Dynamic Learning Algorithm for Medical Image Tasks through Adaptive Attention to Uncertain Samples

by Meiyi Yang


## Introduction
Noisy and ambiguous data in uncertain samples lead to significant challenges during the training process. To reduce the impact of noise samples and improve the robustness of the model, we propose a dynamic learning algorithm incorporating both data detection and a dynamic learning strategy. This algorithm is intended to regulate the model's concentration on uncertain samples. This dynamic learning algorithm incorporates 'dynamic probability' and 'weight decay', facilitating the dynamic adjustment of sample weights across different training stages. Moreover, we introduce an auxiliary task to provide additional supervision signals, assisting the model in better understanding spatial structures and local features within images. Extensive experiments on diverse medical datasets, including clear cell renal cell carcinom clear cell renal cell carcinomaa (CCRCC), liver cancer (Liver), endometrial (EndoM), breast cancer (BreakHis), endoscopic images (Kvasir), and dermatoscopy diagnostic datasets (ISIC), highlight the effectiveness of the proposed method. We also evaluated its performance in lesion detection.

<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/workflow.png" width="1000" height="850" /></div>




## Content
- DLA
  - ISIC (training for ISIC dataset)
    - 📄dataloader_isic.py  (Used for deep learning to load image data)
    - 📄NA_Loss.py(Our loss)
    - 📄auto_clean.py  (Training file for distinguishing between benign and malignant lesions)
    - 📄vit_base.py(the basline)
  - 📁utils(Used for deep learning to load image data)
- 📄Readme.md (help)


## Code 

### Requirements
* Ubuntu (It's only tested on Ubuntu, so it may not work on Windows.)
* Python >= 3.9.7
* PyTorch >= 1.12.1
* torchvision >= 0.13.1
* scikit-learn >=1.1.3
* scipy >= 1.9.3
* numpy >=1.23.3

### Data Preparation

1. Preprocess the data. The default model takes images of size 224x 224.

2. The data files include train.csv test.csv, and val.csv, with formats including

   ```python
   { "id": id, "labels": label}
   ```

3. Create a patient dictionary. This should be a pickle file containing a dict as follows, where img is the image matrix of slice :

   ```python
   dataset = {
       "img": img_data,
       "labels": torch.Tensor([y])[0],
       "id": id,
       "cancer": cancer,
       "image_path": str(self.pic_files[index])
   }
   ```

### Parameters
| Parameters | Value |
|-----------|:---------:|
| image size | 224 | 
| batch size | 64 |
| Initial learning rate | 0.01 | 
| Epoches | 50 | 
| Schedule | [25, 40] | 
| Weight decay | 0.0005 | 
| momentum | 0.9 |
| Optimizer | optim.SGD | 
| Criterion | CrossEntropyLoss | 


### Usage
```
cd ./DLA/ISIC/
python3 auto_clean.py
```


### Category Metrics
* Accuracy
* Area under the receiver operating characteristic curve (AUC)
* Recall
* Precision
* F1 score


### result
<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/workflow.png" width="1000" height="850" /></div>
<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/workflow.png" width="1000" height="850" /></div>
<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/workflow.png" width="1000" height="850" /></div>
<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/workflow.png" width="1000" height="850" /></div>
<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/workflow.png" width="1000" height="850" /></div>
<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/workflow.png" width="1000" height="850" /></div>









