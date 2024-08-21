import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import pandas as pd
from PIL import Image as pilim, ImageDraw
import torch
import matplotlib

def plot_img_bbox(img, box_source):
    """ Вспомогательная функция для визуализации bbox

    Args:
        img (_type_): _description_
        target (_type_): _description_
    """
    num = len(box_source)
    fig, a = plt.subplots(1, num, figsize=(num*5,6))
    if num==1:
        a = np.array([a])
    #fig.set_size_inches(5,5)
    #fig = plt.figure(figsize=(num*4,5))
    #a = fig.add_subplot(111)
    for (n,target), title in zip(enumerate(box_source), ['Target', 'Prediction', 'NMS prediction']):
        a[n].imshow(img)
        for box in (target['boxes']):
            x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
            rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

            # Отрисуем bbox поверх картинки
            a[n].add_patch(rect)
        a[n].set_title(title)
    plt.show()


def get_transform(train):
    
    if train:
        return A.Compose([
                            A.HorizontalFlip(0.5),
                            A.RandomBrightnessContrast(p=0.2),
                            #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                            A.Rotate(limit=(-90, 90)),
                            ToTensorV2(p=1.0) 
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    

def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
   
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def apply_nms(orig_prediction, iou_thresh=0.3):
    """вспомогательная функция принимает исходный прогноз и порог iou    

    Args:
        orig_prediction (_type_): _description_
        iou_thresh (float, optional): _description_. Defaults to 0.3.

    Returns:
        _type_: _description_
    """
    
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction


def torch_to_pil(img):
    return transforms.ToPILImage()(img).convert('RGB')


def plot_image(img_tensor, annotation,predict=True):
    
    fig,ax = plt.subplots(1)
    #fig.set_size_inches(18.5, 10.5)
    fig.set_size_inches(8, 6)
    img = img_tensor.cpu().data
    mask_dic = {1:'wo_mask', 2:'w_mask', 3:'mask_incorr'}

    ax.imshow(img.permute(1, 2, 0))
    
    for i,box in enumerate(annotation["boxes"]):
        xmin, ymin, xmax, ymax = box

        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

        ax.add_patch(rect)
        label = mask_dic[int(annotation['labels'][i].data)]
        if predict:
            score = int((annotation['scores'][i].data) * 100)
            ax.text(xmin, ymin, f"{label} : {score}%", horizontalalignment='center', verticalalignment='center',fontsize=10,color='b')
        else:
            score=''
            ax.text(xmin, ymin, f"{label}", horizontalalignment='center', verticalalignment='center',fontsize=10,color='b')
    plt.show()


def plot_image_2(img_tensor, box_source,predict_bool=[True, False]):
    
    #fig,ax = plt.subplots(1)
    #fig.set_size_inches(18.5, 10.5)
    #fig.set_size_inches(8, 6)
    
    num = len(box_source)
    fig, a = plt.subplots(1, num, figsize=(num*7,8))
    if num==1:
        a = np.array([a])
    img = img_tensor.cpu().data
    mask_dic = {1:'WO_mask', 2:'W_mask', 3:'mask_INCORR'}

    for (n, source), pred in zip(enumerate(box_source), predict_bool):
        a[n].imshow(img.permute(1, 2, 0))
    
        for i,box in enumerate(source["boxes"]):
            xmin, ymin, xmax, ymax = box

            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

            a[n].add_patch(rect)
            label = mask_dic[int(source['labels'][i].data)]
            if pred:
                score = int((source['scores'][i].data) * 100)
                a[n].text(xmin, ymin, f"{label} : {score}%", horizontalalignment='center', verticalalignment='center',fontsize=10,color='orange')
                a[n].set_title('NMS prediction')
            else:
                score=''
                a[n].text(xmin, ymin, f"{label}", horizontalalignment='center', verticalalignment='center',fontsize=10,color='b')
                a[n].set_title('Target')
    plt.show()


def get_scores_(annotation, prediction, threshold):
    
    s_precision = dict()
    s_iou= dict()
    s_recall = dict()

    for label in set(list(annotation['labels'].numpy()) + list(prediction['labels'].numpy())):#set(annotation['labels'].numpy()):
        
        targ = annotation['labels'].numpy()
        pred = prediction['labels'].numpy()
        ids_targ = np.asarray(targ==label).nonzero()
        ids_pred = np.asarray(pred==label).nonzero()
        
        m = torchvision.ops.complete_box_iou(
            annotation['boxes'][ids_targ], 
            prediction['boxes'][ids_pred]
            ).numpy()
        
        TP = (m > threshold).sum()
        FP = len(ids_pred[0]) - TP
        FN = len(ids_targ[0]) - TP
        print(label,TP, FP, FN)
        s_iou[label] = m[m>0].sum()/len(ids_pred[0]) if len(ids_pred[0]) != 0 else 0
        s_precision[label] = TP / (TP + FP) if (TP + FP) != 0 else 0
        s_recall[label] = TP / (TP + FN) if (TP + FN) != 0 else 0
    
    return s_iou, s_precision, s_recall #dict((k,statistics.mean(v)) for k,v in id_iou_dict.items()) 


def get_scores2(df, annotation, prediction, threshold):
    
    for label in set(list(prediction['labels'].numpy())):#set(annotation['labels'].numpy()):
        
        targ = annotation['labels'].numpy()
        pred = prediction['labels'].numpy()
        ids_targ = np.asarray(targ==label).nonzero()
        ids_pred = np.asarray(pred==label).nonzero()
        #print(targ)
        #print(ids_targ)

        m = torchvision.ops.complete_box_iou(
            annotation['boxes'][ids_targ], 
            prediction['boxes'][ids_pred]
            ).numpy()
        
        for n,item in enumerate(pred[ids_pred]):        
            entry = {'TP': [0], 'FP': [0], 'label': label, 'threshold': threshold, 'GT': len(ids_targ[0])}
            entry['score'] = prediction['scores'][ids_pred][n].item()
            entry['id'] = str(annotation['image_id'].item()  )
            try:
                iou = m[:,n][np.argmax(m[:,n])]
            except:
                iou = 0
            entry['iou'] = iou
            if iou > threshold and iou == np.max(m[np.argmax(m[:,n])]):
                entry['TP'] = [1]
            else:
                entry['FP'] = [1]
            
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) #print(entry)
        #print(m)
        
        
    return df #dict((k,statistics.mean(v)) for k,v in id_iou_dict.items()) 


def get_scores3(df, annotation, prediction, threshold, yolov=False):
    if yolov:
        prediction_df = prediction.pandas().xyxy[0]
        if len(prediction_df)==0:
            return df
        #print(prediction_df)
        prediction = dict()
        prediction['boxes'] = torch.tensor(prediction_df[['xmin', 'ymin', 'xmax', 'ymax']].values)
        prediction['scores'] = torch.tensor(prediction_df['confidence'].values)
        prediction['labels'] = torch.tensor(prediction_df['class'].values)
        #print(prediction)

    for label in set(list(prediction['labels'].numpy())):#set(annotation['labels'].numpy()):
        
        targ = annotation['labels'].numpy()
        pred = prediction['labels'].numpy()
        ids_targ = np.asarray(targ==label).nonzero()
        ids_pred = np.asarray(pred==label).nonzero()
        #print(targ)
        #print(ids_targ)

        m = torchvision.ops.complete_box_iou(
            annotation['boxes'][ids_targ], 
            prediction['boxes'][ids_pred]
            ).numpy()
        
        for n,item in enumerate(pred[ids_pred]):        
            entry = {'TP': [0], 'FP': [0], 'label': label, 'threshold': threshold, 'GT': len(ids_targ[0])}
            entry['score'] = prediction['scores'][ids_pred][n].item()
            entry['id'] = str(annotation['image_id'].item()  )
            try:
                iou = m[:,n][np.argmax(m[:,n])]
            except:
                iou = 0
            entry['iou'] = iou
            if iou > threshold and iou == np.max(m[np.argmax(m[:,n])]):
                entry['TP'] = [1]
            else:
                entry['FP'] = [1]
            
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True) #print(entry)
        #print(m)
        
        
    return df #dict((k,statistics.mean(v)) for k,v in id_iou_dict.items()) 


def get_AP(precision, recall):
    
    # Initialize variables
    AP = 0
    max_precision = 0
    prev_recall = 0
    
    # Calculate AP
    for i in range(len(recall)):
        if precision[i] > max_precision:
            max_precision = precision[i]
        AP += max_precision * (recall[i] - prev_recall)
        prev_recall = recall[i]

    return AP


def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w,h = image.size

    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    #print(transformed_annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h

    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]

    mask_dic = {1:'WO_mask', 2:'W_mask', 3:'mask_INCORR'}
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        print(w,h, obj_cls, x0, y0, x1, y1)
        plotted_image.rectangle(((x0,y0), (x1,y1)), width=1)
        plotted_image.text((x0, y0 - 10), mask_dic[(int(obj_cls))], fill="red")


    plt.imshow(np.array(image))
    plt.show()


def get_yolo(path1, path2):
    be = plt.get_backend()
    model = torch.hub.load(path1, 'custom', path1+path2, source='local')
    matplotlib.use(be)
    return model