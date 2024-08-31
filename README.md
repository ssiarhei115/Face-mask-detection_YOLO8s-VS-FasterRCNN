# FasterRCNN VS YOLOv8 for face mask detection

## Main goal
Compare two types of NN (Neural Network) detectors using the following criterias:
* mAP
* training time for the same number of epochs (50)
* weights file size

## Models
Model for comparison:
1) fasterrcnn_resnet50_fpn_v2
2) Yolo V8s

### fasterrcnn_resnet50_fpn_v2
Improved Faster R-CNN model with a ResNet-50-FPN backbone. The FPN (Region Proposal Network) module in Faster RCNN is responsible for giving an output of rectangular object proposals. ResNet (Residual Neural Network) is a deep neural network designed to solve the gradient decay problem. It uses the concept of “skip connections” or “residual connections” that allow information to flow directly from one layer to another, bypassing intermediate layers. This allows for training deeper networks with better performance.

### Yolo V8s
YOLO (You Only Look Once) is one of the most well-known models for real-time object detection in images. It is based on convolutional neural networks and allows achieving high processing speed without sacrificing accuracy. YOLO divides the image into a grid of cells and each cell predicts the boundaries and classes of objects contained within it. YOLOv8s - COCO Pretrained Models.

Both models are pretrained on <strong>COCO dataset</strong> which is highly popular for its diversity and large number of images across multiple categories.
* COCO contains 330K images, with 200K images having annotations for object detection, segmentation, and captioning tasks.
* The dataset comprises 80 object categories, including common objects like cars, bicycles, and animals, as well as more specific categories such as umbrellas, handbags, and sports equipment.
* Annotations include object bounding boxes, segmentation masks, and captions for each image.

Both model were used with pretrained weights, then finetuned for particular number of classes detection.


## Data description
The original dataset (https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/data) contains 853 images belonging to the 3 classes, as well as their bounding boxes in the PASCAL VOC format. The classes are:

* With mask;
* Without mask;
* Mask worn incorrectly.

This dataset was concatenated with https://www.kaggle.com/datasets/siarheis/face-mask-detection-add-incorr in order to enrich data with the least class 3 instances ('Mask worn incorrectly'). This extra dataset was created for this particular project. It contains 105 images annotated with Roboflow.


## Metric

Main metric used - mAP.
This metric was computed from scratch and compared with values provided by YOLO as reference.

### Important terms

Intersection Over Union (IOU)

Intersection Over Union (IOU) is a measure based on Jaccard Index that evaluates the overlap between two bounding boxes. It requires a ground truth bounding box and a predicted bounding box . By applying the IOU we can tell if a detection is valid (True Positive) or not (False Positive).

<img src='imgs/1.png'>

Some basic concepts used by the metrics:
* True Positive (TP): A correct detection. Detection with IOU ≥ threshold
* False Positive (FP): A wrong detection. Detection with IOU < threshold
* False Negative (FN): A ground truth not detected
* True Negative (TN): Does not apply. It would represent a corrected misdetection. 

***Precision x Recall curve***

The Precision x Recall curve is a good way to evaluate the performance of an object detector as the confidence is changed by plotting a curve for each object class. An object detector of a particular class is considered good if its precision stays high as recall increases, which means that if you vary the confidence threshold, the precision and recall will still be high. Another way to identify a good object detector is to look for a detector that can identify only relevant objects (0 False Positives = high precision), finding all ground truth objects (0 False Negatives = high recall).

A poor object detector needs to increase the number of detected objects (increasing False Positives = lower precision) in order to retrieve all ground truth objects (high recall). That's why the Precision x Recall curve usually starts with high precision values, decreasing as recall increases. You can see an example of the Prevision x Recall curve in the next topic (Average Precision). This kind of curve is used by the PASCAL VOC 2012 challenge and is available in our implementation.

***Average Precision***

Another way to compare the performance of object detectors is to calculate the area under the curve (AUC) of the Precision x Recall curve. As AP curves are often zigzag curves going up and down, comparing different curves (different detectors) in the same plot usually is not an easy task - because the curves tend to cross each other much frequently. That's why Average Precision (AP), a numerical metric, can also help us compare different detectors. In practice AP is the precision averaged across all recall values between 0 and 1.

The Precision x Recall curve is plotted by calculating the precision and recall values of the accumulated TP or FP detections. For this, first we need to order the detections by their confidences, then we calculate the precision and recall for each accumulated detection. Note that for recall computation, the denominator term ("Acc TP + Acc FN" or "All ground truths") is constant.

Calculating the interpolation performed in all points
By interpolating all points, the Average Precision (AP) can be interpreted as an approximated AUC of the Precision x Recall curve. The intention is to reduce the impact of the wiggles in the curve.

<img src='imgs/2.png'>

## Summary

YOLOv8 showed better results over all criteria used:

    * it spends much less time for training (6x advantage)
    * weights file of YOLO 22.5 MB VS 173 MB for FasterRCNN
    * YOLOv8 provides better mAP50 but this metric values for both models are similar and comparable

Futhermore AP per class & mAP50 computed from scratch are comparable to those values provided by YOLO8s   


## Libraries & tools used
* see the requirements
