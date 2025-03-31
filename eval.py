import numpy as np
import torch
from train_app.loss.metrics import IoU, PixelAccuracy, AP, AF1, AR

class Evaluator(object):
    def __init__(self, num_class):
        kwargs = {'device_fn': lambda x:x,
                  'log_fn':lambda x:x,
                  'log_image_fn':lambda x:x,
                  'log_prefix_fn':lambda x:x}
        self.iou = IoU(K=num_class, **kwargs)
        self.pix = PixelAccuracy(**kwargs)
        self.map = AP(K=num_class, **kwargs)
        self.mar = AR(K=num_class, **kwargs)
        self.maf1 = AF1(K=num_class, **kwargs)

        self.num_class = num_class
        self.scores = {
            "IoU": [],
            "PixelAccuracy": [],
            "AP": [],
            "AR": [],
            "AF1": [],
            "IoU_dict": {
                "background": [],
                "Building_No_Damage": [],
                "Building_Minor_Damage": [],
                "Building_Major_Damage": [],
                "Building_Total_Destruction": [],
                "Vehicle": [],
                "Road": [],
                "Tree": []
            }
        }
        self.eps = 1e-8

    @property
    def Precision(self):
        return self.scores["AP"]

    @property
    def Recall(self):
        return self.scores["AR"]

    @property
    def F1(self):
       return self.scores["AF1"]

    @property
    def Intersection_over_Union(self):
        return self.scores["IoU"]

    @property
    def pixelAccuracy(self):
        return self.scores["PixelAccuracy"]

    @property
    def IoU_dict(self):
        return self.scores["IoU_dict"]
    
    @property
    def mIoU(self):
        return np.mean(list(self.scores["IoU_dict"].values()))

    def add_batch(self, gt_image, pre_image):
        # assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
        #                                                                                          gt_image.shape)
        self.scores["IoU"].append(self.iou(pre_image, gt_image).cpu().item())
        self.scores["AP"].append(self.map(pre_image, gt_image))
        self.scores["AR"].append(self.mar(pre_image, gt_image))
        self.scores["AF1"].append(self.maf1(pre_image, gt_image))
        self.iou_per_class(pre_image, gt_image)
        self.scores["PixelAccuracy"].append(self.pix(pre_image, gt_image).cpu().item())

    def eval(self):
        
        iou_dict = {k: np.mean(v) for k, v in self.scores["IoU_dict"].items()}
        self.scores = {k: np.mean(v) for k, v in self.scores.items() if "dict" not in k}
        self.scores["IoU_dict"] = iou_dict
    
    def iou_per_class(self, pre_image, gt_image):
        pre_image_ = torch.argmax(torch.softmax(pre_image, dim=1), dim=1)
        
        for i in range(self.num_class):
            gt_image_cls = (gt_image == i).float()
            pre_image_binary = (pre_image_ == i).float()
            
            intersection = torch.sum(gt_image_cls * pre_image_binary)
            union = torch.sum(gt_image_cls) + torch.sum(pre_image_binary) - intersection
            
            iou = ((intersection + self.eps) / (union + self.eps)).item()
            
            class_name = list(self.scores["IoU_dict"].keys())[i]
            self.scores["IoU_dict"][class_name].append(iou)
