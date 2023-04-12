import os
import warnings
from typing import Tuple, Sequence, Callable
import json
import random
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
from pytorch_lightning.loggers import WandbLogger
import wandb
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

warnings.filterwarnings('ignore')


class ConstDataset(Dataset):
    def __init__(
        self,
        image_dir: os.PathLike,
        label_path: os.PathLike,
        transforms: Sequence[Callable]=None
    ) -> None:
        self.image_dir = image_dir
        self.label_path = label_path
        self.transforms = transforms

        with open(self.label_path, 'r') as f:
            annots = json.load(f)
        
        self.annots = annots

    def __len__(self) -> int:
        return len(self.annots['images'])
    
    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.annots['images'][index]
        file_name = os.path.join(self.image_dir, image_id['file_name'])
        image = Image.open(file_name).convert('RGB')
        image = np.array(image)

        annots = [x for x in self.annots['annotations'] if x['image_id'] == image_id['id']]
        boxes = np.array([annot['bbox'] for annot in annots], dtype=np.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.array([annot['category_id'] for annot in annots], dtype=np.int64)

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)
            transformed_img = transformed['image']
            transformed_bbox = transformed['bboxes']
            transformed_label = transformed['class_labels']


        transformed_img = transformed_img.transpose(2,0,1) # hwc to hwc
        transformed_img = transformed_img / 255.0  # 0-1
        transformed_img = torch.Tensor(transformed_img)


        target = {
            'boxes': transformed_bbox,
            'labels': transformed_label
        }

        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)

        return transformed_img, target


class TrainingModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = create_model()

    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
        # return optim.Adam(self.model.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        self.log(
            'loss', losses, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(
            'class_loss', loss_dict['loss_classifier'], on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(
            'box_loss', loss_dict['loss_box_reg'], on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(
            'obj_loss', loss_dict['loss_objectness'], on_step=True, on_epoch=False, prog_bar=True
        )

        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        metric.update(outputs, targets)
        return outputs
    
    def validation_epoch_end(self, outputs):
        metric_compute = metric.compute()
        map = metric_compute['map'].numpy().tolist()
        map_50 = metric_compute['map_50'].numpy().tolist()
        map_75 = metric_compute['map_75'].numpy().tolist()
        
        self.log(
            'val_mAP', map, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            'val_mAP_50', map_50, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            'val_mAP_75', map_75, on_step=False, on_epoch=True, prog_bar=True
        )
        metric.reset()
    
    def predict_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        return outputs


class DataModule(pl.LightningDataModule):
    def __init__(self,
        train_path, 
        train_annt_path, 
        valid_path, 
        valid_annt_path, 
        batch_size,
        num_workers):
        super().__init__()
        self.train_path = train_path
        self.train_annt_path = train_annt_path
        self.valid_path = valid_path
        self.valid_annt_path = valid_annt_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        self.valid_transforms = A.Compose([
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


    def setup(self, stage: str):
        if stage == "fit":
            self.trainset = ConstDataset(self.train_path, self.train_annt_path, self.train_transforms)
            self.testset = ConstDataset(self.valid_path, self.valid_annt_path, self.valid_transforms)
        if stage == 'predict':
            self.testset = ConstDataset(self.valid_path, self.valid_annt_path, self.valid_transforms)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_fn, shuffle=True, drop_last=True, pin_memory=True)
        
    def val_dataloader(self):
        return DataLoader(self.testset, batch_size=int(self.batch_size/4), num_workers=self.num_workers, collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.testset, batch_size=int(self.batch_size), num_workers=self.num_workers, collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)
        

def create_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    min_size = 600
    max_size = 1000
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    model.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def run(args):
    pl.seed_everything(42)
    
    data_module = DataModule(
        train_path = args.train_path, 
        train_annt_path = args.train_annt_path, 
        valid_path = args.valid_path, 
        valid_annt_path = args.valid_annt_path, 
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )

    model = TrainingModule()
    
    if args.mode == 'train':
        ckpt_path = args.weight_path
        num_gpus = args.gpus
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_mAP',
            dirpath=ckpt_path,
            filename='{epoch}-{val_mAP:.2f}',
            save_top_k=-1,
            mode='max',
            save_weights_only=True,
        )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_mAP',
            min_delta=0.00,
            patience=100,
            verbose=False,
            mode='min'
        )
        
        trainer = pl.Trainer(
            log_every_n_steps=1,
            logger=wandb_logger,
            max_epochs=args.max_epochs,
            accelerator="gpu",
            strategy='ddp_find_unused_parameters_false',
            gpus = num_gpus,
            precision=16,
            # callbacks=[checkpoint_callback, early_stopping_callback]
            callbacks=[checkpoint_callback]
        )

        trainer.fit(model, data_module)
        
    if args.mode == 'test':
        model = model.load_from_checkpoint(args.checkpoint)

        trainer = pl.Trainer(
            gpus=1,
            precision=16
        )
        
        output_lists = trainer.predict(model, data_module)
        outputs = []
        for i in range(len(output_lists)):
            outputs += output_lists[i]
        # outputs = outputs[0]+ outputs[1]
        print(len(outputs))
        
        
        with open(args.valid_annt_path, "r") as f:
            annots = json.load(f)

        images = annots['images']

        image_ids = []
        for image in images:
            image_id = image['id']
            image_ids.append(image_id)
        # eval
        predicts = []
        idx = 0
        for i in range(len(outputs)):
            # print(outputs[i])
            boxes = outputs[i]['boxes'].detach().cpu().numpy().tolist()
            scores = outputs[i]['scores'].detach().cpu().numpy().tolist()
            labels = outputs[i]['labels'].detach().cpu().numpy().tolist()

            for bbox,label,score in zip(boxes,labels,scores):
                # print(bbox,label,score)
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]
                tmp = {"image_id": int(image_ids[idx]), "category_id": int(label), "bbox": bbox, "score": float(score)}
                predicts.append(tmp)
            idx += 1

        with open('predict.json', 'w') as f:
            json.dump(predicts, f)
        
        coco_gt = COCO(args.valid_annt_path)
        coco_pred = coco_gt.loadRes('predict.json')
        coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster-RCNN')
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--project', type=str, default='my_faster')
    parser.add_argument('--weight_path', type=str, default='weights/faster/')
    parser.add_argument('--train_path', type=str, default='data/images/train/')
    parser.add_argument('--train_annt_path', type=str, default='data/annotations/train_annotations.json')
    parser.add_argument('--valid_path', type=str, default='data/images/valid/')
    parser.add_argument('--valid_annt_path', type=str, default='data/annotations/valid_annotations.json')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpus', type=list, default=[0,1,2,3])
    parser.add_argument('--checkpoint', type=str, default='coco_weights/faster/best.ckpt')
    args = parser.parse_args()
    
    if args.mode == 'train':
        wandb_logger = WandbLogger(project=args.project)
        metric = MeanAveragePrecision()
        
    run(args)