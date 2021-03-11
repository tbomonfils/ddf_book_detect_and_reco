import sys
import argparse

import tensorflow as tf

from dev.python.Yolo import model_architecture as m_a
from dev.python.Yolo import train as t


def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for using ScaledYOLOv4.')
    #save model
    parser.add_argument('--output-model-dir', default='./output_model')
    #training
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--start-eval-epoch', default=10, type=int)
    parser.add_argument('--eval-epoch-interval', default=1)
    #model
    parser.add_argument('--model-type', default='tiny', help="choices=['tiny','p5','p6','p7']")
    parser.add_argument('--use-pretrain', default=False, type=bool)
    parser.add_argument('--tiny-coco-pretrained-weights',
                        default='./pretrain/ScaledYOLOV4_tiny_coco_pretrain/coco_pretrain')
    parser.add_argument('--p5-coco-pretrained-weights',
                        default='./pretrain/ScaledYOLOV4_p5_coco_pretrain/coco_pretrain')
    parser.add_argument('--p6-coco-pretrained-weights',
                        default='./pretrain/ScaledYOLOV4_p6_coco_pretrain/coco_pretrain')
    parser.add_argument('--checkpoints-dir', default='./checkpoints',help="Directory to store  checkpoints of model during training.")
    #loss
    parser.add_argument('--box-regression-loss', default='ciou',help="choices=['giou','diou','ciou']")
    parser.add_argument('--classification-loss', default='bce', help="choices=['ce','bce','focal']")
    parser.add_argument('--focal-alpha', default= 0.25)
    parser.add_argument('--focal-gamma', default=2.0)
    parser.add_argument('--ignore-thr', default=0.7)
    parser.add_argument('--reg-losss-weight', default=0.05)
    parser.add_argument('--obj-losss-weight', default=1.0)
    parser.add_argument('--cls-losss-weight', default=0.5)
    #dataset
    parser.add_argument('--dataset-type', default='voc', help="voc,coco")
    parser.add_argument('--num-classes', default=20)
    parser.add_argument('--class-names', default='voc.names', help="voc.names,coco.names")
    parser.add_argument('--dataset', default='/home/wangem1/dataset/VOC2007_2012')#
    parser.add_argument('--voc-train-set', default='VOC2007,trainval,VOC2012,trainval')
    parser.add_argument('--voc-val-set', default='VOC2007,test')
    parser.add_argument('--voc-skip-difficult', default=True)
    parser.add_argument('--coco-train-set', default='train2017')
    parser.add_argument('--coco-valid-set', default='val2017')
    '''
    voc dataset directory:
        VOC2007
                Annotations
                ImageSets
                JPEGImages
        VOC2012
                Annotations
                ImageSets
                JPEGImages
    coco dataset directory:
        annotations/instances_train2017.json
        annotations/instances_val2017.json
        images/train2017
        images/val2017
    '''
    parser.add_argument('--augment', default='mosaic',help="choices=[None,'only_flip_left_right','ssd_random_crop','mosaic']")
    parser.add_argument('--multi-scale', default='416',help="Input data shapes for training, use 320+32*i(i>=0)")#896
    parser.add_argument('--max-box-num-per-image', default=100)
    #optimizer
    parser.add_argument('--optimizer', default='sgd', help="choices=[adam,sgd]")
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--nesterov', default=True)
    parser.add_argument('--weight-decay', default=5e-4)
    #lr scheduler
    parser.add_argument('--lr-scheduler', default='warmup_cosinedecay', type=str, help="choices=['step','warmup_cosinedecay']")
    parser.add_argument('--init-lr', default=1e-3, type=float)
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--lr-decay-epoch', default=[160, 180])
    parser.add_argument('--warmup-epochs', default=10, type=int)
    parser.add_argument('--warmup-lr', default=1e-6, type=float)
    #postprocess
    parser.add_argument('--nms', default='diou_nms', help="choices=['hard_nms','diou_nms']")
    parser.add_argument('--nms-max-box-num', default=300)
    parser.add_argument('--nms-iou-threshold', default=0.2, type=float)
    parser.add_argument('--nms-score-threshold', default=0.01, type=float)
    #anchor
    parser.add_argument('--anchor-match-type', default='wh_ratio',help="choices=['iou','wh_ratio']")
    parser.add_argument('--anchor-match-iou_thr', default=0.2, type=float)
    parser.add_argument('--anchor-match-wh-ratio-thr', default=4.0, type=float)

    parser.add_argument('--label-smooth', default=0.0, type=float)
    parser.add_argument('--scales-x-y', default=[2., 2., 2., 2., 2.])
    parser.add_argument('--accumulated-gradient-num', default=1, type=int)

    parser.add_argument('--tensorboard', default=True, type=bool)

    return parser.parse_args(args)
	
if __name__ == "__main__":
	args = args = parse_args(sys.argv[1:])
	model = m_a.Yolov4_tiny(args, training=True)
	t.train_model(model, tf.keras.losses.BinaryCrossentropy(), nb_epoch=20, batch_size=args.batch_size)