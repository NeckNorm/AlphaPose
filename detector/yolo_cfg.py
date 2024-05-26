from easydict import EasyDict as edict

# 스크립트의 절대 경로 얻기
cfg = edict()
cfg.CONFIG = '../AlphaPose/detector/yolo/cfg/yolov3-spp.cfg'
cfg.WEIGHTS = '../AlphaPose/detector/yolo/data/yolov3-spp.weights'
cfg.INP_DIM =  608
cfg.NMS_THRES =  0.6
cfg.CONFIDENCE = 0.1
cfg.NUM_CLASSES = 80
