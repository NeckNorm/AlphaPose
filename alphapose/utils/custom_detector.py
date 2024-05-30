import os
import sys
from threading import Thread
from queue import Queue

import cv2
import numpy as np

import torch
import torch.multiprocessing as mp

from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL
from alphapose.models import builder

class DetectionLoader():
    def __init__(self, detector, cfg, opt, queueSize=128):
        self.cfg = cfg
        self.opt = opt
        self.device = opt.device
        self.detector = detector
        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE
        self._sigma = cfg.DATA_PRESET.SIGMA

        # SimpleTransform 설정
        pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        self.transformation = SimpleTransform(
            pose_dataset, scale_factor=0,
            input_size=self._input_size,
            output_size=self._output_size,
            rot=0, sigma=self._sigma,
            train=False, add_dpg=False, gpu_device=self.device)
        
        # initialize the queue used to store data
        """
        image_queue: the buffer storing pre-processed images for object detection
        det_queue: the buffer storing human detection results
        pose_queue: the buffer storing post-processed cropped human image for pose estimation
        """
        if opt.sp:
            self._stopped = False
            self.image_queue = Queue(maxsize=queueSize)
            self.det_queue = Queue(maxsize=10 * queueSize)
            self.pose_queue = Queue(maxsize=10 * queueSize)
        else:
            self._stopped = mp.Value('b', False)
            self.image_queue = mp.Queue(maxsize=queueSize)
            self.det_queue = mp.Queue(maxsize=10 * queueSize)
            self.pose_queue = mp.Queue(maxsize=10 * queueSize)

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        image_detection_worker = self.start_worker(self.image_detection)
        image_postprocess_worker = self.start_worker(self.image_postprocess)

        return [image_detection_worker, image_postprocess_worker]

    def stop(self):
        # clear queues
        self.clear_queues()

    def terminate(self):
        if self.opt.sp:
            self._stopped = True
        else:
            self._stopped.value = True
        self.stop()

    def clear_queues(self):
        self.clear(self.image_queue)
        self.clear(self.det_queue)
        self.clear(self.pose_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()
    
    def enqueue_image(self, image, image_name):
        img = self.detector.image_preprocess(image)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        orig_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_dim_list = orig_img.shape[1], orig_img.shape[0]

        self.image_queue.put((img, orig_img, image_name, im_dim_list))
 
    def image_detection(self):
        batch_imgs = []
        batch_orig_imgs = []
        batch_im_names = []
        batch_im_dim_lists = []

        while True:
            try:
                img, orig_img, image_name, im_dim_list = self.wait_and_get(self.image_queue)
                if img is None:  # 종료 신호를 받으면
                    if batch_imgs:  # 남아 있는 마지막 배치 처리
                        self.process_batch(batch_imgs, batch_orig_imgs, batch_im_names, batch_im_dim_lists)
                    self.wait_and_put(self.det_queue, (None, None, None, None, None, None, None))
                    break

                # 배치에 이미지 추가
                batch_imgs.append(img)
                batch_orig_imgs.append(orig_img)
                batch_im_names.append(image_name)
                batch_im_dim_lists.append(im_dim_list)

                # 배치가 충분히 채워지면 처리
                if len(batch_imgs) >= self.opt.detbatch:
                    self.process_batch(batch_imgs, batch_orig_imgs, batch_im_names, batch_im_dim_lists)
                    batch_imgs = []
                    batch_orig_imgs = []
                    batch_im_names = []
                    batch_im_dim_lists = []

            except Exception as e:
                print("An error occurred in image_detection: ", e)
                continue

    def process_batch(self, imgs, orig_imgs, im_names, im_dim_lists):
        # 배치 데이터를 torch tensor로 변환
        imgs = torch.cat(imgs)
        im_dim_lists = torch.cat([torch.FloatTensor([dim_list]) for dim_list in im_dim_lists]).repeat(1, 2)

        # 검출 수행
        with torch.no_grad():
            dets = self.detector.images_detection(imgs, im_dim_lists)
            if isinstance(dets, int) or dets.shape[0] == 0:
                # 검출된 객체가 없는 경우, det_queue에 None을 넣어 다음 단계에서 처리하도록 함
                for orig_img, im_name in zip(orig_imgs, im_names):
                    continue
                    # self.wait_and_put(self.det_queue, (orig_img, im_name, None, None, None, None, None))
                return
            dets = dets.cpu()
            boxes = dets[:, 1:5]
            scores = dets[:, 5].unsqueeze(1)
            ids = dets[:, 6].unsqueeze(1) if self.opt.tracking else torch.zeros(scores.shape)

            # 검출 결과를 det_queue에 넣음
            for i, (orig_img, im_name) in enumerate(zip(orig_imgs, im_names)):
                # 유효한 박스가 없으면 continue
                if boxes.shape[0] == 0:
                    continue

                # 포즈 추정 입력 데이터 준비
                inps = torch.zeros(boxes.size(0), 3, *self._input_size)
                cropped_boxes = torch.zeros(boxes.size(0), 4)

                for j, box in enumerate(boxes):
                    inps[j], cropped_box = self.transformation.test_transform(orig_img, box)
                    cropped_boxes[j] = torch.FloatTensor(cropped_box)

                self.wait_and_put(self.det_queue, (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes))

    def image_postprocess(self):
        while True:
            try:
                # det_queue에서 데이터를 기다리며 받아옵니다.
                orig_img, im_name, boxes, scores, ids, inps, cropped_boxes = self.wait_and_get(self.det_queue)

                # 종료 신호를 확인합니다. 예를 들어, None을 종료 신호로 사용할 수 있습니다.
                if orig_img is None:
                    self.wait_and_put(self.pose_queue, (None, None, None, None, None, None, None))
                    break  # 종료 신호에 따라 루프 종료

                if boxes is None or boxes.nelement() == 0:
                    # 검출된 객체가 없으면 포즈 큐에 결과를 저장하고 계속 진행
                    self.wait_and_put(self.pose_queue, (None, orig_img, im_name, boxes, scores, ids, None))
                    continue

                # 검출된 각 객체에 대해 포즈 추정 입력 데이터 준비
                for i, box in enumerate(boxes):
                    inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                    cropped_boxes[i] = torch.FloatTensor(cropped_box)

                # 준비된 데이터를 포즈 큐에 저장
                self.wait_and_put(self.pose_queue, (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes))
            except Exception as e:
                print("Error in image_postprocess:", e)
                continue  # 예외 발생 시 계속 진행

    def read(self):
        return self.wait_and_get(self.pose_queue)

    @property
    def stopped(self):
        if self.opt.sp:
            return self._stopped
        else:
            return self._stopped.value

    @property
    def length(self):
        return self.datalen
