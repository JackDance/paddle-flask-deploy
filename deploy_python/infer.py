import os
import yaml
import glob

import json
from pathlib import Path
from functools import reduce

import cv2
import numpy as np
import math
import datetime
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

import sys
# add deploy path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'])))
sys.path.insert(0, parent_path)


from preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride, LetterBoxResize, WarpAffine, Pad, decode_image

from visualize import visualize_box_mask
from utils import Timer, get_current_memory_mb




class Detector(object):
    """
    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        enable_mkldnn_bfloat16 (bool): whether to turn on mkldnn bfloat16
        output_dir (str): The path of output
        threshold (float): The threshold of score for visualization
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT.
                                    Used by action model.
    """

    def __init__(self,
                 model_dir,
                 batch_size=1,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 enable_mkldnn_bfloat16=False,
                 output_dir='output',
                 threshold=0.5,
                 delete_shuffle_pass=False):
        self.pred_config = self.set_config(model_dir)
        self.predictor, self.config = load_predictor(
            model_dir,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            enable_mkldnn_bfloat16=enable_mkldnn_bfloat16,
            delete_shuffle_pass=delete_shuffle_pass)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.threshold = threshold

    def set_config(self, model_dir):
        return PredictConfig(model_dir)

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        for im_path in image_list:
            im, im_info = preprocess(im_path, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        input_names = self.predictor.get_input_names() # 获取模型所有输入Tensor的名称
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i]) # 获取输入Tensor的指针
            input_tensor.copy_from_cpu(inputs[input_names[i]]) # 将data中的数据拷贝到tensor中

        return inputs

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        np_boxes_num = result['boxes_num']
        if np_boxes_num[0] <= 0:
            print('[WARNNING] No object detected.')
            result = {'boxes': np.zeros([0, 6]), 'boxes_num': [0]}
        result = {k: v for k, v in result.items() if v is not None}
        return result

    def filter_box(self, result, threshold):
        np_boxes_num = result['boxes_num']
        boxes = result['boxes']
        start_idx = 0
        filter_boxes = []
        filter_num = []
        for i in range(len(np_boxes_num)):
            boxes_num = np_boxes_num[i]
            boxes_i = boxes[start_idx:start_idx + boxes_num, :]
            idx = boxes_i[:, 1] > threshold
            filter_boxes_i = boxes_i[idx, :]
            filter_boxes.append(filter_boxes_i)
            filter_num.append(filter_boxes_i.shape[0])
            start_idx += boxes_num
        boxes = np.concatenate(filter_boxes)
        filter_num = np.array(filter_num)
        filter_res = {'boxes': boxes, 'boxes_num': filter_num}
        return filter_res

    def predict(self, repeats=1):
        '''
        Args:
            repeats (int): repeats number for prediction
        Returns:
            result (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's result include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        # model prediction
        np_boxes, np_masks = None, None
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            boxes_num = self.predictor.get_output_handle(output_names[1])
            np_boxes_num = boxes_num.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()
        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        return result

    def merge_batch_result(self, batch_result):
        if len(batch_result) == 1:
            return batch_result[0]
        res_key = batch_result[0].keys()
        results = {k: [] for k in res_key}
        for res in batch_result:
            for k, v in res.items():
                results[k].append(v)
        for k, v in results.items():
            if k != 'masks':
                results[k] = np.concatenate(v)
        return results

    def get_timer(self):
        return self.det_times

    def predict_image(self,
                      image_list,
                      visual=True):
        results = []

        # preprocess
        self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image_list)
        self.det_times.preprocess_time_s.end()

        # model prediction
        self.det_times.inference_time_s.start()
        result = self.predict()
        self.det_times.inference_time_s.end()

        # postprocess
        self.det_times.postprocess_time_s.start()
        result = self.postprocess(inputs, result)
        self.det_times.postprocess_time_s.end()
        self.det_times.img_num += len(image_list)

        if visual:
            visualize(
                image_list,
                result,
                self.pred_config.labels,
                output_dir=self.output_dir,
                threshold=self.threshold)

        results.append(result)

        # 加入计数函数,jack
        image_path_and_object_num = self.bbox_counting(image_list, results)

        return image_path_and_object_num

    @staticmethod
    def format_coco_results(image_list, results, save_file=None):
        coco_results = []
        image_id = 0

        for result in results:
            start_idx = 0
            for box_num in result['boxes_num']:
                idx_slice = slice(start_idx, start_idx + box_num)
                start_idx += box_num

                image_file = image_list[image_id]
                image_id += 1

                if 'boxes' in result:
                    boxes = result['boxes'][idx_slice, :]
                    per_result = [
                        {
                            'image_file': image_file,
                            'bbox':
                            [box[2], box[3], box[4] - box[2],
                             box[5] - box[3]],  # xyxy -> xywh
                            'score': box[1],
                            'category_id': int(box[0]),
                        } for k, box in enumerate(boxes.tolist())
                    ]

                elif 'segm' in result:
                    import pycocotools.mask as mask_util

                    scores = result['score'][idx_slice].tolist()
                    category_ids = result['label'][idx_slice].tolist()
                    segms = result['segm'][idx_slice, :]
                    rles = [
                        mask_util.encode(
                            np.array(
                                mask[:, :, np.newaxis],
                                dtype=np.uint8,
                                order='F'))[0] for mask in segms
                    ]
                    for rle in rles:
                        rle['counts'] = rle['counts'].decode('utf-8')

                    per_result = [{
                        'image_file': image_file,
                        'segmentation': rle,
                        'score': scores[k],
                        'category_id': category_ids[k],
                    } for k, rle in enumerate(rles)]

                else:
                    raise RuntimeError('')

                # per_result = [item for item in per_result if item['score'] > threshold]
                coco_results.extend(per_result)

        if save_file:
            with open(os.path.join(save_file), 'w') as f:
                json.dump(coco_results, f)

        return coco_results

    # 根据format_coco_results函数，实现计数功能
    def bbox_counting(self, image_list, results):
        per_result = {
            'image_file': image_list,
            'count': 0,
        }
        count = 0

        for result in results:
            if 'boxes' in result: # 判断是否有检测到目标
                for box in result['boxes'].tolist(): #
                    if box[1] > self.threshold: # 判断检测到的bbox的置信度是否大于设定阈值
                        count += 1
                per_result = {
                    'image_file':image_list,
                    'count':count,
                }
            else:
                raise RuntimeError('')

        # print(per_result)
        return per_result

def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of images (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0], )).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info[0]['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info[0]['scale_factor'], )).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs


class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']

        # self.print_config()

    # 打印config信息
    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def load_predictor(model_dir,
                   cpu_threads=1,
                   enable_mkldnn=False,
                   enable_mkldnn_bfloat16=False,
                   delete_shuffle_pass=False):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT.
                                    Used by action model.
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    """
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))

    config.disable_gpu()
    config.set_cpu_math_library_num_threads(cpu_threads)
    if enable_mkldnn:
        try:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if enable_mkldnn_bfloat16:
                config.enable_mkldnn_bfloat16()
        except Exception as e:
            print(
                "The current environment does not support `mkldnn`, so disable mkldnn."
            )
            pass

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    if delete_shuffle_pass:
        config.delete_pass("shuffle_channel_detect_pass")
    predictor = create_predictor(config)
    return predictor, config


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--image_file or --image_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    print("Found {} inference images in total.".format(len(images)))

    return images


def visualize(image_list, result, labels, output_dir='output/', threshold=0.5):
    # visualize the predict result
    start_idx = 0
    for idx, image_file in enumerate(image_list):
        im_bboxes_num = result['boxes_num'][idx]
        im_results = {}
        if 'boxes' in result:
            im_results['boxes'] = result['boxes'][start_idx:start_idx +
                                                  im_bboxes_num, :]
        if 'masks' in result:
            im_results['masks'] = result['masks'][start_idx:start_idx +
                                                  im_bboxes_num, :]
        if 'segm' in result:
            im_results['segm'] = result['segm'][start_idx:start_idx +
                                                im_bboxes_num, :]
        if 'label' in result:
            im_results['label'] = result['label'][start_idx:start_idx +
                                                  im_bboxes_num]
        if 'score' in result:
            im_results['score'] = result['score'][start_idx:start_idx +
                                                  im_bboxes_num]

        start_idx += im_bboxes_num

        im = visualize_box_mask(
            image_file, im_results, labels, threshold=threshold)
        if isinstance(image_file, np.ndarray):
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            img_name = current_time + '.jpg'
        else:
            img_name = os.path.split(image_file)[-1]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = os.path.join(output_dir, img_name)
        im.save(out_path, quality=95)
        # print("save result to: " + out_path)


def main():
    paddle.enable_static()
    # 加载导出模型的配置文件infer_cfg.yml
    model_dir = r'./models/ppyolo'
    deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
    with open(deploy_file) as f:
        yml_conf = yaml.safe_load(f)

    # 配置并加载模型 (相当于实例化了该脚本中的Detector类)
    detector = eval('Detector')(
        model_dir=model_dir,
        batch_size=yml_conf['batch_size'],
        cpu_threads=yml_conf['cpu_threads'],
        enable_mkldnn=yml_conf['enable_mkldnn'],
        enable_mkldnn_bfloat16=yml_conf['enable_mkldnn_bfloat16'],
        threshold=yml_conf['threshold'],
        output_dir=yml_conf['output_dir'])

    # 数据（图像）获取，预处理，模型推理，结果后处理。(这几个步骤都在detector.predict_image函数中进行了集成)
    # predict from image
    img_list = [yml_conf['image_file']]

    results = detector.predict_image(img_list)
    return results # {'image_file': ['/home/jackdance/Desktop/Program/Flask_study/paddle-flask-deploy-web/dataset/20190515143432.jpg'], 'count': 27}


if __name__ == '__main__':
    # 主函数
    main()
