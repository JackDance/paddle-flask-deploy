# -*- coding: utf-8 -*-
"""
----------------------------------
    File Name: app
    Description:
    Author:    Jack
    Date:      2022/7/19
----------------------------------
"""
import os
import sys
import time

# Flask
from flask import Flask, request, render_template, Response, jsonify, redirect
from gevent.pywsgi import WSGIServer

# paddlepaddle
import paddle
from paddle.inference import Config, create_predictor
from deploy_python.infer import Detector

# some utilities
import yaml
from deploy_python.utils import get_logger

# add project path to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'])))
sys.path.insert(0, parent_path)

def load_global_config():
    """
    加载配置文件infer_cfg.yml
    """
    model_dir = r'./models/ppyolo'
    deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
    with open(deploy_file) as f:
        yml_conf = yaml.safe_load(f)
    return yml_conf

def load_model():
    """
    加载模型
    """
    paddle.enable_static()
    # 加载导出模型的配置文件infer_cfg.yml
    yml_conf = load_global_config()

    # 配置并加载模型 (相当于实例化了该脚本中的Detector类)
    detector = eval('Detector')(
        model_dir='./models/ppyolo',
        batch_size=yml_conf['batch_size'],
        cpu_threads=yml_conf['cpu_threads'],
        enable_mkldnn=yml_conf['enable_mkldnn'],
        enable_mkldnn_bfloat16=yml_conf['enable_mkldnn_bfloat16'],
        threshold=yml_conf['threshold'],
        output_dir=yml_conf['output_dir'])

    # 返回detector
    return detector

# 创建logger
log_file_path = os.path.join(parent_path, "log.log")
logger = get_logger(log_file=log_file_path)
# 加载配置文件
yml_conf = load_global_config()
# 加载模型
detector = load_model()


# Declare a flask app
app = Flask(__name__)

@app.route('/')
def homepage():
    return b"<h1>Welcome to paddle-flask-deploy</h1>", 200

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        results = {}
        # 数据（图像）获取，预处理，模型推理，结果后处理。(这几个步骤都在detector.predict_image函数中进行了集成)
        # predict from image
        img_list = [request.json['image_file']]
        t0 = time.perf_counter()
        result = detector.predict_image(img_list)
        t1 = time.perf_counter()
        results['image_file'] = result['image_file']
        results['count'] = result['count']
        results['consumed_time'] = "{}s".format(t1-t0)
        logger.info("image {0} has detected {1} pigs and consumed time is {2}s".format(result['image_file'], result['count'], (t1-t0)))
        logger.info("visualized image saved to: {}".format('./output_imgs'))
        return jsonify(results)
    else:
        logger.warning("invalid request, abort.")
        return None

if __name__ == '__main__':
    if yml_conf['debug']:
        # 开发阶段用
        logger.info("Now is the debugging phase!")
        app.run(host='0.0.0.0', port=5002, debug=True)
    else:
        # 部署阶段用
        # Serve the app with gevent
        logger.info("Now is the implement phase!")
        http_server = WSGIServer(('0.0.0.0', 5002), app)
        http_server.serve_forever()
