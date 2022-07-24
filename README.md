# paddle-flask-deploy

## 1. 介绍

PaddlePaddle是国内优秀的深度学习框架，Flask是一个使用Python编写的轻量级Web应用框架。本项目以猪只计数为例，旨在利用Flask框架实现云部署任务。

## 2. 功能
本项目一共实现了两个功能，第一个是在本地实现flask的调用功能（详见第4节），第二个是将服务封装成docker形式（详见第5节），方便部署。
## 3. 目录
```
paddle-flask-deploy
	| -- deploy_python # 预测和可视化代码
		｜ -- infer.py
		｜ -- preprocess.py
		｜ -- utils.py
		｜ -- visualize.py
	| -- models # 模型文件夹
		｜ -- ppyolo
			｜ -- infer_cfg.yml
			｜ -- model.pdiparams
			｜ -- model.pdiparams.info
			｜ -- model.pdmodel
	｜ -- app.py # 启动文件
	｜ -- Dockerfile
	｜ -- requirement.txt
```
## 4. 本地安装
### 4.1 环境安装
4.1.1 新建并启动conda环境

```
conda create -n paddle-flask python=3.7
conda activate paddle-flask
```
4.1.2 安装cpu版本的paddlepaddle
```
python3 -m pip install paddlepaddle--2.3.0 -i https://mirrors.baidu.com/pypi/simple
```

4.1.3 克隆本仓库

```
git clone https://github.com/JackDance/paddle-flask-deploy.git
```
4.1.4 安装requirement.txt

将路径切换到项目下，安装所需依赖

```
cd paddle-flask-deploy
pip install requirements.txt
```
### 4.2 flask调用
4.2.1 启动服务

```
python app.py
```
启动成功后，控制台会输出如下提示：

```
[2022/07/22 14:11:56] root INFO: Now is the debugging phase!
 * Serving Flask app 'app' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
INFO 2022-07-22 14:11:56,510 app.py:101] Now is the debugging phase!
 * Running on all addresses.
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://192.168.3.8:5002/ (Press CTRL+C to quit)
 * Restarting with stat
```
点击提示里的链接即可在浏览器中看到服务的欢迎界面
![](https://github.com/JackDance/paddle-flask-deploy/blob/master/picture_bed/welcome.jpg)

4.2.2 执行调用预测

下面使用postman进行模型的调用预测。调用的流程如下图所示。
![](https://github.com/JackDance/paddle-flask-deploy/blob/master/picture_bed/postman.jpg)

在request body中，image_file为本机图片的路径。
在response body中，consumed_time表示该图预测所消耗的时间，count表示该图中包含的猪只的数量，image_file表示该图的路径。

## 5. Run with Docker

### 5.1 Clone the repo
```
git clone https://github.com/JackDance/paddle-flask-deploy
cd paddle-flask-deploy
```

### 5.2 Build Docker Image
```
docker build -t flask-paddle-deploy:v0.1 .
```
### 5.3 Run
```
docker run -d --name flask-paddle-deploy -v /home/jackdance/Desktop/Program/Flask_study/docker_map/output_imgs:/app/output_imgs -v /home/jackdance/Desktop/Program/Flask_study/docker_map/test_imgs:/app/test_imgs -p 5002:5002 --privileged=true flask-paddle-deploy:v0.1
```
提示：在Run中，添加了两个容器卷以实现宿主机到docker容器的映射。第一个容器卷是输出可视化的图片文件夹的映射，第二个容器卷是输入的图片文件夹的映射。各位可根据自己实际宿主机的路径进行更改。

使用 `docker ps`命令查看flask-paddle-deploy容器是否启动
![](https://github.com/JackDance/paddle-flask-deploy/blob/master/picture_bed/docker_ps.jpg)

### 5.4 执行调用预测
第一步：将要预测的图片放入宿主机的输入图片文件夹中，则图片会同步到docker对应的文件夹中

第二步：使用postman实现调用预测
使用postman的调用预测步骤可参考4.2.2节。
请求体中的image_file对应的图片路径更改成`/app/test_imgs/your_test_img_name`


