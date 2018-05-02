# Ali_FashionAI
## URL : http://fashionai.alibaba.com/
### A repository for competition.

# 天池大数据竞赛——服装关键点检测

## TYUT_Landmark - 127/2332

0. 最终提交版本说明
	
	我们尝试了使用VGG直接回归关键点坐标和基于Yolo思想的方法

	YolobasedModel并不是先画出检测框，再裁剪图像。而是使用yolo的思想将图像划分为若干个格子，每个格子提若干个坐标点。并经过筛选。

	经过试验，这种方法效果并不理想。可能是因为关键点相对集中在少数几个格子里，大多数格子得不到训练。

	最终提交的版本是基于faster-RCNN和coco预训练权重，检测服装位置，裁剪后进行vgg19回归。

	最终在初赛中取得了127/2332的成绩，未进入第二轮的复赛，作为一个baseline进行开源，如果有疑问请提issue。

	以下说明均为VGG方法的说明。

1. Requirement

	- Python 3.6.5
	- Tensorflow-GPU 1.7.0
	- Keras 2.1.5
	- Numpy 1.14.2
	- Pillow 5.1.0
	- Pandas 0.22.0
	- Scikit-image 0.13.1
	- Scikit-learn 0.19.1
	- Pycocotools 2.0 - 需要源码编译安装，不能直接使用pip

2. 预处理步骤
	
	* ReshapeImages
		- 由 croded_data_generator.py 脚本处理原始图像数据，将图像都处理为512x512的尺寸
		- 原始图像某一方向（宽或高）不足512的，使用像素值为0的黑色像素点填补至原始图像两侧（原始图像位于中心）
		- 同时对关键点坐标进行变换（x或y方向增加一个值）
		- 坐标变换结果写入train.csv和test.csv，并增加填补黑边宽度（高度）数据，用于预测时还原坐标
		
		![原始图像](ReadmeImg/2.jpg "原始图像")&nbsp;&nbsp;&nbsp;
		![填补后的图像](ReadmeImg/1.jpg "填补后的图像")

	* CropAndResize
		- 使用基于coco训练的MaskRCNN模型，分割填补图像，得到服装目标的左上角坐标(X<sub>1</sub>,Y<sub>1</sub>)和右下角坐标(X<sub>2</sub>,Y<sub>2</sub>)
		- 对于未能检测出服装目标的图像，坐标值设为(0,0,0,0)
		- 若(X<sub>1</sub>,Y<sub>1</sub>,X<sub>2</sub>,Y<sub>2</sub>)不为0，则在训练过程的生成器中实时地裁剪图像、并二次填补黑边，使其保持512x512的尺寸。
		- 将reshapeImage步骤的坐标变换结果继续写入train.csv和test.csv，并增加(X<sub>1</sub>,Y<sub>1</sub>,X<sub>2</sub>,Y<sub>2</sub>)信息，用于预测时还原坐标

3. 训练步骤
	* 模型设计思想：使用vgg19模型直接回归出24组坐标值.计算loss时考虑可见性。
	* 执行方式：python3 Train.py

	* Losses.py - 在Keras框架中自定义Loss函数，实际上就是评估指标NE的实现。
	* Model.py - 基于vgg19定义了深度模型，由于评估指标中不考虑可见性的预测精度，输入(512,512,3),输出(24,2)
	* Utils.py - 本项目使用fit_generator方法，使用python生成器无限生成数据，生成器及NP计算的步骤，定义于此。按照指定的batch_size，返回一组X,y数据，图像像素值被调整至[0,1]，X.shape=(batch, 512, 512, 3)，y.shape=(batch, 25, 3)，其中y[:,-1,:]为当前图像的NP数据，为匹配输出数组的形状，repeat了3份
	* Train.py - 训练代码，使用ImageNet训练的vgg19预训练权重做迁移学习，使用SGD优化器，在训练过程中定时保存checkpoint模型权重。具体参数见代码。

4. 预测步骤
	* Predict.py - 预测图像关键点坐标。模型预测的关键点坐标，实际上是针对填补+裁剪+填补的图像的，需要经过反向的坐标计算，将预测的点映射至原始图像的坐标空间中。并将[24,2]的输出补充可见性维度，用以和提交格式相匹配。
	* 执行方式：python3 Predict.py

	* 此外，Predict脚本中还包含img_type_filter。模型会对每一幅图像都预测出24个坐标点，但实际上每一类别只需要一部分坐标，这个函数就是用来将不需要的坐标数据替换为-1，并在坐标值明显不合理（例如小于0或大于512）时，使用该类别图像在该点处的坐标均值替换。但实际上这一替换步骤并未生效。

