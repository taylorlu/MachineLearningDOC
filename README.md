## 图像、人脸、OCR、语音相关算法整理
##### [概述-图像语音机器学习（Outline-Image & Audio & Machine Learning）](#0)
##### 1.  [通用物体检测和识别（General Object Detection/Recognition）](#1)
##### 2.  [特定物体检测和识别和检索（Specific Object Detection/CBIR）](#2)
##### 3.  [物体跟踪（Object Tracking）](#3)
##### 4.  [物体分割（Object Segmentation）](#4)
##### 5.  [人脸检测（Face Detection）](#5)
##### 6.  [人脸关键点对齐（Face Alignment）](#6)
##### 7.  [人脸识别（Face Recognition）](#7)
##### 8.  [人像重建（Face Reconstruct）](#8)
##### 9.  [OCR字符识别（Wild Scene & Hand Written）](#9)
##### 10.  [语音识别（Automatic Speech Recognition/Speech to Text）](#10)
##### 11.  [说话人识别（Speaker Recognition/Identification/Verification）](#11)
##### 12.  [说话人语音分割（Speaker Diarization）](#12)
##### 13.  [语音合成（Text To Speech）](#13)
##### 14.  [声纹转换（Voice Conversion）](#14)
##### 15.  [人脸生物特征（Age Gender）](#15)
<span id="0"></span>
**概述-图像语音机器学习（Outline-Image & Audio & Machine Learning）**
+ 图像：
  ```
  1. 变换(Transform)，分为旋转、放缩、平移、仿射、投影
  ```
  Rotation和Scale可以看做是一个SVD分解，对于二维图像，对应2x2矩阵。
  Translate为了支持矩阵相加，需要扩充一列，所以前三者结合变成一个2x3或3x3矩阵。
  Affine加上了翻转和斜切，保持点的共线性和直线的平行性，共有6个自由度dof。
  Projection变换不是线性的，共有8个自由度。
  可参考[Transformations](https://courses.cs.washington.edu/courses/csep576/11sp/pdf/Transformations.pdf)。
  通过对变换做处理，可用于变形OCR的纠正，比如[TPS算法](https://profs.etsmtl.ca/hlombaert/thinplates)。
  ```
  2. 卷积(convolution)，分为一阶、二阶
  ```
  一阶算子有Roberts、Sobel、Prewitt，由于只求了一阶导数，所以一次只能检测一个方向的边缘。
  二阶算子有Laplace、LoG、DoG，是角点检测的第一步，不抗噪。
  卷积其实就是信号处理里面的求积再求和运算，在CNN中，卷积核是需要训练的参数，但由于大多数是共享的，参数量并不大，一般不需要Dropout。由于训练出的卷积核大多并不对称，所以并没有旋转不变性(rotation invariant)，对于放缩和平移不变性也只能由pooling层起很小的作用。最初的方法是通过Data Argument，在NIPS2015上，[spatial transformer networks](https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf)提出了一种自动学习变换矩阵的BP网络，对于数据增强的依赖大大降低。
  ```
  3. 大津阈值二值化，分水岭分割
  离散傅里叶变换DFT，离散余弦变换DCT，小波变换Wavelet
  图像的一阶二阶矩，形状描述
  颜色空间(RGB, YUV, HSV)
  以上用于视频编码和图像分析的多
  ```
  ```
  4. 图像融合
  ```
  图像融合可用在深度学习后处理，比如分割后的物体融合到另一个背景，人像换脸等。常用的有[poisson Image Editing](https://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf)
  
+ 语音：
  ```
  1. wav和mfcc
  ```
  由于语音是含有时域信息的，在进行实时频域转换的时候会采用加窗的短时STFT变换，根据不同的窗函数，会生成不同频段的频谱值。mfcc是基于梅尔频率的倒谱，是非线性的对数倒频谱。在进行ASR、SV时，一般都会先将wav文件转成mfcc进行处理，当然也不排除直接用wav的，比如wavenet, sincnet等。采用mfcc的好处是既含有时域信息也含有频域信息，由小窗函数将数据压缩成二维可采用普通CNN网络对其进行处理。
  ```
  2. 听歌识曲，哼唱识别
  ```
  曾经研究过的传统方法，基于mfcc和倒排索引。
  1. A Highly Robust Audio Fingerprinting System
  2. ROBUST AUDIO FINGERPRINT EXTRACTION ALGORITHM
  3. An Industrial-Strength Audio Search Algorithm</br>
  深度学习的检索</br>
  A Tutorial on Deep Learning for Music Information Retrieval


+ 统计学习：
  ```
  1. SVM支持向量机
  ```
  这个是老外写的一本《支持向量机导论》，网上中文英文都有。</br>
  an introduction to support vector machines and other kernel-based learning methods</br>
  包含从核函数到VC维最大泛化间隔，到KKT不等式约束的拉格朗日对偶问题，再到SMO算法求解拉格朗日乘子，算是很完整的一个教材了。

  ```
  2. Adaboost
  ```
  从弱学习机到强学习机，是一种迭代算法，只要分类器比随机分类器好一点，它就能逐渐迭代出一个强分类器。优点是不容易过拟合，缺点对噪声敏感。</br>
  1. A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting
  2. Multi-class AdaBoost

  ```
  3. Decision tree决策树
  ```
  主要用在数据挖掘，最优树的生成常用有ID3/4/5,CART等算法，缺点是不稳定，特别是样本数量不一致的情况。
  
  ```
  4. 贝叶斯网络、随机森林
  ```
  
  ```
  5. EM/GMM模型
  ```
  含有隐变量的聚类模型。隐变量/隐分布就是每个概率分布的权重以及每个样本属于每个分布的概率。</br>
  EM算法分为2步，E-Step是固定已知变量利用Jensen不等式求对数似然函数的极值，更新隐变量，M-Step是在固定隐变量求整个似然函数的极值，更新已知变量
  GMM模型是先假定分布是高斯分布，已知变量即均值和方差，隐变量即高斯分布的权重。</br>
  EM算法对初始值敏感，无法保证全局最优。用途很多，比如聚类、声纹模型UBM。</br>
  神经网络求解EM算法:</br>
  1. Neural Expectation Maximization</br>
  https://github.com/sjoerdvansteenkiste/Neural-EM
  
  ```
  6. 无监督聚类Kmeans、Meanshift,基于图模型的Spectral Clustering
  ```
  
  ```
  7. 不用指定聚类个数的模型DBSCAN、Chinese Whisper
  ```
+ 深度学习：
  深度学习即完全基于神经网络的模型，包括CNN空域、RNN时域等模型，重点在于网络设计、损失函数设计，以及优化器这3方面。</br>
  **网络设计**代表性的有CNN、空洞卷积、通道可分离卷积、DropOut、RNN/LSTM/GRU、Attention/Self-Attention/Transformer、Resnet、Inception系列、Squeezenet/Mobilenet/Shufflenet等</br>
  **损失函数**代表性的有Triplet loss、Center loss、SphereFace、ArcFace、AMSoftmax等</br>
  **优化器**主要有SGD、Moment、Adagrad、Adadelta、Adam、RMSprop、Adabound、Admm等，还有其他加快收敛防止过拟合的方法如Batchnorm，正则化等。
  
<span id="1"></span>
1. **通用物体检测和识别（General Object Detection/Recognition）**
+ 传统方法：
  ```
    1. 基于Bag Of Words词袋模型的，SIFT/SURF+KMeans+SVM
    2. 基于Sparse Coding稀疏编码的，LLC
    3. 基于聚合特征的，Fisher Vector/VLAD
    4. 基于变形部件组合模型的，DPM用到HOG/Latent SVM
    5. 有关角点的检测和描述，近几年有基于深度学习的方法，如LIFT、DELP、LFNET，缺点是速度慢
  ```
- 相关论文：
  ```
    1. Visual Object Recognition, Kristen Grauman
    2. Locality-constrained Linear Coding for Image Classification 
    3. Fisher Kernels on Visual Vocabularies for Image Categorization
    4. Improving the Fisher Kernel for Large-Scale Image Classification 
    5. Aggregating local descriptors into a compact image representation
    6. Object Detection with Discriminatively Trained Part Based Models
    7. LIFT: Learned Invariant Feature Transform
    8. Large-Scale Image Retrieval with Attentive Deep Local Feature
    9. LF-Net: Learning Local Features from Images
  ```
- 相关开源地址：
  * http://www.vlfeat.org
  * https://github.com/rbgirshick/voc-dpm
  * https://github.com/cbod/cs766-llc
  * https://github.com/nashory/DeLF-pytorch
  * https://github.com/vcg-uvic/lf-net-release
</br>

+ 深度学习：
  ```
  RCNN/SPPNet/Faster RCNN，Yolo系列，SSD，R-FCN，RetinaNet，CFENet
  ```
- 相关论文：
  ```
  1. Rich feature hierarchies for accurate object detection and semantic segmentation
  2. Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
  3. Fast R-CNN
  4. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
  5. You Only Look Once: Unified, Real-Time Object Detection
  6. YOLO9000: Better, Faster, Stronger
  7. YOLOv3: An Incremental Improvemen
  8. SSD: Single Shot MultiBox Detector
  9. R-FCN: Object Detection via Region-based Fully Convolutional Networks
  10. Focal Loss for Dense Object Detection
  11. CFENet: An Accurate and Efficient Single-Shot Object Detector for Autonomous Driving
  ```
- 相关开源地址：
  * https://github.com/rbgirshick/rcnn
  * https://github.com/rbgirshick/fast-rcnn
  * https://github.com/rbgirshick/py-faster-rcnn
  * https://github.com/balancap/SSD-Tensorflow
  * https://github.com/chuanqi305/MobileNet-SSD
  * https://github.com/gliese581gg/YOLO_tensorflow
  * https://github.com/choasup/caffe-yolo9000
  * https://github.com/qqwweee/keras-yolo3
  * https://github.com/daijifeng001/R-FCN
  * https://github.com/YuwenXiong/py-R-FCN
  * https://github.com/daijifeng001/caffe-rfcn
  * https://github.com/facebookresearch/Detectron

<span id="2"></span>
2. **特定物体检测和识别和检索（Specific Object Detection/CBIR）**
  - 特定物体只识别一张特定的图，不能进行大样本训练，也即不需要进行训练和学习。大多数只是用Artificial Feature手工特征，比如特征点，而且对于刚性物体，特征点匹配可以用SVD分解和RANSAC计算出仿射变换矩阵，进而判断物体边缘的方向。也有基于神经网络的，如R-MAC，NetVlad，但用的都是backpone预训练模型。
  - 特征点匹配，基于欧氏距离的，如SIFT/SURF，基于海明距离的，如AKAZE/FREAK，欧氏距离的检索可以用KD-Tree或者其他算法如hnsw、Falconn，海明距离的检索用LSH。
  - 基于Fisher Vector/VLAD，采用随机超平面的方式切换成海明距离进行检索
  - 检索，基于欧式距离的检索有hnsw、Falconn、Faiss等开源库。
+ 相关论文：
  ```
  1. Aggregating Deep Convolutional Features for Image Retrieval
  2. PARTICULAR OBJECT RETRIEVAL WITH INTEGRAL MAX-POOLING OF CNN ACTIVATIONS
  3. Deep Learning of Binary Hash Codes for Fast Image Retrieval
  4. Learning Compact Binary Descriptors with Unsupervised Deep Neural Networks
  5. Bags of Local Convolutional Features for Scalable Instance Search
  6. Deep Image Retrieval: Learning global representations for image search
  7. Region-Based Image Retrieval Revisited
  ```
+ 相关开源地址：
  * https://github.com/Relja/netvlad
  * https://github.com/uzh-rpg/netvlad_tf_open
  * https://github.com/nmslib/hnswlib
  * https://github.com/facebookresearch/faiss
  * https://github.com/FALCONN-LIB/FALCONN
  * https://github.com/imatge-upc/retrieval-2016-icmr

<span id="3"></span>
3. **物体跟踪（Object Tracking）**
  - 光流法
  - 卡尔曼滤波器
  - 均值漂移
  物体跟踪在OpenCV里面都有实现，大多都是针对刚性物体，对于人脸这种物体不适合。
  深度学习的方法：
  - CFNet
+ 相关论文：
  ```
  End-to-end representation learning for Correlation Filter based tracking
  ```
+ 相关开源地址：
  * https://github.com/bertinetto/cfnet

<span id="4"></span>
4. **物体分割（Object Segmentation）**
  - 目前主流的都是基于神经网络的。
  - FCN、SegNet、PSPNet、MaskRCNN 、DeepLab系列、RefineNet、DeeperLab
+ 相关论文：
  ```
  1. Fully Convolutional Networks for Semantic Segmentation
  2. SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
  3. Pyramid Scene Parsing Network
  4. Mask R-CNN
  5. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
  6. Rethinking Atrous Convolution for Semantic Image Segmentation
  7. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
  8. RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
  9. DeeperLab: Single-Shot Image Parser
  10. MobileNetV2: Inverted Residuals and Linear Bottlenecks
  ```

+ 相关开源地址：
  * https://github.com/shekkizh/FCN.tensorflow
  * https://github.com/alexgkendall/caffe-segnet
  * https://github.com/hszhao/PSPNet
  * https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow
  * https://github.com/matterport/Mask_RCNN
  * https://github.com/sthalles/deeplab_v3
  * https://github.com/DrSleep/tensorflow-deeplab-resnet
  * https://github.com/guosheng/refinenet
  * https://github.com/DrSleep/light-weight-refinenet

<span id="5"></span>
5.	**人脸检测（Face Detection）**
+ 传统方法：特征提取+分类器的方式
  ```
  特征主要有HOG、HAAR等，分类器有Adaboost、SVM、Cascade等。
  常用的开源库有：OpenCV、Dlib等。
  ```
+ 深度学习：
  ```
  MTCNN、PyramidBox、HR、Face R-CNN、SSH、RSA、S3FD、FaceBoxes
  ```
+ 相关论文：
  ```
  1. Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
  2. PyramidBox: A Context-assisted Single Shot Face Detector.
  3. Finding Tiny Faces
  4. Face R-CNN
  5. SSH: Single Stage Headless Face Detector
  6. Recurrent Scale Approximation for Object Detection in CNN
  7. S 3FD: Single Shot Scale-invariant Face Detector
  8. FaceBoxes: A CPU Real-time Face Detector with High Accuracy
  ```
+ 相关开源地址：
  * https://github.com/kpzhang93/MTCNN_face_detection_alignment
  * https://github.com/EricZgw/PyramidBox
  * https://github.com/cydonia999/Tiny_Faces_in_Tensorflow
  * https://github.com/mahyarnajibi/SSH
  * https://github.com/sciencefans/RSA-for-object-detection
  * https://github.com/louis-she/sfd.pytorch
  * https://github.com/sfzhang15/FaceBoxes

<span id="6"></span>
6. **人脸关键点对齐（Face Alignment）**
+ 一些人脸检测算法中会集成有人脸关键点对齐，在训练时2个任务的误差函数加权相加。对齐有2D和3D的区别，2D只考虑二维信息，3D需要有3维模型，能预测人脸的姿态信息。
+ 2D关键点对齐：DCNN、MTCNN、TCDCN、LAB
+ 3D关键点对齐：3DDFA、DenseReg、FAN、PRNet、PIPA
+ 相关论文：
  ```
  1. Facial Landmark Detection by Deep Multi-task Learning
  2. Deep Convolutional Network Cascade for Facial Point Detection
  3. Look at Boundary: A Boundary-Aware Face Alignment Algorithm
  4. Face Alignment Across Large Poses: A 3D Solution
  5. Pose-Invariant Face Alignment via CNN-Based Dense 3D Model Fitting
  6. Dense Face Alignment
  7. DenseReg: Fully Convolutional Dense Shape Regression In-the-Wild
  8. How far are we from solving the 2D & 3D Face Alignment problem
  9. Learning Dense Facial Correspondences in Unconstrained Images
  10. Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
  11. Dense Face Alignment
  ```
+ 相关开源地址：
  * https://github.com/zhzhanp/TCDCN-face-alignment
  * https://github.com/wywu/LAB
  * https://github.com/cleardusk/3DDFA
  * https://github.com/ralpguler/DenseReg
  * https://github.com/YadiraF/PRNet
  * http://cvlab.cse.msu.edu/project-pifa.html

<span id="7"></span>
7. **人脸识别（Face Recognition）**
+ 非神经网络：GaussianFace高斯脸
+ 深度学习：大多数和损失函数设计有关
+ DeepFace、DeepID系列、VGGFace、FaceNet、CenterLoss、MarginalLoss、SphereFace、ArcFace、AMSoftmax
+ 相关论文：
  ```
  1. Surpassing Human-Level Face Verification Performance on LFW with GaussianFace
  2. DeepFace: Closing the Gap to Human-Level Performance in Face Verification
  3. Deep Learning Face Representation from Predicting 10,000 Classes
  4. Deep Learning Face Representation by Joint Identification-Verification
  5. DeepID3: Face Recognition with Very Deep Neural Networks
  6. Deep Face Recognition
  7. FaceNet: A Unified Embedding for Face Recognition and Clustering
  8. A Discriminative Feature Learning Approach for Deep Face Recognition
  9. Marginal Loss for Deep Face Recognition
  10. SphereFace: Deep Hypersphere Embedding for Face Recognition
  11. ArcFace: Additive Angular Margin Loss for Deep Face Recognition
  12. Additive Margin Softmax for Face Verification
  ```
+ 相关开源地址:
  * https://github.com/jangerritharms/GaussianFace
  * http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
  * https://github.com/davidsandberg/facenet
  * https://github.com/wy1iu/sphereface
  * https://github.com/xialuxi/arcface-caffe
  * https://github.com/deepinsight/insightface

<span id="8"></span>
8. **人像重建（Face Reconstruct）**
+ 基本上都是基于3D的，人像重建后可以进行姿态估计，以及换脸。有的换脸算法需要多张人脸训练GAN网络。
+ PRNet、VRN、Face2Face
+ 相关论文：
  ```
  1. State of the Art on Monocular 3D Face Reconstruction, Tracking, and Applications
  2. 3D Face Reconstruction with Geometry Details from a Single Image
  3. Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
  4. CNN-based Real-time Dense Face Reconstruction with Inverse-rendered Photo-realistic Face Images
  5. Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression
  6. Deep Video Portraits
  7. VDub: Modifying Face Video of Actors for Plausible Visual Alignment to a Dubbed Audio Track
  8. paGAN: Real-time Avatars Using Dynamic Textures
  9. On Face Segmentation, Face Swapping, and Face Perception
  10. Extreme 3D Face Reconstruction: Looking Past Occlusions
  ```
+ 相关开源地址:
  * https://github.com/YadiraF/PRNet
  * https://github.com/AaronJackson/vrn
  * https://github.com/deepfakes/faceswap
  * https://github.com/datitran/face2face-demo
  * https://github.com/YuvalNirkin/face_swap
  * https://github.com/anhttran/extreme_3d_faces

<span id="9"></span>
9. **OCR字符识别（Wild Scene & Hand Written）**
+ OCR涉及到字符场景定位和分割，以及字符识别。传统的方法是采用垂直方向直方图形式对字符进行分割，然后一个个字符分别送入分类器进行识别。由于CNN/RNN/CTC动态规划算法及Attention机制的出现，当今的主流模型是CNN+RNN+CTC，采用和语音识别类似的自动语素分割的方式。检测框一般是水平的，如果要纠正还需要用Hough变换把文本方向纠正。近几年又出现了很多支持不同形状的文本区域检测方法，一种是基于分割的，如PixelLink、TextSnake，一种是基于回归的，如TextBoxes、DMPNet、RSDD，还有结合2者的，如SSTD。还有检测和识别端到端的，如FOTS、EAA、Mask TextSpotter、STN-OCR。
+ 字符区域检测：
  CTPN、EAST、TextBoxes++、AdvancedEast、TextSnake、Mask TextSpotter、DMPNet、RSDD、LOMO、PSENet、Pixel-Anchor
+ 相关论文：
  ```
  1. Detecting Text in Natural Image with Connectionist Text Proposal Network
  2. Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes
  3. Single Shot Scene Text Retrieval
  4. EAST: An Efficient and Accurate Scene Text Detector
  5. DeepTextSpotter: An End-to-End Trainable Scene Text Localization and Recognition Framework
  6. Recursive Recurrent Nets with Attention Modeling for OCR in the Wild
  7. Multi-Oriented Text Detection with Fully Convolutional Networks
  8. Accurate Text Localization in Natural Image with Cascaded Convolutional Text Network
  9. TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes
  10. An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes
  11. Rotation-Sensitive Regression for Oriented Scene Text Detection
  12. Character Region Awareness for Text Detection
  13. Look More Than Once: An Accurate Detector for Text of Arbitrary Shapes
  14. Shape Robust Text Detection with Progressive Scale Expansion Network
  15. Pixel-Anchor: A Fast Oriented Scene Text Detector with Combined Networks
  16. 总结Overview：https://github.com/whitelok/image-text-localization-recognition
  17. 挑战赛：http://rrc.cvc.uab.es
  18. An end-to-end textspotter with explicit alignment and attention
  19. STN-OCR: A single Neural Network for Text Detection and Text Recognition
  ```
+ 字符识别：
  针对wild形变场景，检测到的框有粗糙的矩形，也有精确的多边形，在识别之前一般要进行纠正。关于纠正其实大体分为2个方向，一个是基于character划分的，如TextSnake、Char-Net，还有一种是通过TPS+STN网络自动去训练多点纠正的参数，这在很多Paper里面都有介绍。</br>
  CRNN、GRCNN、CRAFT、ASTER、MORAN、ESIR、FAN，支持垂直方向文本识别的AON
+ 相关论文：
  ```
  1. Gated Recurrent Convolution Neural Network for OCR
  2. An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition
  3. What is wrong with scene text recognition model comparisons? dataset and model analysis
  4. ASTER: An Attentional Scene Text Recognizer with Flexible Rectification
  5. Char-Net: A Character-Aware Neural Network for Distorted Scene Text Recognition
  6. MORAN: A Multi-Object Rectified Attention Network for Scene Text Recognition
  7. SEE: Towards Semi-Supervised End-to-End Scene Text Recognition
  8. ESIR: End-to-end Scene Text Recognition via Iterative Image Rectification
  9. AON: Towards Arbitrarily-Oriented Text Recognition
  10. Simultaneous Recognition of Horizontal and Vertical Text in Natural Images
  11. Focusing Attention: Towards Accurate Text Recognition in Natural Images
  ```
+ 相关开源地址：
  * https://github.com/eragonruan/text-detection-ctpn
  * https://github.com/MhLiao/TextBoxes_plusplus
  * https://github.com/lluisgomez/single-shot-str
  * https://github.com/huoyijie/AdvancedEAST
  * https://github.com/MichalBusta/DeepTextSpotter
  * https://github.com/Jianfeng1991/GRCNN-for-OCR
  * https://github.com/princewang1994/TextSnake.pytorch
  * https://github.com/clovaai/deep-text-recognition-benchmark
  * https://github.com/bgshih/aster
  * https://github.com/liuheng92/tensorflow_PSENet
  * https://github.com/whai362/PSENet
  * https://github.com/Canjie-Luo/MORAN_v2
  * https://github.com/Bartzi/see
  * https://github.com/huizhang0110/AON
  * https://github.com/Bartzi/stn-ocr

+ 手写字体识别：
  hand written由于各种书法风格，难度远高于印刷字体。NIPS上发表的几篇基于2维LSTM-RNN的方法，后面又有提速版的attention机制，这种方法支持一段手写文本的自动分行及对齐。后面ECCV又出现了一篇分多步的方法。
+ 相关论文：
  ```
  1. Offline Handwriting Recognition with Multidimensional Recurrent Neural Networks
  2. Scan, Attend and Read: End-to-End Handwritten Paragraph Recognition with MDLSTM Attention
  3. Joint Line Segmentation and Transcription for End-to-End Handwritten Paragraph Recognition
  4. Start, Follow, Read: End-to-End Full-Page Handwriting Recognition
  5. Handwriting Recognition with Large Multidimensional Long Short-Term Memory Recurrent Neural Networks
  6. Handwriting Recognition of Historical Documents with few labeled data
  7. Measuring Human Perception to Improve Handwritten Document Transcription
  8. Learning Spatial-Semantic Context with Fully Convolutional Recurrent Network for Online Handwritten Chinese Text Recognition
  9. Gated Convolutional Recurrent Neural Networks for Multilingual Handwriting Recognition
  10. Joint Recognition of Handwritten Text and Named Entities with a Neural End-to-end Model
  ```
+ 相关开源地址：
  * https://github.com/cwig/start_follow_read
  * https://github.com/0x454447415244/HandwritingRecognitionSystem
  * http://www.tbluche.com/scan_attend_read.html

<span id="10"></span>
10. **语音识别（Automatic Speech Recognition/Speech to Text）**
+ 传统方式基于GMM-HMM模型和Vertibi算法
+ 深度学习：对WAV进行MFCC短时频谱信号提取，依次采用CNN卷积网络和LSTM循环网络以及CTC Loss误差函数进行建模。
    GRU-CTC、DFCNN、DFSMN、DeepSpeech、CLDNN
+ 相关论文
  ```
  1. DEEP-FSMN FOR LARGE VOCABULARY CONTINUOUS SPEECH RECOGNITION
  2. Deep Speech: Scaling up end-to-end speech recognition
  3. CONVOLUTIONAL, LONG SHORT-TERM MEMORY, FULLY CONNECTED DEEP NEURAL NETWORKS
  ```
+ 相关开源地址：
  * https://github.com/buriburisuri/speech-to-text-wavenet
  * https://github.com/Kyubyong/tacotron
  * https://github.com/PaddlePaddle/DeepSpeech

<span id="11"></span>
11. **说话人识别（Speaker Recognition/Identification/Verification）**
+ 声纹识别的主要问题在于语音时长、文本无关、开集比对、背景噪声等问题上。目前基于d-vector、x-vector的深度学习模型和TE2E/GE2E等的损失函数设计在短时长上比较占优势。传统方法的state-of-the-art是i-vector，采用pLDA信道补偿算法，所有基于深度学习的模型都会引用ivector的ERR作为baseline进行比对。以前的方法有UBM-GMM和JFA信道补偿，但是需要大量的不同信道的语料样本。传统方法的相关开源框架有Kaldi、ALIZE、SIDEKIT、pyannote-audio等。深度学习的方法有d-vector、x-vector、j-vector（文本有关）以及结合E2E损失函数的模型。还有基于GhostVlad和直接基于wave信号的SINCNET。
+ 相关开源地址：
  * http://www-lium.univ-lemans.fr/sidekit/
  * https://alize.univ-avignon.fr/
  * http://www.kaldi-asr.org/
  * https://github.com/rajathkmp/speaker-verification
  * https://github.com/wangleiai/dVectorSpeakerRecognition
  * https://github.com/Janghyun1230/Speaker_Verification
  * https://github.com/pyannote/pyannote-audio
  * https://github.com/WeidiXie/VGG-Speaker-Recognition
  * https://github.com/mravanelli/SincNet

<span id="12"></span>
12. **说话人语音分割（Speaker Diarization）**
- 语音智能分割是基于说话人识别的，说话人识别效果的好坏决定语音分割的效果，当然还有切换点的识别效果也很重要。首先需要用VAD静音检测对语音进行分割，最简单的是用振幅来判断，如果有背景音则需要设计其他的VAD算法。切换点的判断可以通过BIC贝叶斯准则，最后就是聚类，判断哪些片段属于一个说话人，对于无监督学习算法，先验信息说话人数量显得尤为重要。目前基于深度学习的框架也有不少，比如最近Google出的UIS-RNN(其实是另类的聚类方法)，还有法国LIUM团队的S4D。
+ 相关论文：
  ```
  1. FULLY SUPERVISED SPEAKER DIARIZATION
  2. SPAKER DIARIZATION WITH LSTM
  3. S4D: Speaker Diarization Toolkit in Python
  ```
+ 相关开源地址：
  * https://github.com/google/uis-rnn
  * https://github.com/wq2012/SpectralCluster
  * https://projets-lium.univ-lemans.fr/s4d

<span id="13"></span>
13. **语音合成（Text To Speech）**
- 文本转语音，传统方法是采用语素拼接，这种方式合成的语音比较生硬，没有语调。当前Baidu、Google、FaceBook等出了很多基于深度学习的方法。一般的流程是先Encoder再Decoder，最后用Griffin-Lim算法或者WaveNet自回归模型将MFCC变成wave信号。
  WaveNet系列（MFCC-->WAVE）、DeepVoice系列、Tacotron系列、VoiceLoop、ClariNet

+ 相关论文：
  ```
  1. VOICELOOP: VOICE FITTING AND SYNTHESIS VIA A PHONOLOGICAL LOOP
  2. TACOTRON: TOWARDS END-TO-END SPEECH SYNTHESIS
  3. NATURAL TTS SYNTHESIS BY CONDITIONING WAVENET ON MEL SPECTROGRAM PREDICTIONS
  4. Deep Voice: Real-time Neural Text-to-Speech
  5. Deep Voice 2: Multi-Speaker Neural Text-to-Speech
  6. DEEP VOICE 3: 2000-SPEAKER NEURAL TEXT-TO-SPEECH
  7. WAVENET: A GENERATIVE MODEL FOR RAW AUDIO
  8. Parallel WaveNet: Fast High-Fidelity Speech Synthesis
  9. ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech
  10. SAMPLE EFFICIENT ADAPTIVE TEXT-TO-SPEECH
  11. FastSpeech: Fast, Robust and Controllable Text to Speech
  ```
+ 相关开源地址：
  * https://github.com/ibab/tensorflow-wavenet
  * https://github.com/keithito/tacotron
  * https://github.com/Kyubyong/tacotron
  * https://github.com/c1niv/Voiceloop_TensorFlow
  * https://github.com/israelg99/deepvoice
  * https://github.com/andabi/parallel-wavenet-vocoder
  * https://github.com/xcmyz/FastSpeech

<span id="14"></span>
14.	**声纹转换（Voice Conversion）**
- 声纹转换其实就是TTS的多人版，根据说话人的不同将文本生成不同的wave信号。大多数都是在网络架构中加入说话人Embedding向量，如DeepVoice2/DeepVoice3，Tacotron2，有的甚至会在声码器Vocoder中加入，比如WaveNet。
+ 相关开源地址：
  * https://github.com/r9y9/deepvoice3_pytorch
  * https://github.com/Kyubyong/deepvoice3
  * https://github.com/Rayhane-mamah/Tacotron-2
  * https://github.com/GSByeon/multi-speaker-tacotron-tensorflow

<span id="15"></span>
14.	**人脸生物特征（Age Gender Estimate）**
- 经典的DEX模型，SSR-NET精简模型
+ 相关论文：
  ```
  1. DEX: Deep EXpectation of apparent age from a single image
  2. Age Progression/Regression by Conditional Adversarial Autoencode
  3. SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation
  4. Deep Regression Forests for Age Estimation
  ```
+ 相关开源地址：
  * https://github.com/truongnmt/multi-task-learning
  * https://github.com/ZZUTK/Face-Aging-CAAE
  * https://github.com/yu4u/age-gender-estimation
  * https://github.com/shamangary/SSR-Net
  * https://github.com/shenwei1231/caffe-DeepRegressionForests
  
