# Contents
1. [Ⅰ. Model Compression Frameworks](#rⅠ)
	+ [PocketFlow](#rⅠ1)
	+ [TF-model-pruning](#rⅠ2)
2. [Ⅱ. Automatic ML Frameworks](#rⅡ)


# <span id='rⅠ'>Ⅰ. Model Compression Frameworks: <br> 👉 PocketFlow & TF-model-pruning</span> 

## <span id='rⅠ1'>1. PocketFlow</span>

* [Original paper about this framework](https://pdfs.semanticscholar.org/dbe1/7becf67bbc97650b6db70184bdbbc9ca23ae.pdf)
* [My reading notes about the original paper(More specific details about PocketFlow are inside)](https://github.com/WenjayDu/ML_Study_Notes/blob/master/2019-03/20190317-PocketFlow.md)

The overall design of PocketFlow:
<img style="display:block; margin-left:auto; margin-right:auto; width: 700px;" src="https://cdn.safeandsound.cn/ML_Study_Notes/image/20190317205951.png?imageslim"/>

### ⅰ. Compressing methods
Model compression algos supported by PocketFlow are listed below:

Name | Description 
-----|-------------
ChannelPrunedLearner | channel pruning with LASSO-based channel selection 
DisChnPrunedLearner | discrimination-aware channel pruning 
WeightSparseLearner | weight sparsification with dynamic pruning ratio schedule
UniformQuantLearner | weight quantization with uniform reconstruction levels
NonUniformQuantLearner | weight quantization with non-uniform reconstruction levels

### ⅱ. References
* [**Bengio et al., 2015**] Yoshua Bengio, Nicholas Leonard, and Aaron Courville. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*. CoRR, abs/1308.3432, 2013.
* [**Bergstra et al., 2013**] J. Bergstra, D. Yamins, and D. D. Cox. *Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures*. In International Conference on Machine Learning (ICML), pages 115-123, Jun 2013.
* [**Han et al., 2016**] Song Han, Huizi Mao, and William J. Dally. *Deep Compression: Compressing Deep Neural Network with Pruning, Trained Quantization and Huffman Coding*. In International Conference on Learning Representations (ICLR), 2016.
* [**He et al., 2017**] Yihui He, Xiangyu Zhang, and Jian Sun. *Channel Pruning for Accelerating Very Deep Neural Networks*. In IEEE International Conference on Computer Vision (ICCV), pages 1389-1397, 2017.
* [**He et al., 2018**] Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, and Song Han. *AMC: AutoML for Model Compression and Acceleration on Mobile Devices*. In European Conference on Computer Vision (ECCV), pages 784-800, 2018.
* [**Hinton et al., 2015**] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. *Distilling the Knowledge in a Neural Network*. CoRR, abs/1503.02531, 2015.
* [**Jacob et al., 2018**] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2704-2713, 2018.
* [**Lillicrap et al., 2016**] Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. *Continuous Control with Deep Reinforcement Learning*. In International Conference on Learning Representations (ICLR), 2016.
* [**Mockus, 1975**] J. Mockus. *On Bayesian Methods for Seeking the Extremum*. In Optimization Techniques IFIP Technical Conference, pages 400-404, 1975.
* [**Zhu & Gupta, 2017**] Michael Zhu and Suyog Gupta. *To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression*. CoRR, abs/1710.01878, 2017.
* [**Zhuang et al., 2018**] Zhuangwei Zhuang, Mingkui Tan, Bohan Zhuang, Jing Liu, Jiezhang Cao, Qingyao Wu, Junzhou Huang, and Jinhui Zhu. *Discrimination-aware Channel Pruning for Deep Neural Networks*. In Annual Conference on Neural Information Processing Systems (NIPS), 2018.

## <span id='rⅠ2'>2. TF-model-pruning</span>
>Tensorflow model pruning is an API of tensorflow framework that facilitates magnitude-based pruning of neural network's weight tensors.
It helps inject necessary tensorflow op into the training graph so the model can be pruned while it is being trained.

### ⅰ. References
* [**Zhu & Gupta, 2017**] Michael Zhu and Suyog Gupta. *To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression*. CoRR, abs/1710.01878, 2017.

# <span id='rⅡ'>Ⅱ. Automatic ML Frameworks</span> 

In the paper[<sup>[1]</sup>](#r1), authors test 4 automatic ML frameworks -- auto-sklearn, TPOT, auto_ml and H2O, and compare their performance.

## TO BE COMTINUE...

### References
1. <span id='r1'>Adithya Balaji, Alexander Allen: Benchmarking Automatic Machine Learning Frameworks. In 2018.</span>