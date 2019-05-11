# MetricGAN: Generative Adversarial Networks based Black-box Metric Scores Optimization for Speech Enhancement (ICML 2019)


### Introduction
MetricGAN is a Generative Adversarial Networks (GAN) based black-box metric scores optimization method.
By associating the discriminator (D) with the metrics of interest, MetricGAN can be treated as an iterative
process between surrogate loss learning and generator learning as shown in the following figure.

This code (developed with Keras) applies MetricGAN to optimize PESQ or STOI score for Speech Enhancement.
It can be easily extended to optimize other metrics.

For more details and evaluation results, please check out our  [paper](http://ieeexplore.ieee.org/document/7797132/).

![teaser](https://github.com/JasonSWFu/MetricGAN/blob/master/images/MetricGAN_learning.png)

### Dependencies:
* Python 2.7
* keras=2.0.9
* librosa=0.5.1

### Citation

If you find the code and datasets useful in your research, please cite:
    
    @inproceedings{fu2019metricGAN,
      title     = {MetricGAN: Generative Adversarial Networks based Black-box Metric Scores Optimization for Speech Enhancement},
      author    = {Fu, Szu-Wei and Liao, Chien-Feng and Tsao, Yu and Lin, Shou-De},
      booktitle = {International Conference on Machine Learning (ICML)},
      year      = {2019}
    }
    
### Contact

e-mail: jasonfu@iis.sinica.edu.tw or d04922007@ntu.edu.tw
