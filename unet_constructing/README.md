# U-Net Constructing

Scripts here are used to construct U-Net and train it with datasets.

## Types
### implementation
Scripts with `keras_impl` in name are implemented with Keras.

Relatively, those with `tf_impl` in name are implemented with TensorFlow.

### architecture
The architecture of U-Net constructed in scripts with `original` in name is the same with the one proposed in the paper<sup>[[1]](#r1)</sup>.

With `smaller` in name has half of filters in each layer, compared with the `original`.

With `with_BN` in name is the same with the architecture proposed in the paper<sup>[[2]](#r2)</sup>, with Batch Normalization in the contracting path of U-Net.

With `with_BN_2` in name is modified based on `with_BN` by myself.

## References
1.<span id="r1"> Olaf Ronneberger, Philipp Fischer, Thomas Brox: *U-Net: Convolutional Networks for Biomedical Image Segmentation.* In 2015.</span>

2.<span id="r2"> Karttikeya Mangalam, Mathieu Salzamann: *On Compressing U-net Using Knowledge Distillation.* In 2018.</span>
