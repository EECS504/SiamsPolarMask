## SiamPolarMask
This is an implementation of Siam Polar Mask, which is a mixed model of Siam Car and Polar Mask, for more details, please visit: </br >
1. [Polar Mask (CVPR 2020)](https://github.com/xieenze/PolarMask) </br >
2. [Siam Car (CVPR 2020, Oral)](https://github.com/ohhhyeahhh/SiamCAR) </br >
3. [Siam Mask (CVPR 2019)](https://github.com/foolwood/SiamMask) </br >

## Proposed Model
We propose a neural network consisting of two parts: one Siamese subnetwork for feature extraction and one classification-mask subnetwork for polar mask prediction. We will first use resnet-50 as backbone, and the network architecture is shown below.
### Network Architecture
The "star" in the figure below denotes the depth-wise cross correlation.
![](images/network_architecture.png)

### Polar Representation
Instead of using pixel level mask (like mask-rcnn), we represent the mask by one center and 36 rays with the same angle interval (10 degrees) in Polar coordinate, as shown below. Since the angle interval is pre-defined, only the length of the ray needs to be predicted. Therefore, we formulate the instance segmentation as instance center classification and dense distance regression in a Polar coordinate.

![](images/polar_rep.png)

## Training
