_At present, the method of video super-resolution based on deep learning takes a large number of LR videos and HR videos as inputs for neural networks, performs frame alignment, feature extraction/fusion training, and then outputs a sequence of HR video frames through the network. The video super-resolution process mainly includes an alignment module, a feature extraction and fusion module, and a reconstruction module._  
_* Datasets are not provided in this repo_

# Main Modules
**Alignment module:** Extract motion information to align adjacent frames with the target frame through distortion or offset.  
**Feature extraction and fusion module:** Construct a convolutional neural network to learn and extract the main features of LR and fuse them.  
**Reconstruction module:** The feature network is resized and overlaid with the upsampling results of the original image to obtain the final HR image.  
![image](https://github.com/user-attachments/assets/666a183f-a09d-4456-b201-05986b173ba4)

## **Core structure and process of system algorithm：**
1. Video Segmentation: The video is segmented into a sequence of frames, and the low-resolution (LR) frame sequence is stored in .png image format.
2. Setting Window Size Based on Model: Set a window size according to the training model. Read in the LR frame sequence in RGB channels and normalize the pixel values of each frame, converting the pixel range from [0, 255] to [0, 1] to adapt to the network. After channel transformation for all frames, concatenate them together and convert them into a Tensor.
3. Input Tensor to Neural Network: The Tensor is used as input for the neural network. Load the model and process the input data. The output of the final layer is the super-resolved image of the frame sequence, and the super-resolved result is saved as an image.
4. Move the Window: Train the next high-resolution (HR) image.
5. Combine Frames into a Video: After all frame sequences are trained, merge all images into a video.

# Alignment
![image](https://github.com/user-attachments/assets/7fa3ee16-62ff-45e8-a41e-8ea98a0bd4fa)  
The alignment module aligns the features of adjacent frames with the features of the current frame, calculates the offset using convolution, and performs feature alignment using deformable convolution.  
Using a three-level pyramid (FPN) structure, align the features layer by layer from bottom to top, and upsample the lower offset and features to the L-1 layer for convolution, The offset and feature map of the lower layer will make the calculation of the upper layer more accurate. Finally, the current frame and aligned features will be subjected to deformable convolution to obtain the final feature.

# Feature Extraction
![image](https://github.com/user-attachments/assets/cc6da61d-7496-4a89-826d-56fa410ec14b)  
To extract features, the image is first convolved using five residual blocks. Each residual block is formed by concatenating convolutional layers - ReLU - convolutional layers, and finally adding the result to the input.  
In order to perform alignment operations, two convolutional layers are used for downsampling to extract features from Level 1, Level 2, and Level 3, respectively.  
_* The reconstruction module consists of 10 residual blocks that are consistent with the feature extraction module. After upsampling convolution and LeakyReLU activation function, the final super-resolution image is obtained._

# Feature Fusion
![image](https://github.com/user-attachments/assets/79ab535e-881b-45cf-aaba-e537fe0a2687)  
For aligned feature maps, the reference frame and adjacent frames are subjected to different convolutional layers to extract features, and the similarity between adjacent frames and the reference frame is calculated to obtain a temporal weight map. Multiplying the aligned feature maps in the spatial dimension is equivalent to adjusting the weights of feature maps at different times.  
Next, all feature maps are convolved for feature fusion, and then feature maps are obtained at different scales through a pyramid structure. After upsampling, the final weighted fused feature map is obtained.
