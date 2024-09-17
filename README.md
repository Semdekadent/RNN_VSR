_At present, the method of video super-resolution based on deep learning takes a large number of LR videos and HR videos as inputs for neural networks, performs frame alignment, feature extraction/fusion training, and then outputs a sequence of HR video frames through the network. The video super-resolution process mainly includes an alignment module, a feature extraction and fusion module, and a reconstruction module._

**Alignment module:** Extract motion information to align adjacent frames with the target frame through distortion or offset.  
**Feature extraction and fusion module:** Construct a convolutional neural network to learn and extract the main features of LR and fuse them.  
**Reconstruction module:** The feature network is resized and overlaid with the upsampling results of the original image to obtain the final HR image.  

**Core structure and process of system algorithm：**
1. Video Segmentation: The video is segmented into a sequence of frames, and the low-resolution (LR) frame sequence is stored in .png image format.
2. Setting Window Size Based on Model: Set a window size according to the training model. Read in the LR frame sequence in RGB channels and normalize the pixel values of each frame, converting the pixel range from [0, 255] to [0, 1] to adapt to the network. After channel transformation for all frames, concatenate them together and convert them into a Tensor.
3. Input Tensor to Neural Network: The Tensor is used as input for the neural network. Load the model and process the input data. The output of the final layer is the super-resolved image of the frame sequence, and the super-resolved result is saved as an image.
4. Move the Window: Train the next high-resolution (HR) image using the sliding window.
5. Combine Frames into a Video: After all frame sequences are trained, merge all images into a video.
