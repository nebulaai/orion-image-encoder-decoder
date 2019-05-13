# Model Description #

This trial project is a simplified image encoder and decoder model.
The input data is the CF10 image data-set.

# Encode - latent - decode #
Firstly, the model encode the image into latent representation. 
The latent representation is a "concentrate" representation of the origianl image. It only keep
the important features in the image and exclude the irrelevant features. 

In the image of plot_pattern.png, you can find that different category of image has distinct latent representation.

The decoder take the latent representation and project the information back the original image size. 
This is a reconstruction process. 

# Denoise of the encoder and decoder #

Since the encoder and decoder only keep the important feature, it naturally has the capebility to 
filter away the noise. 

In the figure2.png, you can find the denoise effect of the encoder and decoder.    
![denoise](https://github.com/nebulaai/orion-image-encoder-decoder/blob/master/figure2.png)
