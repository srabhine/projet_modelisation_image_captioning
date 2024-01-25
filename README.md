## This repository is an implementation of the paper "Towards Unsupervised Remote Sensing Image Captioning and Retrieval with Pre-Trained Language Models" 

## Introduction 

Recent improvements in remote sensing technology have led to an unprecedented growth in the applications of Remote Sensing Images (RSI) for Geo-spatial Information
Systems. One such application is known as Land Usage / Land Cover Mapping (LULC), which aims to classify and understand the terrain and usage of the piece of land captured in the RSI. LULC can aid in land cover monitoring, resource planning, and land management.

## Open Earth Map

OpenEarthMap (OEM) is one such benchmark LULC dataset that offers diverse locations and high quality land cover annotations.  Cross-modal text-image retrieval and captioning systems present an interesting opportunity to combine Geo-spatial information with Natural Language Processing (NLP) techniques.

Hoxha et al.[5] successfully used deep learning networks to measure similarities between captions generated from the RSI in the database with the queried text. However, 
these frameworks depend on a large amount of labelled training data to successfully annotate images and measure similarity with query text.

## Motivation 

We are motivated to study unsupervised approaches for RSI captioning to reduce the overhead of manually generating such annotations to train for retrieval systems.
Template-based methods are popular to extract visual descriptions into a rule-based output. The annotations by this approach are arguably limited for describing higher level semantic information [5]. However, language models have since made considerable leaps, one such being the release of ChatGPT, a pre-trained language model trained to perform like a chat bot.

We aim to explore if it is possible for such pre-trained language models to recognize higher-level information such as landscape (e.g., rural, urban, and forest) with controlled generation based on statistical LULC information.

## Data 

I used 2 sources of that to make this work:

### Open Earth Map

OpenEarthMap is a benchmark dataset for global high-resolution land cover mapping. OpenEarthMap consists of 5000 aerial and satellite images with manually annotated 8-class land cover labels and 2.2 million segments at a 0.25-0.5m ground sampling distance, covering 97 regions from 44 countries across 6 continents. Land cover mapping models trained on OpenEarthMap generalize worldwide and can be used as off-the-shelf models in a variety of applications.

### RSCID 

The Remote Sensing Image Captioning Dataset (RSICD) is a dataset for remote sensing image captioning task. It contains more than ten thousands remote sensing images which are collected from Google Earth, Baidu Map, MapABC and Tianditu. The images are fixed to 224X224 pixels with various resolutions. The total number of remote sensing images is 10921, with five sentences descriptions per image.



## Method 
<div align="center">
  <img width="493" alt="image" src="https://github.com/srabhine/projet_modelisation_image_captioning/assets/45555197/2fcf4bf9-9779-4224-a61e-7ef4e26e8f97">
</div>

<div align="center">
  Figure 1: The 3-step framework for unsupervised natural language caption generation.
</div>


### Land Cover Mapping  

In our end-to-end model for unlabelled remote sensing image captioning, we first generate land cover maps using a a UNet model, known for its effectiveness in semantic segmentation and state of the art performance. The U-NEt is trained on the Open Earth Map Dataset, using pairs of Remote Sensing Images (RSI) and Land Use/Land Cover (LULC) images. The output is a detailed 2D array representing the predicted land cover class for each pixel, providing a foundational layer for subsequent captioning processes.

### Statistical Image Understanding 

Feature extraction plays a key role to project one mode of information to another. 
We translate the image composition and land cover structure into a template statistical prompt for the next step. The land cover classes are mapped to their text labels pre-defined by the training dataset in the prompt.

#### Image Composition

Where the composition = # predicted pixels as C / Total # of pixels

#### Land Cover Structure

To support image composition, we also describe the locations of prominent land classes. Using the median location of top 2 classes in the LULC, we locate the centroid of
each class. Unlike calculating mean, median is more robust to outliers. If the centroid falls into a circle with a radius of 10% of image size from the center, it is classified with Center location. This threshold accounts as a reasonable margin to describe the location of the LULC class. Any centroid outside of this area is classified as either Upper left, Upper Right, Lower Left, or Lower Right by dividing the image into four quadrants with the center of the image as the center of the quadrant.

### Natural Language Generation 

In the natural language generation stage, we use the inference of large pre-trained models to convert low level statistical text to a natural sounding caption. In this paper, we use ChatGPT. 

To test the model against some ground truth, we created four captions as positive examples for ChatGPT. The motivation for controlled generation stems from supervised few-shot training, where we give ChatGPT a few positive examples for generated captions and then ask it to generate given novel statistical prompts, in an attempt to raise behavior expected from the model.



To test the framework, we train a UNet model on the OEM Mini, a smaller version of the OEM dataset with 1068 examples with sub-meter resolution classified into 8
land cover labels. We used a 70-30 train-test split of the dataset for training, and trained with a batch size of 8. 
The RGB images were divided into three input channels, and each image was transformed into a 512 * 512 size image. The learning rate was set at 10−3 with a weight decay 10−6. The model is trained with the Dice loss function and Adam optimisation. We objectively evaluated the land cover model using pixel-wise mean Intersection over Union (mIoU), a standard metric to measure the coverage between the true and predicted classes.

The extracted statistical information was fed with controlled prompts into the ChatGPT, and we found that ChatGPT was able to successfully convert the prompts to natural sounding captions.

## Metrics 

In evaluating the performance of machine learning models, particularly in the realms of computer vision and natural language processing, two metrics is used in this paper: Intersection over Union (IoU) and BLEU score. 

IoU is a common metric used in object detection tasks to quantify the accuracy of a predicted bounding box against the ground truth. It is calculated as the area of overlap between the predicted bounding box and the ground truth bounding box divided by the area of union of these two boxes. This metric effectively measures how well the model's prediction aligns with the actual object's location in an image. 

<div align="center">
  <img width="493" alt="image" src="https://github.com/srabhine/projet_modelisation_image_captioning/assets/45555197/edce1285-5eba-41b4-8408-964aff334a8b">
</div>



On the other hand, the BLEU score, or the Bilingual Evaluation Understudy score, is widely used in evaluating machine translation models. It measures the correspondence between a machine's output and that of a human translator, focusing on the precision of word choice and order in sentences. The BLEU score does this by comparing the n-gram overlap between the machine-generated text and a set of reference translations, thus giving an indication of the translation's quality and fluency.

<div align="center">
  <img width="493" alt="image" src="https://github.com/srabhine/projet_modelisation_image_captioning/assets/45555197/934676f7-23f4-438a-9963-ec77f6b8ec9ab">
</div>

## Results

After training the U-Net model for semanctic segmentation: 

| Epochs     | IoU        | 
|------------|:----------:|
| 132        | 0.25       |

Results of the comparaisons between the text generated and the ground truth:

|  n-gram    | Score      | 
|------------|:----------:|
| Score 1    | 0.35       |
| Score 2    | 0.08       |
| Score 3    | 0          |
| Score 4    | 0          |
| Score      | 0          |






## Conclusion 

ChatGPT showed some promising results from transforming statistical descriptions of remote sensing images into sentences. However for future works, we should try different U-Net to see if the results of the IoU as we saw the semantic segmentation model is the maj structure here, if we have a bad training then we can't generalize and have good statistical image understanding and hence good text generation. In the future directions we can add several adjustements such has improving the u-net model, or considering an another architecture. Also we can try different type of metric to compare the generated text against the reference (ROUGE for example) .

## References 

[5]:  Genc Hoxha, Farid Melgani, and Beg ¨um Demir. Toward remote sensing image retrieval under a deep image captioning perspective. IEEE Journal of Selected Topics
in Applied Earth Observations and Remote Sensing, Vol. 13, pp. 4462–4475, 2020.


