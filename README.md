# Shelf Classification

Enabling Classification of Heavily-occluded Objects through Class-agnostic Image Manipulation
Benjamin Gallauner, Stefan Thalhammer, and Markus Vincze.

status: BibTex to come

%[//]: # (@INPROCEEDINGS{, author={S. {Thalhammer} and T. {PaEnabling Classification of Heavily-occluded Objects through Class-agnostic Image Manipulationtten} and M. {Vincze}}, booktitle={}, title={PyraPose: Feature Pyramids for Fast and Accurate Object Pose Estimation under Domain Shift}, year={2021})

## 1. Class-agnostic object segmentation
To improve classification performance on heavily occluded shelves class agnostic segmentation is used to generate object masks.  
For this task Detectron (Mask-RCNN) was trained with the YCB video Dataset.  
Detectron can be found here: https://github.com/facebookresearch/Detectron

### Detectron training (optional)
This step is optional as weights of the trained model are provided in /object_mask_generation/model_final.pkl  

The following repository was used to convert YCB to COCO format to be compatible with detectron: https://github.com/iyezhiyu/Mask_RCNN_on_YCB_Video_Dataset

Modifications made to files in the repository:
* **video_data_annotations_generator.py**  
Changed all classes to the same one, so that network can learn a "generalized object".
* **dummy_datasets.py**  
Changed classes to only background and object.
* **infer_simple.py**  
In addition to the standard output saves a mask that indicates where objects are.
* **shelf_classification_config.yaml**  
Config file for training and inference.

The modified repository can be found in /object_mask_generation/Mask_RCNN_on_YCB_Video_Dataset.  
For detailed information on how to train Detectron please refer to the readme in the linked repository.

### Using trained Detectron (Mask-RCNN) for object mask generation
Using the modified infer_simple.py and shelf_classification_config.yaml object masks can be generated.  
If the pretrained model weights are used, the image resolution should be close to  640x480.

```
python tools/infer_simple.py \
      --cfg configs/shelf_classification_config.yaml \
      --output-dir out_dir \
      --image-ext png \ 
      --always-out \
      --output-ext png \
      --thresh 0.5 \
      --wts out_dir/train/ycb_video/generalized_rcnn/model_final.pkl \
      images_folder
```
The object masks are saved as filename.npy.npz

## 2. Segmentation augmentation 
The previously generated masks are used to apply random noise to image parts where objects were detected.  
In /classify_images/segmentation_augmentation/segmentation_augmentation.py masks and image path have to be selected.  
The generated output images have segmentation augmentation applied and can be classified, which is explained in the next section.


## 3. Shelf classification
/classify_images/classify_shelves.py is used for classification. Input image path and model weights need to be selected in the file.  
The output is classification_results.txt which contains the predicted class and percentages for bucket, hanging and standing for every input image.

## 4. Train the classification network (optional)
In /classifier_training files are provided to train the classification network with individualized parameters.  
For training 3 datasets (train, val, test) are needed.
1. Generate object masks using Detectron, which is explained in section 1.
2. Place generated masks and unaugmented images in the corresponding folders.
3. Use training_segmentation_augmentation.py for segmentation augmentation of all 3 datasets at once.
4. Configure parameters in train_shelf_classifier.py and start training.
5. Model weights with early stopping (best.h5) and without (last.h5) are saved.
