# Empathizer-1430-final

There're three tasks specified in the `run.py`:
- Task 1: training a CNN from scratch 
- Task 2: training backbone model MobileNetV3Small with a classification head
- Task 3: training backbone model VGG16 with a classification head
- Task 4: training backbone model ResNet50 with a classification head


Download the dataset first before running the model:
https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

You can run different model by:
```{bash}
python run.py --task 1
```
Can also find other possible arguments in `run.py`.

You can run the real-time facial expression detection demo by running `expression_detection.py`.

* Before running `expression_detection.py`, you should unzip our self-designed model wight 'your.weights.e049-acc0.6002.h5' in file checkpoints/your_model/050823-053020. (the original file size exceeding the github uploading requirements).

You can also find our pretrained model weights for VGG, MobileNet, and ResNet as well as the misclassifed examples at:
https://drive.google.com/drive/folders/1EuBxT-Gcvn16KJ9Hq_Dsk1Hf_u2mL-Fk?usp=share_link

Note:

(1) `hyperparameter.py` includes non model-specific hyperparameters:  `img_size`, `num_classes`, `batch_size`, `preprocess_sample_size`, `num_ep_decrease`, `num_epochs`, `max_num_weights`.
Model-specific parameters are included in `models.py` in the constructor of each model.

(2) Added a scheduler in `run.py` to add more flexibility in changing learning rate along the training process, by increasing or decreasing learning rate after a certain number of epochs.

(3) There are different versions of MobileNet, can try different versions when training the model: MobileNetV2, MobileNetV3Small, MobileNetV3Large, etc.



