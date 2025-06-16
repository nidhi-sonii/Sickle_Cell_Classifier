### Sickle Cell Disease Classification Using Neural networks
This project aims to classify between healthy and sickle cells using different deep learning methods to compare their performance and output.
The first model includes four different variants of the Convolutional Neural Network and the other model implements a Vision Transformer.

### Dataset

The dataset is a sickle cell images dataset collected from Teso region in Uganda. This is in the Eastern part of Uganda and from Kumi and Soroti districts specifically. We picked the samples from Kumi Hospital, Soroti Regional Referral Hospital and Soroti University. 140 patients provided their blood samples which were processed using two methods: Field stains and Leichman stains. Their microscopic images were captured and are hereby presented.

The dataset has 422 positive (sickle cell) images and 147 negative images.
Due to the small size of the dataset as well as its imbalanced nature, different techniques were implemented throughout some of the variants to deal with this, for example, data augmentation and stratified cross validation.

### Variant 1

The first variant of the CNN is a model trained from scratch.
Model Architecture
Input Layer: Accepts images of shape (256, 256, 3).
Convolutional and Pooling Layers:
Conv2D with 16 filters, kernel size (3, 3), stride 1, and ReLU activation, followed by a MaxPooling2D layer.
Conv2D with 32 filters, kernel size (3, 3), stride 1, and ReLU activation, followed by a MaxPooling2D layer.
Conv2D with 16 filters, kernel size (3, 3), stride 1, and ReLU activation, followed by a MaxPooling2D layer.
Flattening Layer: Converts 2D feature maps into a 1D feature vector.
Dense (Fully Connected) Layers:
First dense layer with 256 units and ReLU activation.
Output dense layer with 1 unit and sigmoid activation for binary classification.

Features
ReLU Activation: Used in the convolutional and dense layers to introduce non-linearity.
Pooling Layers: Reduce the spatial dimensions of feature maps to prevent overfitting and reduce computation.
Sigmoid Activation: Outputs a probability for binary classification.
Flatten Layer: Converts 2D feature maps to a format suitable for dense layers.

### Variant 2: EfficientNetB0 with K-Fold Cross-Validation

Architecture - Uses **EfficientNetB0** as a pre-trained base model for feature extraction, followed by global average pooling, dropout, and a dense layer with sigmoid activation for binary classification.
Dataset Handling: Implements manual image loading and labeling, with a `tf.data` pipeline for preprocessing and batching.
Cross-Validation: Employs **5-Fold Cross-Validation** to evaluate the model's performance across different splits of the dataset.
Performance Metrics: Outputs a classification report (precision, recall, F1-score) for each fold, providing a detailed evaluation of the model's predictive ability.

### Variant 3: EfficientNetB0 with Data Augmentation and Class Balancing

Architecture
Utilizes **EfficientNetB0** as a pre-trained feature extractor, followed by global average pooling, a dense layer with ReLU activation, and a final dense layer with sigmoid activation for binary classification. Incorporates dropout for regularization.
Dataset Handling
  * Implements data augmentation with random flips, rotations, and zoom to enhance model generalization.
  * Splits the dataset into training (70%), validation (20%), and test (10%) subsets.
Class Balancing: Calculates and applies class weights to address class imbalance in the dataset, ensuring fair training.

Training and Evaluation
  * Trains the model for 20 epochs, leveraging class weights.
  * Evaluates performance using classification reports, ROC AUC scores, and confusion matrices.
  * Visualizes training and validation accuracy and loss trends.

Visualization:
Generates a prediction visualization for an individual test image, showing the model's classification result.
Key Metrics: Outputs comprehensive performance metrics, including precision, recall, F1-score, and ROC AUC, for robust evaluation.

### Variant 4 Densenet21
Features of this Variant:
- **Pre-trained DenseNet121:** Leverages the power of a deep convolutional network pre-trained on a large dataset (ImageNet) to extract meaningful features from the medical images. Fine-tuning is applied to adapt the model to the specific sickle cell classification task.
- **Data Loading and Preprocessing:** Images are loaded, resized to a target size of 224x224, and normalized.
- **Data Augmentation:** Uses the **Albumentations** library to apply various random transformations to the training images (e.g., rotation, brightness changes, blurring). This significantly increases the size and variability of the training data, helping the model generalize better and reduce overfitting.
- **K-Fold Cross-Validation:** The training data is split into multiple folds (5 in this case), and the model is trained and evaluated on different combinations of these folds. This provides a more reliable estimate of the model's performance and helps assess its stability.
- **Model Architecture:** Builds a Sequential model on top of the pre-trained DenseNet121 base, adding layers for Global Average Pooling, Dense layers with ReLU activation and L2 regularization, Batch Normalization, and Dropout for further regularization. The final layer is a Dense layer with a sigmoid activation for binary classification output.
- **Training:** The model is compiled with the Adam optimizer and Binary Crossentropy loss. Training is performed for a specified number of epochs within each fold of the cross-validation.
- **Evaluation:** Performance is evaluated using key metrics like **Precision, Recall, F1 Score, Average Precision, and the Confusion Matrix** on a dedicated test set. The **ROC curve and AUC** are also plotted to visualize the model's trade-off between true positive and false positive rates.
  
### Model 2: Vision Transformer
1. Data Preparation
Dataset: The dataset is split into training, validation, and test sets.
Data Augmentation: Includes transformations like resizing, cropping, and normalization.
Data Loaders: Efficient loading of batches for training, validation, and testing.
2. Vision Transformer Model Setup
Model: A pre-trained ViT model from pytorch_pretrained_vit is used and fine-tuned for the classification task.
Patch Embeddings: Images are split into patches and linearly embedded.
Positional Embeddings: Adjusted to fit the input image dimensions.
3. Training Loop
Loss Function: Cross-entropy loss for multi-class classification.
Optimizer: Adam optimizer with a learning rate scheduler.
Metrics: Tracks training and validation loss and accuracy over epochs.
4. Validation Loop
Evaluates the model on the validation set to prevent overfitting and tune hyperparameters.
5. Resizing Positional Embeddings
Adjusted the positional embeddings to accommodate the image resolution and patch size.
This step ensures compatibility between the model's expected input size and the dataset.
6. Testing Loop
Evaluates the trained model on the test set.
7. Visualization
Displays a selection of test images with their true and predicted labels
