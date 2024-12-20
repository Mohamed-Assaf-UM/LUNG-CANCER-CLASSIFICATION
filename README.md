# LUNG-CANCER-CLASSIFICATION


### Overview of the Lung Cancer Classification Project

In this project, I have developed a deep learning-based system to classify lung cancer types from histopathological images using a Convolutional Neural Network (CNN). The main goal of this project is to accurately classify lung cancer into four distinct categories:

1. **Adenocarcinoma**
2. **Large Cell Carcinoma**
3. **Squamous Cell Carcinoma**
4. **Normal** (Non-cancerous)

### Novelty and Approach

- **Pre-trained Model (Transfer Learning)**: I leveraged the power of the **Xception model**, a pre-trained deep learning architecture, for feature extraction. This helps the model learn complex patterns in the images that are essential for accurate classification. By freezing the layers of the pre-trained model, I was able to fine-tune it for my specific problem, rather than training from scratch.
  
- **Custom Data Augmentation**: To improve the model's robustness and prevent overfitting, I applied **data augmentation techniques** such as horizontal flipping and rescaling during training. This helps the model generalize better when it encounters unseen data.

- **Image Preprocessing**: Each input image is processed by resizing to a fixed size (299x299 pixels) and normalizing pixel values (scaling between 0 and 1), ensuring consistency and optimizing the model's performance.

- **Model Training and Optimization**: To further improve the model, I used various callbacks such as **ReduceLROnPlateau** (to reduce learning rate when loss plateaus), **EarlyStopping** (to stop training if the model stops improving), and **ModelCheckpoint** (to save the best model during training). This helps prevent overfitting and ensures the best version of the model is saved.

- **Final Classifier**: After feature extraction using Xception, the model is fine-tuned with a **Global Average Pooling** layer followed by a **Dense layer** with a softmax activation function. This architecture allows the model to output a probability distribution over the four categories, making it suitable for multi-class classification.

### Impact and Applications

- **Medical Diagnosis**: The model is designed to assist medical professionals in diagnosing lung cancer by automatically classifying histopathological images into the four categories of cancer types. This helps in providing faster and more accurate diagnoses, ultimately improving patient outcomes.
  
- **Real-time Predictions**: With the trained model, doctors can upload patient images, and the model will predict the type of lung cancer in a matter of seconds, which can assist in deciding the treatment strategy.

---

This approach ensures high accuracy and efficiency in classifying lung cancer types, contributing to advancements in medical image analysis and supporting healthcare professionals in making more informed decisions.
---

### **1. Reading Data**
#### Code:
```python
print("Reading training images from:", train_folder)
print("Reading validation images from:", validate_folder)
```

- **Purpose:**  
  These lines display the paths of the folders containing the training and validation datasets. The `train_folder` and `validate_folder` are assumed to contain subdirectories, each representing a specific class (e.g., "Cancerous," "Non-Cancerous").

- **Why:**  
  It ensures the user knows where the data is being loaded from, helping in debugging and confirming correct file paths.

---

### **2. Image Preprocessing**
#### Libraries:
`ImageDataGenerator` from **Keras** is used here for data augmentation and normalization.

#### Parameters Explained:
```python
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
```

- **`rescale=1./255`:**
  - Pixel values in images typically range from 0 to 255. Dividing by 255 normalizes these values to the range [0, 1].
  - **Why:** Normalization helps CNNs converge faster during training and improves performance.

- **`horizontal_flip=True`:**
  - Randomly flips the images horizontally.
  - **Why:** Lung X-rays and CT scans may have different orientations due to image capture processes. Horizontal flipping increases the diversity of the training data, reducing overfitting.

---

### **3. Defining Batch Size**
```python
batch_size = 8
```

- **Purpose:**  
  The batch size determines how many images are processed together in one forward/backward pass during training.

- **Why `8`:**
  - A smaller batch size can reduce memory usage, making it suitable for systems with limited resources.
  - It also ensures frequent weight updates, potentially leading to better model generalization.

---

### **4. Data Loading and Augmentation**
#### Training Data:
```python
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)
```

- **`train_folder`:** Path to the training dataset. It expects subfolders named after class labels (e.g., "Cancer" and "No_Cancer").
- **`target_size=IMAGE_SIZE`:** Resizes all images to `(350, 350)` pixels. This ensures consistency in image dimensions fed into the CNN.
- **`batch_size=batch_size`:** Specifies the number of images processed in each batch.
- **`color_mode="rgb"`:** Loads images as RGB (3 color channels). For grayscale images, you would use `"grayscale"`.
- **`class_mode='categorical'`:**  
  - Indicates the output labels are in **one-hot encoded** format (e.g., `[1, 0]` for "Cancer" and `[0, 1]` for "No_Cancer").  
  - **Why:** This is required for multi-class classification.

#### Validation Data:
```python
validation_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)
```

- **Similar to `train_generator`, but:**
  - `test_datagen` does **not include data augmentation.**
  - **Why:** Validation data should remain unchanged to evaluate the model’s performance on unaltered, real-world data.

---

### **5. Outputs of `flow_from_directory`**
Both `train_generator` and `validation_generator`:
- **Return a generator** that loads images in batches.
- Automatically assigns labels to images based on their folder names.
- **Example Directory Structure:**
  ```
  train_folder/
      Cancer/
          image1.jpg
          image2.jpg
      No_Cancer/
          image3.jpg
          image4.jpg
  ```
  - The generator assigns:
    - Images in `Cancer/` → Label `[1, 0]`
    - Images in `No_Cancer/` → Label `[0, 1]`

---

### **Why This Approach?**
1. **Efficiency:**  
   - Generators load images in batches, preventing memory overload from loading all images at once.
2. **Data Augmentation:**  
   - Applies transformations like flipping to increase data diversity and reduce overfitting.
3. **Compatibility:**  
   - Outputs preprocessed data in a format compatible with CNNs.

---
### **Why Use `flow_from_directory` and `train_generator`?**

When training a CNN model, we need to provide the images and their labels in a format the model understands. **`flow_from_directory` and `train_generator`** handle this task for us efficiently.

---

### **Purpose of `flow_from_directory`:**
1. **Automatically Loads Images:**
   - It reads all the images from the folders (like "Cancer" and "No_Cancer") without you manually loading them one by one.

2. **Assigns Labels:**
   - It assigns the correct label to each image based on the folder name.  
     For example:  
     - Images in the folder `Cancer` get the label `[1, 0]` (if using one-hot encoding).  
     - Images in `No_Cancer` get the label `[0, 1]`.

3. **Preprocesses Images:**
   - Resizes the images to the specified size (e.g., 350x350 in your code).
   - Normalizes the pixel values to make them easier for the model to process.
   - Optionally, it applies data augmentation (like flipping or rotating the images) to make the model more robust.

4. **Generates Batches:**
   - It creates batches of images instead of loading all of them at once.  
     This saves memory because the model processes a small number of images at a time (e.g., 8 images per batch).

---

### **Purpose of `train_generator`:**
- The **`train_generator`** is the output of `flow_from_directory`.  
- It is like a pipeline that feeds the preprocessed images and labels to your CNN model during training.

Think of it as:
- **You:** "Hey, train_generator, give me 8 images and their labels to train my model!"  
- **train_generator:** "Sure! Here’s a batch of 8 images, resized, normalized, and labeled for you!"

---

### **In Simple Words:**
Imagine you’re preparing ingredients to cook (your training data).  
- **`flow_from_directory`:** It organizes the ingredients (images) from the pantry (folders), chops them to the right size (resizes), and gets them ready for cooking (training).  
- **`train_generator`:** It delivers the ingredients (images and labels) batch by batch to the chef (your CNN model), so the chef doesn’t get overwhelmed with everything at once.

---
This part of your code defines **callbacks** for training the CNN model. Callbacks are like assistants that monitor the training process and make adjustments or save progress based on specific conditions. Let’s break it down step by step:

---

### **1. ReduceLROnPlateau**
```python
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=5, verbose=2, factor=0.5, min_lr=0.000001)
```

- **Purpose:** Automatically reduces the learning rate when the model stops improving.  
- **Why is it needed?**
  - Sometimes, the model stops learning (loss stagnates or doesn’t decrease further).
  - Lowering the learning rate allows the model to make finer adjustments and escape local minima.

- **Key Parameters:**
  - `monitor='loss'`: Watches the training loss to decide when to reduce the learning rate.
  - `patience=5`: If the loss doesn’t improve for 5 epochs, it reduces the learning rate.
  - `factor=0.5`: Multiplies the current learning rate by 0.5 (halves it).
  - `min_lr=0.000001`: Ensures the learning rate doesn’t go below this value.

---

### **2. EarlyStopping**
```python
early_stops = EarlyStopping(monitor='loss', min_delta=0, patience=6, verbose=2, mode='auto')
```

- **Purpose:** Stops training early if the model’s performance stops improving.  
- **Why is it needed?**
  - To avoid overfitting by stopping the training process when the model isn’t learning anything new.
  - Saves time and computational resources.

- **Key Parameters:**
  - `monitor='loss'`: Watches the training loss.
  - `min_delta=0`: Specifies how small the improvement must be to consider it meaningful. Here, even a very small improvement will be considered.
  - `patience=6`: Stops training if the loss doesn’t improve for 6 consecutive epochs.
  - `mode='auto'`: Automatically determines whether to look for a minimum (loss) or maximum (accuracy) based on the monitored metric.

---

### **3. ModelCheckpoint**
```python
checkpointer = ModelCheckpoint(filepath='best_model.weights.h5', verbose=2, save_best_only=True, save_weights_only=True)
```

- **Purpose:** Saves the model’s weights whenever it achieves the best performance during training.  
- **Why is it needed?**
  - Ensures you don’t lose the best version of your model, even if training continues and performance worsens later (due to overfitting or randomness).
  - Allows you to load the best model later for testing or deployment.

- **Key Parameters:**
  - `filepath='best_model.weights.h5'`: Specifies the file to save the weights.
  - `verbose=2`: Prints messages about when the model is being saved.
  - `save_best_only=True`: Saves the weights only when the model achieves the best performance so far.
  - `save_weights_only=True`: Saves only the weights (not the entire model structure).

---

### **How It Works Together:**
1. **During Training:**
   - `ReduceLROnPlateau`: Adjusts the learning rate dynamically to ensure better learning.
   - `EarlyStopping`: Monitors if the model stops improving and halts training early.
   - `ModelCheckpoint`: Saves the model weights whenever it performs its best.

2. **Why All Together?**
   - Combines efficiency (ReduceLROnPlateau) and safety (EarlyStopping, ModelCheckpoint) to improve the model's training process.

---

### **Simple Analogy:**
Imagine training your model is like climbing a mountain:  
- **ReduceLROnPlateau:** Slows you down when you’re struggling to climb steep parts (adjusts learning rate).  
- **EarlyStopping:** Tells you to stop climbing if you’ve already reached the peak (no more improvement).  
- **ModelCheckpoint:** Takes a photo whenever you reach a higher altitude than before (saves the best model weights).

---

This part of the code constructs a **transfer learning model** using the **Xception** architecture, which is a pre-trained convolutional neural network (CNN). Here’s a breakdown of what each part does and its purpose:

---

### **1. Import Libraries**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, InputLayer
```
- **Purpose:** Import required modules from TensorFlow for building, modifying, and training the model.
  - `Sequential`: Allows building the model layer by layer.
  - `GlobalAveragePooling2D`: Reduces the spatial dimensions of feature maps from the pre-trained model to a single vector for each feature map.
  - `Dense`: Adds a fully connected layer for classification.

---

### **2. Define Constants**
```python
OUTPUT_SIZE = 4  # Number of classes
IMAGE_SIZE = (299, 299)  # Xception requires at least 299x299 images
```
- **OUTPUT_SIZE:** Represents the number of categories (or classes) the model needs to classify. For example, 4 categories in your lung cancer dataset.
- **IMAGE_SIZE:** Specifies the input size for images, matching Xception's requirements (minimum size is 299x299 pixels).

---

### **3. Load the Pre-Trained Xception Model**
```python
pretrained_model = tf.keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3)
)
```
- **Purpose:** Loads the Xception model pre-trained on the **ImageNet dataset** without the top (final classification) layer.
  - `weights='imagenet'`: Initializes the model with weights pre-trained on ImageNet, a large dataset of images.
  - `include_top=False`: Excludes the final fully connected (dense) layer of Xception. This allows adding a custom classifier for your dataset.
  - `input_shape=(299, 299, 3)`: Explicitly specifies the input shape as 299x299 pixels with 3 color channels (RGB).

---

### **4. Freeze Pre-Trained Layers**
```python
pretrained_model.trainable = False
```
- **Purpose:** Prevents the pre-trained model's weights from being updated during training. This retains the general knowledge learned from ImageNet and reduces training time.
- **Why is this done?**
  - The pre-trained layers already capture general features like edges, textures, and shapes.
  - Freezing helps focus on training the new classification layers specific to your dataset.

---

### **5. Build the Model**
```python
model = Sequential()
model.add(InputLayer(input_shape=(299, 299, 3)))
model.add(pretrained_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(OUTPUT_SIZE, activation='softmax'))
```
- **Layer by Layer Construction:**
  1. **InputLayer:** Explicitly defines the input shape to ensure compatibility.
  2. **Pretrained Xception Model:** Uses the frozen pre-trained model for feature extraction.
  3. **GlobalAveragePooling2D:** Reduces the output from the pre-trained model to a smaller, flat feature vector by averaging spatial information.
  4. **Dense Layer:** Adds a fully connected layer with `OUTPUT_SIZE` neurons and a `softmax` activation function for multi-class classification.

---

### **6. Print Model Summaries**
```python
pretrained_model.summary()
model.summary()
```
- **Purpose:** Provides detailed summaries of:
  - **Pre-trained model:** Shows the architecture and number of parameters in the Xception model.
  - **Final model:** Displays the full pipeline, including the pre-trained model and the newly added layers.

---

### **7. Compile the Model**
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
- **Purpose:** Configures the model for training by specifying:
  - **Optimizer (`adam`)**: Adaptive optimizer that adjusts learning rates automatically.
  - **Loss Function (`categorical_crossentropy`)**: Suitable for multi-class classification.
  - **Metrics (`accuracy`)**: Tracks accuracy during training and validation.

---

### **Purpose of this Code**
- **Transfer Learning:** Reuses the pre-trained Xception model’s knowledge to handle your lung cancer dataset. It allows you to leverage a powerful architecture without training it from scratch.
- **Classification:** Builds a custom classifier for your dataset with 4 classes.
- **Efficiency:** Freezing the pre-trained model’s layers and using `GlobalAveragePooling2D` reduces computation and avoids overfitting.

---

### **Simple Analogy:**
- Think of the pre-trained model as a **master artist** who has already learned to draw basic shapes and patterns.  
- You’re asking the artist to **freeze their basic drawing skills** (don’t re-learn) and only focus on using those skills to create artwork specific to your task (lung cancer classification).  
- The **Dense layer** at the end is like the final touch where the artwork is categorized (into 4 classes in your case).
---
When explaining the **novelty** in your CNN implementation during an interview or discussion, you can highlight the following aspects that make your approach unique or thoughtful:

---

### **1. Use of Transfer Learning (Xception)**
- **What’s novel?**  
  Instead of building a CNN from scratch, you leveraged a **state-of-the-art architecture (Xception)** pre-trained on ImageNet.  
  - This shows efficiency: you use a proven architecture that captures generalized features while focusing on training only the classification layers for your specific lung cancer dataset.
  - Xception uses **depthwise separable convolutions**, which are computationally efficient compared to traditional CNNs.

**Key Point to Mention:**  
*"Using Xception, I reduced computational cost and training time while still benefiting from a powerful, pre-trained feature extractor. This balances accuracy and efficiency."*

---

### **2. Optimizing Training with Callbacks**
- **What’s novel?**  
  - You used **ReduceLROnPlateau** to dynamically adjust the learning rate based on training performance.
  - **EarlyStopping** ensured that the model didn’t overfit or waste computational resources.
  - **ModelCheckpoint** saved only the best model weights, adding robustness to your training process.

**Key Point to Mention:**  
*"The use of advanced callbacks optimized my training process, preventing overfitting and ensuring the best possible model performance."*

---

### **3. Fine-Tuned Input Pipeline**
- **What’s novel?**  
  - You used `ImageDataGenerator` to augment the training data with transformations like horizontal flipping, increasing the dataset's diversity without needing more data.
  - You resized the images to fit Xception’s requirements, ensuring compatibility and efficient feature extraction.

**Key Point to Mention:**  
*"By augmenting the data and resizing images for compatibility, I improved the generalization capability of the model."*

---

### **4. Lightweight Final Model**
- **What’s novel?**  
  The combination of **GlobalAveragePooling2D** and a minimal **Dense layer** ensures the model is lightweight and avoids overfitting.  
  - Instead of adding multiple dense layers, which can lead to complexity and overfitting, you simplified the architecture for better efficiency.

**Key Point to Mention:**  
*"I kept the architecture lightweight, balancing simplicity and accuracy, which is especially useful for deployment on resource-constrained systems."*

---

### **5. Practical Application-Oriented Focus**
- **What’s novel?**  
  The entire pipeline is designed for real-world applicability:
  - The model works with augmented medical image data.
  - It is modular, so you can easily swap the base model if needed.
  - The saved weights (`best_model.weights.h5`) ensure reproducibility and deployment readiness.

**Key Point to Mention:**  
*"The pipeline is structured for deployment, ensuring it can be directly applied in a real-world scenario, such as detecting lung cancer from X-rays or CT images."*

---

### **6. Novel Dataset-Specific Adaptation**
- **What’s novel?**  
  - Adapting a pre-trained model like Xception for medical imaging tasks (lung cancer classification) is challenging because medical images differ significantly from natural images in ImageNet.
  - By freezing the pre-trained layers and fine-tuning the classification layers, you’ve adapted it for a specialized dataset.

**Key Point to Mention:**  
*"The adaptation of a general-purpose model like Xception to a medical imaging task demonstrates its flexibility and effectiveness in domain-specific problems."*

---

### **How to Present This During an Interview**
1. **Highlight your thought process:**  
   *"I chose Xception for its efficiency and capability to extract robust features. By freezing the pre-trained layers, I focused on training only the classification head for lung cancer detection, ensuring both accuracy and computational efficiency."*

2. **Showcase the practical benefits:**  
   *"With data augmentation and careful training optimizations like callbacks, I ensured the model generalizes well to unseen data while avoiding overfitting."*

3. **Conclude with real-world readiness:**  
   *"This approach is novel because it combines a powerful pre-trained architecture with a lightweight custom head, making it suitable for deployment in medical imaging applications."*

---
This code is for **predicting the class** of a new image using your trained model. Let’s break it into smaller sections and explain what each part does.

---

### **1. Import Necessary Libraries**
```python
from tensorflow.keras.preprocessing import image
import numpy as np
```
- **Purpose**: Import functions for image preprocessing and numerical operations.
  - `image`: Used to load and process the input image.
  - `numpy`: Handles arrays and numerical computations.

---

### **2. Define a Function to Load and Preprocess the Image**
```python
def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array
```
- **Purpose**: Prepares the input image to be compatible with the trained model.
- **Steps**:
  1. **Load the image**:  
     `image.load_img(img_path, target_size=target_size)` loads the image and resizes it to the target size (e.g., `(299, 299)` for Xception).
  2. **Convert to array**:  
     `image.img_to_array(img)` converts the image to a NumPy array so the model can process it.
  3. **Expand dimensions**:  
     `np.expand_dims(img_array, axis=0)` adds a new dimension to represent the batch size, as the model expects a batch of images.
     - Example: An image of shape `(299, 299, 3)` becomes `(1, 299, 299, 3)`.
  4. **Rescale**:  
     `img_array /= 255.0` scales pixel values to the range `[0, 1]`, just like the training data.

---

### **3. Load the Image**
```python
img_path = '/content/drive/MyDrive/LUNG_CANCER_H5/642x361_SLIDE_4_What_Does_Lung_Cancer_Look_Like.jpg'
img = load_and_preprocess_image(img_path, IMAGE_SIZE)
```
- **Purpose**: Load and preprocess a specific image (`img_path`) for prediction.

---

### **4. Make Predictions**
```python
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])
```
- **Purpose**: Use the trained model to predict the class of the input image.
- **Steps**:
  1. **Predict**:  
     `model.predict(img)` returns probabilities for each class.  
     Example: `[0.1, 0.7, 0.1, 0.1]` means class 1 has a 70% probability.
  2. **Find the class**:  
     `np.argmax(predictions[0])` gives the index of the highest probability (e.g., index `1` for `0.7`).

---

### **5. Map the Class to a Label**
```python
class_labels = list(train_generator.class_indices.keys())
predicted_label = class_labels[predicted_class]
```
- **Purpose**: Convert the numeric prediction (`predicted_class`) into a human-readable label.
  - `train_generator.class_indices`: A dictionary that maps class names to indices.
    - Example: `{'Normal': 0, 'Benign': 1, 'Malignant': 2, 'Other': 3}`
  - `class_labels`: List of class names (e.g., `['Normal', 'Benign', 'Malignant', 'Other']`).
  - `predicted_label`: Gets the class name corresponding to the predicted index.

---

### **6. Display the Prediction**
```python
print(f"The image belongs to class: {predicted_label}")
```
- **Purpose**: Prints the predicted class label.  
  - Example Output: `The image belongs to class: Malignant`

---

### **7. Show the Image with the Prediction**
```python
plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()
```
- **Purpose**: Displays the input image along with its predicted label.
  - `plt.imshow`: Displays the resized image.
  - `plt.title`: Adds the predicted label as a title.
  - `plt.axis('off')`: Hides the axis for a cleaner view.
  - `plt.show()`: Displays the image.

---

### **In Simple Terms**
1. **Load the Image**: Resize the image and prepare it for prediction.
2. **Predict the Class**: Use the trained model to determine which class the image belongs to.
3. **Map the Result**: Convert the numeric prediction into a readable label.
4. **Display the Result**: Show the image with its predicted class for easy interpretation.

---

### **Example Walkthrough**
- Suppose you load an image of a malignant tumor.
- The model outputs: `[0.05, 0.10, 0.80, 0.05]`.
- The predicted class index is `2` (highest probability, 80%).
- The class label mapped to `2` is `'Malignant'`.
- You display the image with the title: `"Predicted: Malignant"`.
---

