# 🍎 Fruit Classification Using CNN

This project uses a **Convolutional Neural Network (CNN)** to classify different types of fruits based on their images.
It involves **image preprocessing with OpenCV**, **deep learning model building**, and **evaluation** to accurately predict the fruit type.

---

## 🚀 Project Overview

The notebook performs the following steps:

1. **Importing Libraries** – Loading Python libraries such as TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib, and Scikit-learn.
2. **Image Preprocessing with OpenCV** –

   * Reading and processing raw images using **OpenCV** (`cv2`).
   * Resizing all images to a fixed size `(256, 256, 3)`.
   * Normalizing pixel values to the range `[0,1]` for stable training.
3. **CNN Model Building** – Constructing a Convolutional Neural Network using **Keras Sequential API** with the following layers:

   * **Conv2D Layers** – For feature extraction with filters of increasing depth (16, 32, 64).
   * **MaxPooling2D** – To reduce spatial dimensions.
   * **Dropout** – To prevent overfitting.
   * **Dense Layers** – For final classification with **softmax activation** (9 output classes).
4. **Model Compilation** – Using **Adam optimizer**, with **accuracy** as the evaluation metric.
5. **Model Training** – Training the CNN on the dataset with validation split to monitor generalization.
6. **Evaluation** –

   * Calculating **accuracy score** on the test dataset.
   * Plotting **training vs validation accuracy** curves.
   * Generating a **confusion matrix** and **classification report**.

---

## 🛠️ Technologies Used

* Python
* OpenCV (Image preprocessing)
* TensorFlow / Keras (Deep Learning)
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn (Evaluation Metrics)
* Jupyter Notebook

---

## 📂 Dataset

The dataset contains fruit images belonging to **9 different categories**.
Each image is classified into one of the following fruit classes:

* Apple
* Banana
* Mango
* Orange
* Grapes
* Watermelon
* Pineapple
* Papaya
* Guava

### Preprocessing Steps:

* Loaded images using **OpenCV** (`cv2.imread`).
* Resized to **256×256 pixels** for uniformity.
* Normalized pixel values to range `[0,1]`.
* Converted to NumPy arrays for CNN input.

---


## 📈 Expected Insights

* The CNN can **differentiate fruits** by learning unique visual features such as color, shape, and texture.
* **OpenCV preprocessing** improves image quality and ensures consistency before training.
* **Training vs validation accuracy plots** help in identifying overfitting and underfitting.

---


