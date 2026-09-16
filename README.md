# Brain Tumor MRI Classifier

## 1. Aim of the Project

To implement a CNN model using TensorFlow and Keras for classifying MRI scans into **tumorous** and **non-tumorous** categories.

---

## 2. Introduction

A brain tumor is caused by the abnormal and uncontrolled growth of cells within the brain, and is broadly classified into **non-cancerous (benign)** and **cancerous (malignant)** types.

Traditionally, the diagnosis of brain tumors relies on manual visual inspection of MRI images by radiologists, or on a stereotactic biopsy in which a neurosurgeon drills into the skull to collect tissue for histological grading. While biopsy is considered the gold standard, it carries significant risks such as bleeding, infection, seizures, stroke, and in rare cases even death, and it is still not 100% accurate.

Because of these risks, non-invasive imaging techniques such as MRI have become the preferred approach for diagnosing brain tumors. However, manual interpretation of MRI scans is time-consuming, tedious, and prone to human error, especially given the unstructured shape, variable size, and unpredictable location of tumors.

This project proposes an automated approach using a **Convolutional Neural Network (CNN)** to classify MRI images as normal or abnormal (tumorous). The use of **transfer learning** with pre-trained architectures such as **VGG16** and **ResNet50** is also explored to improve classification accuracy and reduce training time, ultimately aiming to save radiologist time, reduce diagnostic error, and support timely treatment planning.

---

## 3. Tools to Be Used

The following tools and technologies are used in this project:

- **Dataset** : https://data.mendeley.com/datasets/zwr4ntf94j/1
- **Python 3.6 or above**
- **Google Colab / Jupyter Notebook**
- **TensorFlow & Keras**
- **NumPy**
- **Scikit-learn**

---

## 4. Expected Outcome

### Live Demo

The Brain Tumor MRI Classifier is deployed on Streamlit.  
You can access the application and test the model here:

🔗 **[Open the Streamlit App](https://brain-tumor-mri-classifier-msit.streamlit.app/)**


This project demonstrates a fully functional brain tumor detection system that can:

- Classify MRI images into **tumor / no-tumor** categories with high accuracy.
- Reduce diagnosis time from hours to seconds.
- Provide a non-invasive alternative to traditional biopsy methods.
- Achieve training accuracy above **95%** and validation accuracy above **85%**.
- Reduce computation time by **50%** through transfer learning techniques.

---

## 5. Challenges

The major challenges in brain tumor detection from MRI data include:

### Data Requirements

CNN models require large amounts of labeled training data, which can be difficult to obtain in medical imaging due to privacy concerns and the need for expert annotation.

### Class Imbalance

Medical datasets often have imbalanced classes (more normal scans than tumor scans), which can bias the model and affect detection accuracy.

### Image Variability

MRI images vary in quality, resolution, and acquisition parameters across different hospitals and equipment, requiring preprocessing and normalization techniques.

### Overfitting

Deep learning models are prone to overfitting, especially with limited medical datasets, necessitating techniques like data augmentation, regularization, and cross-validation.

---

## 6. Conclusion

This project demonstrates the use of CNN and transfer learning techniques for automated brain tumor classification from MRI images. The developed system provides fast and accurate tumor/non-tumor classification and can serve as a supportive tool for medical screening and analysis.

> **Note:** This project is intended for research and educational purposes and is not a substitute for professional medical diagnosis.
