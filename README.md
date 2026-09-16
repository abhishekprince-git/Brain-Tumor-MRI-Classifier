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

## 3. Literature Survey

### 3.1 Brain Tumor Detection Using CNN

This survey focuses on CNN architectures for automated brain tumor detection from MRI images. CNNs achieve high accuracy with minimal pre-processing by automatically learning critical features, eliminating the need for manual feature extraction and outperforming traditional methods like SVM for complex medical imaging tasks.

### 3.2 Transfer Learning in Medical Image Analysis

Research shows that transfer learning using pre-trained models like **VGG16** and **ResNet50** significantly improves detection accuracy while reducing computation time. Fine-tuning these models cuts training time by approximately 50% compared to training from scratch, achieving better validation accuracy and minimizing overfitting on limited medical datasets.

### 3.3 Binary Classification Approach for Tumor Detection

Researchers have explored binary classification to distinguish between tumor and non-tumor MRI images, simplifying initial screening with a clear diagnostic output.

These models achieve training accuracy above 97% and validation accuracy of 70–90%, utilizing data augmentation techniques to improve generalization and prevent overfitting.

### 3.4 Comparative Analysis of Deep Learning Models

Comparative research reveals that deeper networks like **ResNet50** outperform simpler models in accuracy and F1 score, as its residual framework prevents degradation in deep networks.

Furthermore, studies highlight the necessity of evaluating **precision, recall, and F1 score** alongside accuracy for a comprehensive assessment in medical diagnostic performance.

---

## 4. Tools to Be Used

The following tools and technologies are used in this project:

- **Dataset** : https://data.mendeley.com/datasets/zwr4ntf94j/1
- **Python 3.6 or above**
- **Google Colab / Jupyter Notebook**
- **TensorFlow & Keras**
- **NumPy**
- **Scikit-learn**

---

## 5. Expected Outcome

This project will demonstrate a fully functional brain tumor detection system that can:

- Classify MRI images into **tumor / no-tumor** categories with high accuracy.
- Reduce diagnosis time from hours to seconds.
- Provide a non-invasive alternative to traditional biopsy methods.
- Achieve training accuracy above **95%** and validation accuracy above **85%**.
- Reduce computation time by **50%** through transfer learning techniques.

---

## 6. Applications

Brain tumor detection using machine learning has extensive applications in different areas:

### Healthcare and Medical Diagnosis

AI-powered detection software assists radiologists in quickly identifying brain tumors from MRI scans, enabling faster treatment decisions and improving patient outcomes.

### Early Screening Programs

Automated detection systems can be deployed in hospitals and diagnostic centers for mass screening, helping identify tumors at early stages when treatment is most effective.

### Telemedicine and Remote Diagnosis

The system can support healthcare delivery in remote areas where specialist radiologists may not be readily available, enabling preliminary diagnosis and timely referrals.

### Medical Education and Training

The detection system can serve as a training tool for medical students and residents, helping them understand tumor characteristics and improve their diagnostic skills.

---

## 7. Challenges

The major challenges in brain tumor detection from MRI data include:

### Data Requirements

CNN models require large amounts of labeled training data, which can be difficult to obtain in medical imaging due to privacy concerns and the need for expert annotation.

### Class Imbalance

Medical datasets often have imbalanced classes (more normal scans than tumor scans), which can bias the model and affect detection accuracy.

### Image Variability

MRI images vary in quality, resolution, and acquisition parameters across different hospitals and equipment, requiring preprocessing and normalization techniques.

### Overfitting

Deep learning models are prone to overfitting, especially with limited medical datasets, necessitating techniques like data augmentation, regularization, and cross-validation.
