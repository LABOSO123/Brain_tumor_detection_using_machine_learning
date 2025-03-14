# Brain Tumor Detection and Segmentation using Machine Learning

## 🚀 Introduction
Brain tumors are among the most **critical health challenges worldwide**, with **high mortality rates** due to late diagnosis and limited access to advanced imaging techniques. **Early detection** significantly improves survival rates, making AI-driven solutions crucial for **faster and more accurate diagnoses.**


## 🧠 What is a Brain Tumor?
A **brain tumor** is an abnormal growth of cells in the brain. Tumors can be:
    - **Benign (non-cancerous)** – Slower growth, less likely to spread.
    - **Malignant (cancerous)** – Aggressive, fast-spreading, and life-threatening.


### 📊 Brain Tumor Statistics
- In **Kenya**, over **3,000 brain tumor cases** are reported annually, with many going undiagnosed due to limited access to specialists.
- **Worldwide**, brain and nervous system cancers cause nearly **250,000 deaths annually** (WHO).
- **Early detection** can improve survival rates by **up to 80%** with timely medical intervention.

## 🌍 Why This Project Matters
This AI-driven project aims to:
✅ **Detect brain tumors** from MRI scans with **high accuracy**
✅ **Segment tumors** for precise medical analysis
✅ **Improve healthcare accessibility** by automating preliminary diagnoses
✅ **Reduce misdiagnoses** and assist radiologists in decision-making

## 🏗️ Project Structure
This project consists of:
1. **CNN-Based Model** for classification (Tumor vs. No Tumor)
2. **Segmentation Model** for tumor localization
3. **Transfer Learning Implementation** to compare performance
4. **Flask Web App** for user-friendly interaction

## 🖥️ Technologies Used
- **Python, TensorFlow, Keras** – Deep Learning Frameworks
- **OpenCV, PIL** – Image Processing
- **Flask** – Web Application
- **NumPy, Pandas, Matplotlib** – Data Handling & Visualization

## 🔬 Model Details
We use **Convolutional Neural Networks (CNNs)** to classify brain scans as **Tumor** or **No Tumor**. Additionally, **U-Net segmentation** is applied to detect tumor regions.

### **Baseline Model** (CNN)
- Image Classification Model trained on MRI scan datasets.
- 64x64 input image size, **Categorical Cross-Entropy Loss**, Adam Optimizer.

### **Transfer Learning (EfficientNetV2 / ResNet50)**
- Pretrained models improve accuracy and reduce training time.

### **Segmentation Model (U-Net / Mask R-CNN)**
- Highlights the **tumor region** in MRI images for better diagnosis.

## 🔧 How to Run the Project
### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Train the Model**
```python
python train.py
```

### **3️⃣ Run Flask App**
```bash
python app.py
```

### **4️⃣ Upload an MRI Scan & Get Prediction!**

## 🚀 Future Improvements
🔹 **Enhance segmentation accuracy** with advanced architectures  
🔹 **Deploy as a cloud-based API for hospitals**  
🔹 **Improve dataset size & diversity** for generalization  
🔹 **Integrate explainability (XAI) for medical insights**

## ❤️ Impact in Healthcare
This project demonstrates how **AI in radiology** can:  
✔️ Improve **diagnostic speed & accuracy**  
✔️ Help **medical professionals detect tumors early**  
✔️ **Reduce misdiagnoses** in low-resource settings  
✔️ **Save lives through early intervention**

---
📌 **GitHub Repo:** [(https://github.com/LABOSO123)]  
✉️ **Contact:** [labosofaith5@gmail.com]  

🚀 **Let's revolutionize AI in healthcare!**

