🧠 Brain Tumor Detection using Deep Learning
📌 Overview
Brain tumors are abnormal growths of cells in the brain. They can be benign (non-cancerous) or malignant (cancerous). Early detection is critical for effective treatment, but manual diagnosis using MRI scans is time-consuming and requires expert radiologists.

This project leverages deep learning to automate brain tumor detection using MRI scans. The model can distinguish between tumor-positive and tumor-negative images, improving diagnostic efficiency.

🌍 Brain Tumors: A Global & Kenyan Perspective
📊 Global Statistics
More than 308,000 new cases of brain tumors were reported worldwide in 2022.
Brain tumors account for over 251,000 deaths globally every year.
The five-year survival rate for malignant brain tumors is only 36% in adults.
Early detection increases survival rates by 50%, allowing timely treatment.
🇰🇪 Brain Tumors in Kenya
The Kenyan National Cancer Institute estimates that cancer kills over 28,000 people annually.
Brain tumors contribute to 2-3% of all cancer-related deaths in Kenya.
Many cases remain undiagnosed due to lack of MRI access and high medical costs.
⚠️ Is Brain Cancer a Silent Killer?
Yes. Brain tumors can grow silently, with symptoms appearing only in advanced stages. Many patients receive a diagnosis too late, reducing survival chances. This makes early detection crucial.

🏥 Does Early Detection Reduce Deaths?
✅ Yes! Studies show that detecting a brain tumor early increases the survival rate by 50%.
✅ Patients who begin treatment before symptoms worsen have better recovery rates.
✅ AI-driven solutions help detect tumors faster and more accurately, enabling quicker medical intervention.

🎯 Why This Project? (Impact on Society & Healthcare)
💡 Challenges in Brain Tumor Detection
✔ Late diagnosis due to lack of awareness and limited MRI access.
✔ Expensive tests make early detection difficult for low-income patients.
✔ Shortage of radiologists in Kenya and many developing countries.

🚀 How This Project Helps
🔹 AI-based Detection: Helps doctors diagnose tumors faster.
🔹 Increased Accuracy: Reduces misdiagnosis & supports early intervention.
🔹 Affordable Screening: AI-powered analysis can be cheaper than manual screening.
🔹 Remote Diagnosis: Can be used in rural areas with limited medical access.

👩‍⚕️ Healthcare Transformation
🏥 Hospitals → AI-assisted diagnosis for faster MRI interpretations.
👨‍⚕️ Doctors → Decision support tool for tumor detection.
🧑‍💻 Researchers → Advancing AI in medical imaging.
🌍 Communities → Raising awareness about early screening benefits.

🏥 What is a Brain Tumor?
A brain tumor is an abnormal mass of tissue in which cells grow uncontrollably. There are two types:

Benign Tumors – Non-cancerous, grow slowly, and are not aggressive.
Malignant Tumors – Cancerous, fast-growing, and can spread to other brain areas.
Common Symptoms:
✅ Persistent headaches
✅ Nausea or vomiting
✅ Vision or speech problems
✅ Memory issues

MRI scans are the gold standard for detecting brain tumors, and AI models can enhance this process.

🚀 Project Features
✔ Classifies MRI scans as tumor-positive or tumor-negative
✔ Uses CNN & Transfer Learning (VGG16) for improved accuracy
✔ Supports image segmentation to highlight tumor regions
✔ Offers a Flask web app for easy image uploads and detection

📂 Project Structure
graphql
Copy
Edit
Brain_Tumor_Detection/
│── datasets/               # Contains tumor (yes) and non-tumor (no) images
│── models/                 # Stores trained CNN and VGG16 models
│── static/                 # Frontend assets (CSS, images)
│── templates/              # HTML templates for the web app
│── utils/                  # Helper functions (preprocessing, segmentation)
│── app.py                  # Flask application (web UI)
│── train.py                # Model training script
│── predict.py              # Image classification script
│── segmentation.py         # Image segmentation script
│── requirements.txt        # Dependencies
│── README.md               # Documentation
train.py → Trains both CNN and VGG16 models
predict.py → Loads the model and predicts tumor presence
segmentation.py → Identifies tumor regions using OpenCV
app.py → Provides a Flask-based web UI
⚙ Technologies Used
Python 🐍
TensorFlow/Keras 🧠 (Deep Learning)
OpenCV 📷 (Image Processing)
Flask 🌐 (Web Framework)
🎯 How to Run the Project
1️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
2️⃣ Train the Model
bash
Copy
Edit
python train.py
(This will generate trained models in models/ folder.)

3️⃣ Run the Web App
bash
Copy
Edit
python app.py
(Open http://127.0.0.1:5000/ in your browser.)

4️⃣ Make a Prediction (CLI)
bash
Copy
Edit
python predict.py test_image.jpg
📊 Results & Performance
Model	Accuracy	Remarks
CNN	92%	Custom-built deep learning model
VGG16	96%	Pretrained model with Transfer Learning
📌 VGG16 performs better due to its pre-learned features from ImageNet.

📸 Sample Predictions
MRI Scan	Prediction
Tumor Detected
No Tumor
🔬 Future Improvements
✅ Improve tumor segmentation with U-Net
✅ Enhance performance with ResNet or EfficientNet
✅ Deploy the model to a cloud platform (AWS, GCP)

🤝 Contributions & Acknowledgments
This project is open-source! Contributions, feedback, and suggestions are welcome.

👩‍⚕️ Special thanks to healthcare professionals for guiding AI applications in radiology.

📜 License
This project is licensed under the MIT License.

⭐ Like the Project? Give it a Star! 🌟
If this project helps you, please star it on GitHub ⭐ to support future improvements.

