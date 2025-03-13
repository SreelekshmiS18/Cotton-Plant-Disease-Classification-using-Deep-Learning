🌱 Cotton Plant Disease Classification using Deep Learning

🌍 Overview
Cotton is a crucial cash crop, but diseases can significantly impact yield and quality. This project develops an AI-powered disease detection system that classifies cotton plant leaves as healthy or infected by specific diseases using Deep Learning. By leveraging Convolutional Neural Networks (CNNs), this system can assist farmers and agricultural experts in early disease detection, leading to better crop management and improved yields.

🗂 Dataset
The dataset consists of six major classes, including:

✅ Healthy Leaf – Leaves with no disease.

🐛 Aphids – Small insects causing plant damage.

🦠 Bacterial Blight – A bacterial infection affecting cotton crops.

🦟 Army Worm – Larvae that feed on cotton leaves.

🌿 Powdery Mildew – A fungal infection forming white patches.

🎯 Target Spot – Fungal disease causing necrotic leaf spots.

🔹 Images were collected from real-world cotton farms and preprocessed to ensure uniformity in size, quality, and distribution.

🛠 Technologies & Tools

🚀 Programming Language: Python

🖥 Frameworks & Libraries:

Data Handling: numpy, pandas, glob

Image Processing: OpenCV, skimage, matplotlib

Machine Learning: scikit-learn

Deep Learning: TensorFlow, Keras

Pretrained Model: MobileNetV2 (for feature extraction)

🔍 Approach

1️⃣ Data Collection & Preprocessing

Loaded images from directories using glob.

Resized and normalized images for optimal training.

Augmented data for better generalization.

2️⃣ Model Architecture

Implemented a CNN-based classifier using TensorFlow Keras.
Used MobileNetV2 as a feature extractor to enhance performance.
Added fully connected layers for classification.
Applied Softmax activation for multi-class classification.

3️⃣ Training & Evaluation

Split dataset into training & testing sets.
Trained using categorical cross-entropy loss.
Evaluated using accuracy, precision, recall, and confusion matrix.

📊 Results
💡 High classification accuracy achieved on unseen images!
📌 The model successfully differentiates between healthy and diseased cotton leaves with robust generalization.

🚀 Future Enhancements & Real-World Applications

🔸 Real-Time Disease Detection – Deploy AI in mobile apps for farmers.

🔸 Drone-Based Monitoring – Integrate with drones for large-scale crop health analysis.

🔸 Edge AI for IoT Devices – Optimize the model for low-power agricultural devices.

🔸 More Disease Classes – Expand the dataset to include other crop diseases.

🔸 Multilingual Mobile App – Provide AI-powered disease diagnosis in local languages for better adoption.

🔗 Let’s Revolutionize Agriculture with AI! 🚀
If you found this project useful, ⭐ Star this repository and join the mission to empower farmers with AI! 🌱🤖
