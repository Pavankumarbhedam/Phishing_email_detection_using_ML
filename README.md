# 🧠 Phishing Email Detection using Machine Learning and Reinforcement Learning

> A hybrid AI model combining Machine Learning and Reinforcement Learning to detect phishing emails. Uses Random Forest for initial classification and a Deep Q-Network to learn from user feedback, improving accuracy and adaptability against evolving phishing attacks.

---

## 📘 Overview
This project focuses on developing an intelligent system capable of detecting phishing emails using a hybrid AI approach. The system combines **supervised learning (Random Forest)** and **reinforcement learning (Deep Q-Network)** to accurately classify emails as phishing or legitimate, while continuously learning from user feedback.

---

## ⚙️ Features
- Hybrid model: Random Forest + Deep Q-Network  
- Learns from user feedback to improve over time  
- Flask web interface for prediction and retraining  
- Evaluation using Precision, Recall, F1-score, and ROC-AUC  
- Adaptable to new phishing techniques

---

## 🧰 Tech Stack
- **Languages:** Python  
- **Frameworks:** Flask, TensorFlow / PyTorch  
- **Libraries:** scikit-learn, pandas, numpy, matplotlib  
- **Model:** Random Forest + DQN  
- **Dataset:** Public phishing email dataset (preprocessed)

---

## 🧪 How It Works
1. Extracts features from email content, subject, and links.  
2. Random Forest predicts whether the email is phishing or legitimate.  
3. User feedback is captured when misclassifications occur.  
4. Deep Q-Network updates model parameters based on feedback.  
5. The model becomes more accurate over time.

---

## 📈 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC Curve

---

## 🚀 Future Scope
- Integration with real-time email APIs  
- Cloud deployment (AWS / GCP)  
- Automated feature extraction and online learning  
- Enhanced DQN optimization

---

## 📎 Project Structure
