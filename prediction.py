import pandas as pd
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from scipy.sparse import hstack
import joblib
def new_preprocess_text(text):
    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english') + list(punctuation))
    tokens = [word for word in word_tokenize(text) if word not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Convert to lowercase
    tokens = [word.lower() for word in tokens]

    # Remove special characters and digits
    tokens = [word for word in tokens if word.isalpha()]

    return ' '.join(tokens)


# Step 3: Preprocess the new email data
def preprocess_new_data(new_data):
    # Apply the preprocessing function to the subject and body columns
    #tfidf_subject = TfidfVectorizer(stop_words='english', max_features=5000)
    #tfidf_body = TfidfVectorizer(stop_words='english', max_features=5000)
    scaler = joblib.load('scaler.pkl')
    tfidf_subject = joblib.load('tfidf_subject.pkl')
    tfidf_body = joblib.load('tfidf_body.pkl')
    encoder = joblib.load('encoder.pkl')

    new_data['subject'] = new_data['subject'].apply(new_preprocess_text)
    new_data['body'] = new_data['body'].apply(new_preprocess_text)

    # Extract features from the subject and body columns
    new_data['subject_length'] = new_data['subject'].apply(len)
    new_data['body_length'] = new_data['body'].apply(len)

    # Extract keyword presence feature
    keywords = [
        "Urgent",
        "Immediate Action Required",
        "Verification Needed",
        "Invoice",
        "Account Issue",
        "Password Reset",
        "Security Alert",
        "Action Required",
        "Support ID",
        "Payment",
        "Update",
        "offer",
        "shipping",
        "customer",
        "free",
        "upgrade",
        "win",
        "support",
        "fast",
        "special",
        "click"
    ]


    # Extract keyword presence feature from either subject or body
    new_data['keyword_presence'] = new_data.apply(lambda row: any(keyword in row['subject'] or keyword in row['body'] for keyword in keywords), axis=1)

    # Extract sender domain feature
    new_data['sender_domain'] = new_data['sender'].apply(lambda x: x.split('@')[-1].split('>')[0] if '>' in x else x.split('@')[-1])

    # Extract receiver domain feature
    new_data['receiver_domain'] = new_data['receiver'].apply(lambda x: x.split('@')[-1].split('>')[0] if '>' in x else x.split('@')[-1])

    new_data = new_data.drop(['receiver', 'sender'], axis=1)

    # Use the pre-fitted vectorizers and encoder to transform the new data
    X_subject = tfidf_subject.transform(new_data['subject'])
    X_body = tfidf_body.transform(new_data['body'])

    X_domains = encoder.transform(new_data[['sender_domain', 'receiver_domain']])

    # Combine all features
    X_other = new_data[['subject_length', 'body_length', 'url_count', 'keyword_presence']].values
    X_other = X_other.astype(np.float64)

    # Stack all features together
    X_combined = hstack([X_subject, X_body, X_other, X_domains])

    # Use the pre-fitted scaler to transform the new data
    X_combined = scaler.transform(X_combined)

    return X_combined

def predict_new_email(email_features, dqn_agent, model):
    rf_pred_prob = model.predict_proba(email_features)[0, 1]
    dqn_obs = np.array([[rf_pred_prob]], dtype=np.float32)
    dqn_action, _ = dqn_agent.predict(dqn_obs)
    return dqn_action  # 0 = Legitimate, 1 = Phishing

def feedback_loop(email_features, true_label, dqn_agent, model, env):
    max_attempts = 10
    for attempt in range(max_attempts):
        predicted_label = predict_new_email(email_features, dqn_agent, model)[0]

        if predicted_label == true_label:
            print(f"Model correctly classified after {attempt + 1} attempts.")
            break
        else:
           # print("Incorrect prediction. Providing feedback...")
            collect_feedback_and_retrain(email_features, true_label, dqn_agent, model, env)
            
def collect_feedback_and_retrain(email_features, true_label, dqn_agent, model, env):
    predicted_label = predict_new_email(email_features, dqn_agent, model)[0]

    if predicted_label != true_label:
        print("Incorrect prediction detected. Updating model with feedback.")
        current_index = env.envs[0].current_index - 1
        env.envs[0].provide_feedback(current_index, true_label)
        corrected_obs = np.array([[model.predict_proba(email_features)[0, 1]]], dtype=np.float32)
        env.envs[0].current_index = current_index
        dqn_agent.learn(total_timesteps=2000)
    return dqn_agent
