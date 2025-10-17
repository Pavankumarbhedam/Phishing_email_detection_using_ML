from flask import Flask, render_template, request, session
from prediction import new_preprocess_text, preprocess_new_data, predict_new_email, feedback_loop
from stable_baselines3 import DQN
from Enviornment import EmailEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import os
import joblib
import pandas as pd
import re

# Load models
model = joblib.load('random_forest_model.pkl')
X_train = joblib.load('X_train.pkl')
y_train= joblib.load('y_train.pkl')
dqn_agent = DQN.load("dqn_model")
# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

# Configure static image folder
picFolder = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = picFolder

# Data dictionary for storing phishing email data
phishing_email = {
    'sender': [],
    'receiver': [],
    'subject': [],
    'body': [],
    'url_count': []
}

# Define global variable for preprocessed data
preprocessed_new_email_data = None
predict_label = 0

@app.route('/')
def home():
    return render_template('phishing_Home.html')

@app.route('/loading', methods=['POST'])
def loading():
    # Get data from form
    sender = request.form['sender']
    receiver = request.form['receiver']
    subject = request.form['subject']
    body = request.form['body']
    url_count = request.form['url_count']

    # Add form data to phishing_email dictionary
    phishing_email['sender'].append(sender)
    phishing_email['receiver'].append(receiver)
    phishing_email['subject'].append(subject)
    phishing_email['body'].append(body)
    phishing_email['url_count'].append(url_count)

    return render_template('loading.html')

@app.route('/predVal', methods=['GET', 'POST'])
def res():
    global preprocessed_new_email_data, predict_label
    
    # Create DataFrame from phishing_email dictionary
    new_email_df = pd.DataFrame(phishing_email)

    # Preprocess email data
    new_email_df['subject'] = new_email_df['subject'].apply(lambda x: re.sub(r'[^A-Za-z0-9 ]+', '', str(x)))
    new_email_df['body'] = new_email_df['body'].apply(lambda x: re.sub(r'<.*?>', '', str(x)))
    
    # Check if url_count is numeric
    try:
        new_email_df['url_count'] = new_email_df['url_count'].astype(float)
    except ValueError:
        return render_template('error.html', message="URL count must be a numeric value.")

    # Further preprocessing and prediction
    preprocessed_new_email_data = preprocess_new_data(new_email_df)  # Store preprocessed data globally
    predicted_action = predict_new_email(preprocessed_new_email_data, dqn_agent, model)
    predict_label = predicted_action[0]

    # Clear phishing_email dictionary after use
    for key in phishing_email:
        phishing_email[key].clear()

    # Display results based on prediction
    if predicted_action[0] == 1:
        result_message = 'Phishing Email! Be Careful!'
        return render_template('output2.html', val=result_message)
    elif predicted_action[0] == 0:
        result_message = 'Your Email is Safe!'
        return render_template('output1.html', val=result_message)
    else:
        result_message = 'The email is flagged for further review.'

@app.route('/feedback_form', methods=['GET'])
def feedback_form():
    return render_template('feedback.html')  # Render the feedback form page

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    global preprocessed_new_email_data, predict_label,dqn_agent  # Access global variables

    # Capture feedback data from the form
    true_label = request.form['correct_label']
    correct_label = 0 if true_label == 'legitimate' else 1
    env = EmailEnv(model, X_train, y_train)
    env = DummyVecEnv([lambda: env])
    dqn_agent.set_env(env)
    dqn_agent=feedback_loop(preprocessed_new_email_data, correct_label, dqn_agent, model,env)
    # Render thank you page with pop-up message
    return render_template('thankyou.html')

if __name__ == '__main__':
    app.run(debug=True)


