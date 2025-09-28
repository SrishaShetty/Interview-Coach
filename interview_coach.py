import os
import random
import re
import tkinter as tk
from tkinter import filedialog, Toplevel
import threading
import PyPDF2
import joblib
import pandas as pd
import numpy as np
import speech_recognition as sr
import pyttsx3
import json
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize
r = sr.Recognizer()
text_speech = pyttsx3.init()
model = SentenceTransformer('all-MiniLM-L6-v2')
semantic_scores = []

def speak(text):
    print(text)
    output_box.config(state=tk.NORMAL)
    output_box.insert(tk.END, text + "\n\n")
    output_box.see(tk.END)
    output_box.config(state=tk.DISABLED)
    text_speech.say(text)
    text_speech.runAndWait()

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.lower()

def record_text(prompt=None, retries=3):
    if prompt:
        speak(prompt)

    for attempt in range(retries):
        try:
            with sr.Microphone() as source:
                speak(" Listening...")
                r.adjust_for_ambient_noise(source, duration=1)
                audio = r.listen(source, timeout=10, phrase_time_limit=20)
                MyText = r.recognize_google(audio, show_all=False)
                print(f"You said: {MyText}")
                return MyText.lower()
        except sr.UnknownValueError:
            speak(" Sorry, I couldn’t understand you.")
        except sr.RequestError:
            speak(" Service error with speech recognition.")
        except Exception as e:
            print(f"[Attempt {attempt+1}] Error: {e}")
            speak(" Something went wrong. Please try again.")

    speak(" I couldn't understand after multiple attempts.")
    return ""

def extract_name(text):
    name_keywords = ['my name is', 'i am', "it's", "this is", "name is"]
    for phrase in name_keywords:
        if phrase in text:
            return text.split(phrase)[-1].strip().split()[0].capitalize()
    return text.split()[0].capitalize()

def load_data_from_folders(base_path):
    texts, labels = [], []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.pdf'):
                    pdf_text = extract_text_from_pdf(os.path.join(folder_path, file))
                    if not pdf_text.strip():
                        print(f"Warning: Empty text extracted from {file}")
                    texts.append(pdf_text)
                    labels.append(folder)
    return texts, labels

def train_model():
    data_path = os.path.join("data", "data")
    texts, labels = load_data_from_folders(data_path)

    if not texts:
        print("No data available for training!")
        return

    vectorizer = TfidfVectorizer(stop_words=None, max_features=5000, min_df=1)
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError as e:
        print(f"Error during vectorization: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    print(f" Model Accuracy: {accuracy_score(y_test, predictions):.2%}")
    report = classification_report(y_test, predictions, output_dict=True)
    print(classification_report(y_test, predictions))

    job_role_scores = {label.upper(): round(score['f1-score'], 2) for label, score in report.items()
                       if label in clf.classes_ and isinstance(score, dict)}

    with open("role_accuracy.json", "w") as f:
        json.dump(job_role_scores, f)

    joblib.dump(clf, 'trained_model(1).pkl')
    joblib.dump(vectorizer, 'vectorizer(1).pkl')
    print("Trained classes:", clf.classes_)

def load_questions():
    qa_df = pd.read_csv("Question_Answer.csv")
    qa_dict = {}
    for _, row in qa_df.iterrows():
        role = str(row['role']).strip().upper()
        question = str(row['question']).strip()
        answer = str(row['answer']).strip()
        if role not in qa_dict:
            qa_dict[role] = []
        qa_dict[role].append((question, answer))
    return qa_dict

def evaluate_answer(user_answer, correct_answer):
    user_answer = user_answer.strip().lower()
    correct_answer = correct_answer.strip().lower()

    if not user_answer:
        return " No answer detected. Please try again."

    skip_phrases = [
        "i don't know", "no idea", "not sure", "skip", "can't answer",
        "no knowledge", "i have no clue", "i don't have much clue", "zero", "idk"
    ]
    if any(phrase in user_answer for phrase in skip_phrases):
        semantic_scores.append(0.2)
        return " No worries, let's move to the next question."

    emb_user = model.encode(user_answer, convert_to_tensor=True)
    emb_correct = model.encode(correct_answer, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(emb_user, emb_correct).item()

    semantic_scores.append(similarity_score)
    print(f"[Semantic Similarity Score]: {similarity_score:.2f}")

    if similarity_score >= 0.80:
        return " Excellent! Your answer shows strong understanding."
    elif similarity_score >= 0.60:
        return " Good understanding. You’re on the right track!"
    elif similarity_score >= 0.40:
        return " Partial understanding — try to explain more clearly."
    else:
        return " The answer doesn't show clear understanding."

def predict_job_role(resume_path, questions_dict, user_name):
    clf = joblib.load('trained_model(1).pkl')
    vectorizer = joblib.load('vectorizer(1).pkl')

    resume_text = extract_text_from_pdf(resume_path)
    if not resume_text.strip():
        speak(" The resume seems empty or unreadable. Please try again with a different file.")
        return

    resume_vector = vectorizer.transform([resume_text])
    predicted_role = clf.predict(resume_vector)[0]

    print(f"\n Suggested Job Role based on Resume: {predicted_role}")

    if predicted_role in questions_dict:
        ask_random_questions([predicted_role], questions_dict)
    else:
        print(f" No specific questions found for {predicted_role}")
        print(" Available roles with questions are:", list(questions_dict.keys()))

def ask_random_questions(matched_roles, questions_dict):
    for role in matched_roles:
        if role in questions_dict:
            questions = questions_dict[role]
            random_questions = random.sample(questions, min(5, len(questions)))
            speak(f" Interview questions for the role: {role}")

            for i, (q, a) in enumerate(random_questions):
                speak(f" {q}")
                user_answer = record_text()
                speak(f" You said: {user_answer}")
                result = evaluate_answer(user_answer, a)
                speak(result)
                # Ensure last score is processed before moving on
                if i == len(random_questions) - 1:
                    print(f"Finished all questions for role: {role}")
                    print(f"Current semantic scores: {semantic_scores}")

    speak("That's all the questions I had for you today. Thank you for your time and effort!")
    show_feedback_graph()


def show_accuracy_graph():
    if not os.path.exists("role_accuracy.json"):
        speak(" Accuracy data not found. Please train the model first.")
        return

    with open("role_accuracy.json", "r") as f:
        accuracy_data = json.load(f)

    roles = list(accuracy_data.keys())
    raw_scores = list(accuracy_data.values())

    if all(0 <= v <= 1 for v in raw_scores):
        scores = [v * 100 for v in raw_scores]
    else:
        scores = raw_scores

    graph_window = tk.Toplevel()
    graph_window.title(" Job Role Accuracy")

    fig = plt.figure(figsize=(10, 10))
    plt.barh(roles, scores, color='springgreen')
    plt.xlabel("F1-Score (%)")
    plt.title("Job Role Prediction Accuracy")
    plt.xlim(0, 100)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

def show_feedback_graph():
    if not semantic_scores:
        speak("No answers were given to generate feedback.")
        return

    indices = list(range(1, len(semantic_scores) + 1))
    print(semantic_scores)
    scores = [round(score * 100, 2) for score in semantic_scores]  # scale to 0-100

    feedback_window = tk.Toplevel()
    feedback_window.title("Interview Feedback")

    fig = plt.figure(figsize=(8, 6))
    plt.plot(indices, scores, marker='o', linestyle='-', color='dodgerblue')
    plt.ylim(0, 100)
    plt.xlabel("Question Number")
    plt.ylabel("Score (%)")
    plt.title("Your Interview Performance")
    plt.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=feedback_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


def start_interview():
    semantic_scores.clear()
    try:
        #Clear previous output
        output_box.config(state=tk.NORMAL)
        output_box.delete('1.0', tk.END)
        output_box.config(state=tk.DISABLED)

        if not os.path.exists('trained_model(1).pkl') or not os.path.exists('vectorizer(1).pkl'):
            speak(" Training the model. Please wait...")
            train_model()
        else:
            speak("Loaded pre-trained model.")

        name_text = record_text("Please say your name clearly.")

        if not name_text.strip():
            speak(" Name not captured. Please type your name.")

            def get_name_from_user():
                name_prompt = Toplevel(root)
                name_prompt.title("Enter Your Name")
                name_prompt.geometry("300x150")
                tk.Label(name_prompt, text="Please enter your name:", font=("Helvetica", 12)).pack(pady=10)
                name_entry = tk.Entry(name_prompt, font=("Helvetica", 12))
                name_entry.pack(pady=5)

                def submit_name():
                    nonlocal name_text
                    name_text = name_entry.get()
                    name_prompt.destroy()

                tk.Button(name_prompt, text="Submit", command=submit_name).pack(pady=10)
                name_prompt.wait_window()

            get_name_from_user()

        if not name_text.strip():
            speak(" Name not provided. Exiting.")
            return

        user_name = extract_name(name_text)
        speak(f"Hello {user_name}, today I am going to take your interview. All the very best!")
        speak("Please upload your resume to begin.")

        status_label.config(text=f"Hello {user_name}, please upload your resume.")
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])

        if file_path:
            questions_dict = load_questions()
            predict_job_role(file_path, questions_dict, user_name)
        else:
            speak(" No file selected.")
    except Exception as e:
        speak(f"Error: {str(e)}")

    ask_button.config(state=tk.NORMAL)
    status_label.config(text="Click 'START INTERVIEW' to start again.")
      

# GUI Setup
root = tk.Tk()
root.title("🧠 AI Interview Coach")
root.geometry("800x800")
root.configure(bg="#121212")

frame = tk.Frame(root, bg="#121212", padx=20, pady=20)
frame.pack()

title_label = tk.Label(frame, text="🎤 AI Interview Coach", font=("Helvetica", 22, "bold"), bg="#121212", fg="#00E676")
title_label.pack(pady=(0, 10))

ask_button = tk.Button(
    frame, text="🚀 START INTERVIEW",
    command=lambda: threading.Thread(target=start_interview).start(),
    font=("Helvetica", 14, "bold"), bg="#1E88E5", fg="white",
    activebackground="#1565C0", activeforeground="white",
    relief=tk.FLAT, padx=20, pady=10
)
ask_button.pack(pady=10)

accuracy_button = tk.Button(
    frame, text=" Show Accuracy", command=show_accuracy_graph,
    font=("Helvetica", 14, "bold"), bg="#43A047", fg="white",
    activebackground="#2E7D32", activeforeground="white",
    relief=tk.FLAT, padx=20, pady=10
)
accuracy_button.pack(pady=10)

status_label = tk.Label(frame, text="Click 'START INTERVIEW' to begin.", font=("Helvetica", 12), bg="#121212", fg="#CCCCCC")
status_label.pack(pady=(10, 10))

output_box = tk.Text(frame, height=30, width=80, font=("Consolas", 11), wrap=tk.WORD,
                     bg="#1E1E1E", fg="#EEEEEE", insertbackground="white")
output_box.pack(pady=10)
output_box.config(state=tk.DISABLED)

root.mainloop()