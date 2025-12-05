import requests
import html
import pandas as pd

# --------------------------
# Settings
# --------------------------
NUM_QUESTIONS = 50  # number of questions to download
CATEGORY = "Any"    # can be "Any", "Science", "Math", etc.

print("Downloading real dataset...")

# --------------------------
# Fetch questions from Open Trivia DB
# --------------------------
url = f"https://opentdb.com/api.php?amount={NUM_QUESTIONS}&type=multiple"
response = requests.get(url).json()

if "results" not in response:
    print("Error: No questions received from API.")
    exit()

questions_list = []
for item in response["results"]:
    question = html.unescape(item["question"])
    correct = html.unescape(item["correct_answer"])
    incorrect = [html.unescape(i) for i in item["incorrect_answers"]]
    
    # Combine options and shuffle
    options = incorrect + [correct]
    import random
    random.shuffle(options)
    
    # Save as dict
    questions_list.append({
        "Question": question,
        "Option1": options[0],
        "Option2": options[1],
        "Option3": options[2],
        "Option4": options[3],
        "Correct_Answer": correct,
        "Category": item.get("category", "General")
    })

# --------------------------
# Convert to DataFrame
# --------------------------
df = pd.DataFrame(questions_list)

# --------------------------
# Save to CSV
# --------------------------
df.to_csv("questions.csv", index=False)
print("Dataset saved as questions.csv with", len(df), "questions.")
