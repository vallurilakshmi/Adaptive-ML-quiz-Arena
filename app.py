import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="ML Quiz Game", layout="wide", page_icon="🧠")

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("questions.csv")  # Columns: Question, Subject, Difficulty, Correct_Answer, Option1..
subjects = df['Subject'].unique().tolist()

# ---------------------------
# ML: Difficulty predictor
# ---------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Question'])
y = df['Difficulty']
difficulty_model = RandomForestClassifier()
difficulty_model.fit(X, y)

def predict_difficulty(q_text):
    return difficulty_model.predict(vectorizer.transform([q_text]))[0]

# ---------------------------
# ML: KMeans clustering
# ---------------------------
X_cluster = vectorizer.transform(df['Question'])
kmeans = KMeans(n_clusters=min(5, len(df)), random_state=42)
kmeans.fit(X_cluster)
df['Cluster'] = kmeans.labels_

# ---------------------------
# Session state
# ---------------------------
if 'players' not in st.session_state:
    st.session_state.players = {}
if 'current_player' not in st.session_state:
    st.session_state.current_player = None
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'round_key' not in st.session_state:
    st.session_state.round_key = 1

# ---------------------------
# Player Login
# ---------------------------
st.sidebar.title("Player Login")
player_name = st.sidebar.text_input("Enter Your Name")

if player_name:
    if player_name not in st.session_state.players:
        st.session_state.players[player_name] = {
            "score": 0,
            "round": 1,
            "current_difficulty": "Easy",
            "last_score": 0
        }
    st.session_state.current_player = player_name
    st.sidebar.success(f"Logged in as: {player_name}")
else:
    st.sidebar.warning("Enter your name to play.")

# ---------------------------
# Quiz Settings
# ---------------------------
NUM_QUESTIONS = st.sidebar.slider("Questions per round", 5, 15)
CATEGORY = st.sidebar.selectbox("Select Subject", ["Any"] + subjects)

# ---------------------------
# Adaptive Difficulty
# ---------------------------
def get_target_difficulty(player_name):
    player = st.session_state.players[player_name]
    last_score = player.get('last_score', 0)

    if last_score == NUM_QUESTIONS:
        return "Medium" if player["current_difficulty"] == "Easy" else "Hard"

    elif last_score <= NUM_QUESTIONS // 2:
        return "Easy" if player["current_difficulty"] == "Medium" else "Medium"

    return player["current_difficulty"]

# ---------------------------
# Fetch Questions
# ---------------------------
def fetch_questions(num, category, player_name):
    target_diff = get_target_difficulty(player_name)
    subset = df if category == "Any" else df[df['Subject'] == category]
    subset = subset[subset['Difficulty'] == target_diff]

    if subset.empty:
        subset = df[df['Difficulty'] == target_diff]

    subset = subset.drop_duplicates(subset=["Question"]).reset_index(drop=True)

    if len(subset) >= num:
        return subset.sample(num, replace=False).to_dict("records")

    selected = subset.copy()
    remaining = num - len(selected)

    pool = df[~df["Question"].isin(selected["Question"].tolist())]
    pool = pool.drop_duplicates(subset=["Question"])

    extra = pool.sample(min(remaining, len(pool)), replace=False)

    final = pd.concat([selected, extra]).sample(frac=1).to_dict("records")
    return final

# ---------------------------
# Start Quiz
# ---------------------------
if player_name and st.sidebar.button("Start / Refresh Quiz"):
    st.session_state.questions = fetch_questions(NUM_QUESTIONS, CATEGORY, player_name)
    st.session_state.user_answers = {}
    st.session_state.players[player_name]['round'] += 1
    st.session_state.round_key += 1

# ---------------------------------------
# Display Questions (NO AUTO-CHANGING FIX)
# ---------------------------------------
if st.session_state.questions and player_name:
    st.subheader(f"Round {st.session_state.players[player_name]['round']} - {CATEGORY} Quiz")

    for idx, q in enumerate(st.session_state.questions):

        qid = f"{player_name}_{st.session_state.round_key}_{idx}"

        st.markdown(f"### Q{idx+1}: {q['Question']}")

        # --------------- FIXED OPTIONS (stored once) ----------------
        if f"{qid}_options" not in st.session_state:
            opts = []

            for col in q:
                if col.startswith("Option") and pd.notna(q[col]):
                    opts.append(q[col])

            if q['Correct_Answer'] not in opts:
                opts.append(q['Correct_Answer'])

            random.shuffle(opts)

            st.session_state[f"{qid}_options"] = opts

        options = st.session_state[f"{qid}_options"]

        # --------------- FIXED USER SELECTION ----------------
        previous = st.session_state.user_answers.get(qid, "")

        selected = st.radio(
            label="",
            options=options,
            index=options.index(previous) if previous in options else 0,
            key=f"radio_{qid}"
        )

        st.session_state.user_answers[qid] = selected

# ---------------------------
# Submit Round + Results
# ---------------------------
if player_name and st.session_state.questions:

    submit_key = f"submit_{player_name}_{st.session_state.players[player_name]['round']}"

    if st.button("Submit Round", key=submit_key):

        score = 0
        st.subheader("Round Results:")

        for idx, q in enumerate(st.session_state.questions):

            qid = f"{player_name}_{st.session_state.round_key}_{idx}"
            user_ans = st.session_state.user_answers.get(qid, "")
            correct = q["Correct_Answer"]

            if user_ans == correct:
                st.success(f"Q{idx+1}: Correct ✔️ | Your Answer: {user_ans}")
                score += 1
            else:
                st.error(f"Q{idx+1}: Wrong ❌ | Your: {user_ans} | Correct: {correct}")

        st.session_state.players[player_name]['score'] += score
        st.session_state.players[player_name]['last_score'] = score
        st.session_state.players[player_name]['current_difficulty'] = get_target_difficulty(player_name)

        st.info(f"Round Score: {score}/{NUM_QUESTIONS} | Total Score: {st.session_state.players[player_name]['score']}")

# ---------------------------
# Leaderboard
# ---------------------------
st.sidebar.subheader("Leaderboard")
leaderboard = sorted(st.session_state.players.items(), key=lambda x: x[1]['score'], reverse=True)

for pname, pdata in leaderboard:
    st.sidebar.write(f"{pname}: {pdata['score']} points (Round {pdata['round'] - 1})")
