import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Shakespeare • Next Word Predictor", page_icon="🌸", layout="centered")

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 10% 10%, rgba(255,182,193,.28), transparent 25%), radial-gradient(circle at 90% 15%, rgba(173,216,230,.30), transparent 28%), radial-gradient(circle at 50% 100%, rgba(221,160,221,.22), transparent 35%), linear-gradient(135deg,#fff9fc 0%,#f5f8ff 48%,#fdf7ff 100%); color:#25233a;}
#MainMenu, footer, header {visibility:hidden;}
.block-container {max-width:900px;padding-top:2.5rem;padding-bottom:3rem;}
.hero {text-align:center;padding:2.3rem 1.5rem 2rem;border-radius:28px;background:rgba(255,255,255,.72);border:1px solid rgba(255,255,255,.9);box-shadow:0 18px 50px rgba(73,62,105,.12);backdrop-filter:blur(12px);margin-bottom:1.5rem;}
.flowers {font-size:2rem;letter-spacing:8px;margin-bottom:.5rem;}
.hero h1 {color:#49356f;font-size:2.55rem;margin:0;font-weight:800;}
.hero h2 {color:#7b5ca8;font-size:1.12rem;font-weight:500;margin-top:.65rem;}
.hero p {color:#68657a;font-size:1rem;margin:.9rem auto 0;max-width:680px;line-height:1.7;}
.card {background:rgba(255,255,255,.76);border:1px solid rgba(255,255,255,.92);border-radius:22px;padding:1.4rem 1.5rem;box-shadow:0 12px 35px rgba(73,62,105,.09);backdrop-filter:blur(10px);margin:1rem 0;}
.section-title {color:#49356f;font-size:1.18rem;font-weight:750;margin-bottom:.4rem;}
.hint {color:#777387;font-size:.92rem;margin-bottom:.8rem;}
div[data-baseweb="input"] {border-radius:14px!important;border:1px solid #ddd5ee!important;background:#fff!important;}
div[data-baseweb="input"]:focus-within {border:1px solid #9b7bc6!important;box-shadow:0 0 0 3px rgba(155,123,198,.12)!important;}
.stButton > button {width:100%;border-radius:14px;border:0;padding:.72rem 1rem;font-size:1rem;font-weight:700;color:white;background:linear-gradient(90deg,#8064a8,#a77bc8);box-shadow:0 8px 20px rgba(128,100,168,.25);}
.stButton > button:hover {transform:translateY(-2px);box-shadow:0 12px 25px rgba(128,100,168,.32);}
.prediction {text-align:center;padding:1.5rem;margin-top:1.1rem;border-radius:20px;background:linear-gradient(135deg,rgba(255,240,248,.96),rgba(241,237,255,.96));border:1px solid #eadcf4;box-shadow:0 10px 30px rgba(120,87,145,.10);}
.prediction-label {color:#7b718b;font-size:.9rem;margin-bottom:.35rem;}
.prediction-word {color:#60427e;font-size:2.1rem;font-weight:800;}
.footer {text-align:center;color:#8b8798;font-size:.82rem;margin-top:2rem;line-height:1.6;}
.badge {display:inline-block;padding:.35rem .75rem;border-radius:999px;background:#f0eafa;color:#70558f;font-size:.78rem;font-weight:650;margin:.2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="flowers">🌸 ✨ 🌷 ✨ 🌸</div>
<h1>Welcome to Shakespeare's Word Garden</h1>
<h2>📖 Next Word Prediction powered by Deep Learning</h2>
<p>Step into the language of Shakespeare's <b>Julius Caesar</b>. Type a few words below and let the trained neural network suggest what word might come next. 🌿</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model = load_model("next_word_gru.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

try:
    model, tokenizer = load_resources()
except Exception:
    st.error("🌷 The model files could not be loaded.")
    st.info("Keep next_word_gru.h5 and tokenizer.pickle in the same folder as streamlit.py.")
    st.stop()

max_sequence_len = model.input_shape[1] + 1

def predict_next_word(text):
    token_list = tokenizer.texts_to_sequences([text.lower()])[0]
    if not token_list:
        return None
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    prediction = model.predict(token_list, verbose=0)
    predicted_index = int(np.argmax(prediction, axis=1)[0])
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    return index_to_word.get(predicted_index)

st.markdown("""
<div class="card">
<div class="section-title">🌿 Begin your Shakespearean sentence</div>
<div class="hint">Enter a few words and the model will predict the next word.</div>
</div>
""", unsafe_allow_html=True)

input_text = st.text_input("Your text", value="Please Enter your text from Novel", placeholder="For example: The world is", label_visibility="collapsed")

if st.button("🌸 Predict the Next Word"):
    if not input_text.strip():
        st.warning("🌷 Please enter some words first.")
    else:
        with st.spinner("✨ Shakespeare's words are blooming..."):
            next_word = predict_next_word(input_text)
        if next_word:
            st.markdown(f"""
            <div class="prediction">
            <div class="prediction-label">✨ The model predicts:</div>
            <div class="prediction-word">🌸 {next_word}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("🌿 I couldn't find a prediction. Try words from the training text.")

st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
for col, icon, title, value in [(c1,"📚","Dataset","Julius Caesar"),(c2,"🧠","Model","GRU Neural Network"),(c3,"✨","Task","Next Word Prediction")]:
    with col:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
        <div style="font-size:1.7rem;">{icon}</div><b>{title}</b><br><span style="color:#777387;">{value}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="card">
<div class="section-title">💡 Try these ideas</div>
<span class="badge">🌸 You come most carefully</span>
<span class="badge">🌿 The world is</span>
<span class="badge">📖 What means this</span>
<span class="badge">✨ Good friends</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">🌷 <b>Shakespeare Next Word Predictor</b> 🌷<br>Built with Python • TensorFlow • NLTK • Streamlit<br>Let the words bloom, one prediction at a time. ✨</div>
""", unsafe_allow_html=True)
