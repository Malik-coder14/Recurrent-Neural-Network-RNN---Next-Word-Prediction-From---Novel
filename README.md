# 🌸 Shakespeare Next Word Prediction

A Deep Learning-based **Next Word Prediction** project that learns language patterns from Shakespeare's *Julius Caesar* and predicts a possible next word from text entered by the user.

## 📖 Project Overview

The project uses Shakespeare's **Julius Caesar** from the NLTK Gutenberg corpus, processes the text with tokenization and sequence generation, trains a GRU neural network using TensorFlow/Keras, and provides an interactive Streamlit application.

### Example

The user can enter:

```text
You come most carefully
```

The application processes the text and predicts a possible next word based on patterns learned during training.

## 🎯 Objectives

- Understand basic Natural Language Processing.
- Work with Shakespearean text.
- Tokenize text using Keras.
- Convert words into numerical sequences.
- Generate sequential training data.
- Train a recurrent neural network.
- Save and reuse a trained model.
- Build an interactive Streamlit application.

## 📚 Dataset

The project uses Shakespeare's **Julius Caesar** from the NLTK Gutenberg corpus.

```python
from nltk.corpus import gutenberg

data = gutenberg.raw('shakespeare-caesar.txt')
```

The text is saved locally as:

```text
julius_caesar.txt
```

## 🔄 Project Workflow

```text
Shakespeare's Julius Caesar
            ↓
       Text Loading
            ↓
     Text Preprocessing
            ↓
        Tokenization
            ↓
   Numerical Word Sequences
            ↓
      Sequence Padding
            ↓
       GRU Neural Network
            ↓
       Model Training
            ↓
    Save Model + Tokenizer
            ↓
      Streamlit Web App
            ↓
       User Input Text
            ↓
      Next Word Prediction
```

## 🧠 Deep Learning Model

The final model uses a GRU (Gated Recurrent Unit) architecture:

```text
Input
  ↓
Embedding Layer
  ↓
GRU — 150 units
  ↓
Dropout — 20%
  ↓
GRU — 100 units
  ↓
Dense Softmax Output
```

The model uses:

- Embedding dimension: 100
- First GRU: 150 units
- Second GRU: 100 units
- Dropout: 20%
- Output activation: Softmax
- Loss: Categorical Cross-Entropy
- Optimizer: Adam

The notebook also contains an LSTM experiment; the Streamlit application uses the trained GRU model.

## 🌸 Streamlit Application

The application, **Shakespeare's Word Garden**, provides:

- 🌸 Shakespeare-themed welcome message
- 📖 Text input
- 🧠 Next-word prediction
- ✨ Prediction result
- 🌿 Example phrases
- 🎨 Professional gradient background
- 📚 Dataset information
- 🧠 Model information

## 📁 Project Structure

```text
Shakespeare-Next-Word-Prediction/
│
├── My_Next_Word_Predict_Julius_Caesar.ipynb
├── streamlit.py
├── julius_caesar.txt
├── next_word_gru.h5
├── tokenizer.pickle
├── requirements.txt
└── README.md
```

| File | Purpose |
|---|---|
| `My_Next_Word_Predict_Julius_Caesar.ipynb` | Data preparation, preprocessing, sequence generation and model training |
| `julius_caesar.txt` | Shakespeare training text |
| `next_word_gru.h5` | Saved trained GRU model |
| `tokenizer.pickle` | Saved tokenizer |
| `streamlit.py` | Interactive web application |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## ⚙️ Installation

Open the project in VS Code and create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Or install the main packages manually:

```bash
python -m pip install nltk tensorflow numpy streamlit scikit-learn
```

## 📥 Download the NLTK Gutenberg Corpus

Run:

```python
import nltk
nltk.download('gutenberg')
```

Test it:

```python
from nltk.corpus import gutenberg

data = gutenberg.raw('shakespeare-caesar.txt')
print(data[:500])
```

## 🧠 Train the Model

Open:

```text
My_Next_Word_Predict_Julius_Caesar.ipynb
```

Run the notebook cells from top to bottom.

The notebook performs:

1. Loading Shakespeare's text.
2. Saving the text locally.
3. Tokenization.
4. Sequence generation.
5. Padding.
6. Model creation.
7. Model training.
8. Saving the trained model and tokenizer.

## 🌐 Run the Streamlit Application

Make sure these files are in the same folder as `streamlit.py`:

```text
next_word_gru.h5
tokenizer.pickle
```

Then run:

```bash
python -m streamlit run streamlit.py
```

Streamlit will provide a local address that can be opened in your browser.

## 💡 Using the Application

Enter a few words and click:

```text
🌸 Predict the Next Word
```

Try phrases such as:

```text
You come most carefully
```

```text
The world is
```

```text
What means this
```

```text
Good friends
```

## 🛠️ Technologies Used

- 🐍 **Python** — main programming language
- 📚 **NLTK** — Shakespeare Gutenberg corpus
- 🧠 **TensorFlow / Keras** — neural network and training
- 🔢 **NumPy** — numerical operations
- 📊 **Scikit-learn** — dataset processing
- 🌸 **Streamlit** — interactive web application

## ⚠️ Limitations

This model learns statistical patterns from the training text rather than understanding language like a human.

Therefore:

- Predictions depend on the training dataset.
- Words outside the learned vocabulary may not be handled well.
- The model can produce grammatically unusual predictions.
- Training-data size and content affect prediction quality.
- Predictions represent learned patterns rather than human-like understanding.

## 🚀 Future Improvements

- Add Top-5 or Top-10 predictions.
- Display prediction probabilities.
- Train on multiple Shakespeare plays.
- Compare LSTM and GRU performance.
- Add training accuracy and loss graphs.
- Improve text preprocessing.
- Use a larger Shakespeare dataset.
- Experiment with different sequence lengths.
- Deploy the Streamlit application online.

## 🎓 Learning Outcomes

This project demonstrates practical experience with:

- Natural Language Processing
- Text preprocessing
- Tokenization
- Sequence generation
- Padding
- Word embeddings
- GRU neural networks
- TensorFlow/Keras
- Model saving and loading
- Streamlit application development
- Connecting a trained ML model to a web interface

## 🌷 Conclusion

This project demonstrates how **NLP, Deep Learning, and Streamlit** can be combined to build an interactive Next Word Prediction system.

By learning patterns from Shakespeare's *Julius Caesar*, the model attempts to predict a possible next word from text entered by the user.

> 📖 *Let the words bloom, one prediction at a time.* 🌸

## 👨‍💻 Author

**Malik**

Python • NLP • Deep Learning • Streamlit

🌸 Happy Coding! 🌸
