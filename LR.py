import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Step 1: Load the Data
data = pd.read_csv('data/mediumcleaned.csv')
text_data = data['Combined']

# Step 2: Use only 20% of the Data (subset)
data_subset = data.sample(frac=0.3, random_state=42)  # Randomly select 20% of the data
text_data_subset = data_subset['Combined']
print('step2')

# Step 3: Prepare Sequences for Predicting the Next Word
def create_sequences(text_data, window_size=2):
    sequences = []
    next_words = []

    for sentence in text_data:
        words = sentence.split()
        for i in range(len(words) - window_size):
            sequences.append(words[i:i + window_size])  # Input: sequence of words
            next_words.append(words[i + window_size])  # Output: next word

    return sequences, next_words


# Create sequences of words (window size = 2 for example)
sequences, next_words = create_sequences(text_data_subset, window_size=2)
print('step3')
# Step 4: Vectorize the Sequences (using TF-IDF)
vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w+', max_features=1000)
X = vectorizer.fit_transform([' '.join(seq) for seq in sequences])  # Convert sequence to string for vectorization
y = np.array(next_words)
print('step4')
# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('step5')
# Step 6: Train Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
print('step6')

# Step 7: Predict and Evaluate the Model
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Example prediction:
sample_sequence = 'visualise'.split()
sample_vec = vectorizer.transform([' '.join(sample_sequence[-2:])])  # Vectorize the last two words
predicted_word = lr_model.predict(sample_vec)
print(f"Predicted next word: {predicted_word[0]}")