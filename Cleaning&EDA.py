import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Explicitly select only the 'Text' and 'Title' columns
data = pd.read_csv('data/medium.csv', usecols=['Text', 'Title'])

# Inspect the dataset
print("Dataset Overview:")
print(data.head())
print("\nSummary of the dataset:")
print(data.info())
print("\nMissing values per column:")
print(data.isnull().sum())

# Drop missing values (optional, based on your use case)
data = data.dropna(subset=['Text', 'Title'])

# Basic statistics for text length
data['text_length'] = data['Text'].apply(len)
data['title_length'] = data['Title'].apply(len)
print("\nBasic statistics of text length:")
print(data[['text_length', 'title_length']].describe())

# Distribution of text length
plt.figure(figsize=(10, 6))
sns.histplot(data['title_length'], bins=30, kde=True, color='orange', label='Title')
plt.title('Distribution of Title Lengths')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print("\nEDA Completed.")

# Distribution of text length
plt.figure(figsize=(10, 6))
sns.histplot(data['text_length'], bins=30, kde=True, color='blue', label='Text')
plt.title('Distribution of Text Lengths')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()

""" Cleaning the data"""

# Import necessary libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Combine 'Title' and 'Text' into one column
data['Combined'] = data['Title'].astype(str) + " " + data['Text'].astype(str)

# Lowercase conversion
data['Combined'] = data['Combined'].str.lower()

# Remove numbers
data['Combined'] = data['Combined'].apply(lambda x: re.sub(r'\d+', '', x))

# Remove punctuation
data['Combined'] = data['Combined'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Remove English stop words
stop_words = set(stopwords.words('english'))
data['Combined'] = data['Combined'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Lemmatization
lemmatizer = WordNetLemmatizer()
data['Combined'] = data['Combined'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Strip extra white space
data['Combined'] = data['Combined'].str.strip()

# Drop sparse terms (infrequent terms)
# Create a document-term matrix
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=0.01)  # Adjust min_df based on your sparsity threshold
dtm = vectorizer.fit_transform(data['Combined'])

# Get the cleaned text back after removing sparse terms
data['Combined'] = [' '.join([vectorizer.get_feature_names_out()[i] for i in row.indices]) for row in dtm]

# Save the cleaned data to a new CSV file
data_cleaned = data[['Combined']]

# Remove rows where 'Combined' column contains the string 'nan'
data_cleaned = data_cleaned[data_cleaned['Combined'].str.strip().str.lower() != 'nan']

# Save the 'Combined' column to a text file
data_cleaned['Combined'].to_csv('mediumcleaned.txt', index=False, header=False)
data_cleaned['Combined'].to_csv('mediumcleaned.csv', index=False)

