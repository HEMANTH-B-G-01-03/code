import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
# Sample text
text = "NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet."
# a. Tokenization by word and sentence using nltk
sentences = sent_tokenize(text)
print("Sentence Tokenization:")
print(sentences)
# Word tokenization
words = word_tokenize(text)
print("\nWord Tokenization:")
print(words)
# b. Eliminate stop words using nltk
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print("\nFiltered Words (Stop Words Removed):")
print(filtered_words)
# c. Perform stemming using nltk
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print("\nStemmed Words:")
print(stemmed_words)
# d. Perform Parts of Speech tagging using nltk
pos_tags = pos_tag(words)
print("\nPart of Speech Tagging (Original Words):")
print(pos_tags) 
# If you want POS tagging after stopword removal, use:
filtered_pos_tags = pos_tag(filtered_words)
print("\nPart of Speech Tagging (After Stopword Removal):")
print(filtered_pos_tags)