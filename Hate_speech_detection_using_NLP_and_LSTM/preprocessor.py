import pandas as pd
import spacy
from collections import Counter
import pickle
import os

class Preprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.vocab = None
        self.MAX_LEN = 20

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        # Check if columns exist before dropping
        cols_to_drop = ['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither']
        existing_cols = [col for col in cols_to_drop if col in df.columns]
        if existing_cols:
            df.drop(existing_cols, axis=1, inplace=True)
        return df

    def preprocess_text(self, df):
        # Clean text
        df['processed_tweet'] = df['tweet'].str.replace("[^a-zA-Z]", " ", regex=True)
        df['processed_tweet'] = df['processed_tweet'].str.replace(r'[\s]+', ' ', regex=True)
        
        # Lemmatization and stopword removal
        df['lemma_tweet'] = df['processed_tweet'].apply(self.lemmatization)
        df['final_tweet'] = df['lemma_tweet'].apply(self.remove_stopwords)
        
        # Tokenization
        df['tokens'] = df['final_tweet'].apply(self.spacy_tokenizer)
        return df

    def lemmatization(self, text):
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def remove_stopwords(self, text):
        doc = self.nlp(text)
        return " ".join([token.text for token in doc if not token.is_stop])

    def spacy_tokenizer(self, text):
        doc = self.nlp(text.lower())
        return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    def build_vocab(self, tokens_list):
        all_tokens = [token for tokens in tokens_list for token in tokens]
        vocab_counter = Counter(all_tokens)
        vocab = {word: idx + 1 for idx, (word, _) in enumerate(vocab_counter.most_common())}
        vocab['<UNK>'] = len(vocab) + 1
        vocab['<PAD>'] = 0
        return vocab

    def tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

    def pad_sequence(self, seq):
        if len(seq) >= self.MAX_LEN:
            return seq[:self.MAX_LEN]
        else:
            return seq + [self.vocab['<PAD>']] * (self.MAX_LEN - len(seq))

    def process(self, file_path):
        df = self.load_data(file_path)
        df = self.preprocess_text(df)
        
        # Build vocabulary
        self.vocab = self.build_vocab(df['tokens'].tolist())
        
        # Convert tokens to IDs and pad sequences
        df['input_ids'] = df['tokens'].apply(self.tokens_to_ids)
        df['padded_ids'] = df['input_ids'].apply(self.pad_sequence)
        
        # Save vocabulary
        with open('vocab.pkl', 'wb') as f:
            pickle.dump((self.vocab, self.MAX_LEN), f)
        
        return df[['padded_ids', 'class']]

# Main function to run when script is executed directly
if __name__ == '__main__':
    # Initialize preprocessor
    preprocessor = Preprocessor()
    
    # Path to raw data - update if your file is in a different location
    input_file = r'Hate_speech_detection_using_NLP_and_LSTM\labeled_data.csv'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found. Please check the path.")
    
    # Process data
    processed_df = preprocessor.process(input_file)
    
    # Save processed data to a pickle file
    output_file = 'preprocessed_data.pkl'
    processed_df.to_pickle(output_file)
    
    print(f"Preprocessing complete! Results saved to {output_file}")
    print(f"Vocabulary saved to vocab.pkl")
    print(f"Processed data shape: {processed_df.shape}")