# Importing Necessary Libraries

import os
import pandas as pd

#for reprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tqdm import tqdm

os.chdir('..')


# Importing Dataset and basic data check

if os.path.isfile('./data/all_data.csv'):
    df = pd.read_csv('./data/all_data.csv', index_col=0)
else:
    from zipfile import ZipFile

    dfs = []
    for archive in tqdm(os.listdir('./data')):
      path_to_archive = os.path.join('./data', archive)
      with ZipFile(path_to_archive, 'r') as archive:
        for file in archive.filelist:
          with archive.open(file.filename, 'r') as fp:
            dfs.append(pd.read_csv(fp, lineterminator='\n'))

    df = pd.concat(dfs)
    df = df[['content']]
    df.content = df.content.str.replace('\n',' ')
    df.reset_index(drop=True, inplace=True)
    df.to_csv('./data/all_data.csv')

print('Shape of the dataframe: ', df.shape)
df.head()


# Data Cleaning
if os.path.isfile('./data/cleaned_data.csv'):
    df = pd.read_csv('./data/cleaned_data.csv', index_col=0)
else:
    # Lower Casing
    df['content'] = df.content.str.lower()

    # Removal of URLs
    def remove_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    df['content'] = df.content.apply(lambda x:remove_urls(x))

    # Removing Punctuations
    def remove_punctuations(text):
        return text.translate(str.maketrans('','', string.punctuation))

    df['content'] = df.content.apply(lambda x:remove_punctuations(x))

    # Removing extra spaces between words if any
    df['content'] = df['content'].apply(lambda x: " ".join(x.split()))

    # Removing Stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    def remove_stopwords(text):
        return ' '.join([words for words in str(text).split() if words not in stop_words])

    df['content'] = df['content'].apply(lambda x: remove_stopwords(x))

    # Removal of Emojis
    def remove_emoji(string):
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

    df['content'] = df['content'].apply(lambda x: remove_emoji(x))

    # Removing tweets with non-english words
    nltk.download('words')
    nltk.download('omw-1.4')
    words = set(nltk.corpus.words.words())
    to_drop = []

    for i in df.index:
        for w in nltk.wordpunct_tokenize(df.loc[i, 'content']):
            if w not in words or w.isalpha()==False:
                to_drop.append(i)
                break
            break
            
    df.drop(to_drop,inplace = True)
    
    # Lemmatization
    nltk.download('wordnet')
    le = WordNetLemmatizer()

    def lemmatize_text(text):
        return ' '.join([le.lemmatize(word,pos='v') for word in text.split()])

    df['content'] = df['content'].apply(lambda x: lemmatize_text(x))

    # Saving Cleaned Data
    df.to_csv('./data/cleaned_data.csv')

print('Length of the Dataframe after removing non-english tweets: ', df.shape[0])
df.head()


