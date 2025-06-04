import pandas as pd
import numpy as np
import re
import pickle
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = inputs * a
        return K.sum(output, axis=1)

with open("fakenewsnet_tokenizer1.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("fakenewsnet_final_model_with_kaggle.h5", compile=False,
                   custom_objects={'AttentionLayer': AttentionLayer})

df_main = pd.read_csv("model/fakenewsnet_combined2.csv")
df_aug = pd.read_csv("model/real_news_augmented.csv")
df_fake = pd.read_csv("model/Fake.csv")
df_true = pd.read_csv("model/True.csv")

df_aug['label'] = 1
df_fake['label'] = 0
df_true['label'] = 1

df_fake['text'] = df_fake['title'].astype(str) + " " + df_fake['text'].astype(str)
df_true['text'] = df_true['title'].astype(str) + " " + df_true['text'].astype(str)

df_kaggle = pd.concat([df_fake[['text', 'label']], df_true[['text', 'label']]], ignore_index=True)
df_main = df_main[['text', 'label']]
df_aug = df_aug[['text', 'label']]

df = pd.concat([df_main, df_aug, df_kaggle]).drop_duplicates(subset='text')
df.dropna(subset=['text', 'label'], inplace=True)
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(int)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", 'url', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(clean_text)

X = pad_sequences(tokenizer.texts_to_sequences(df['text']), maxlen=500)
y = df['label'].values
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

accuracy = (y_pred == y_test).mean()
print(f"\n Test Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
