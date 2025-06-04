import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Layer, Input, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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

tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])

max_len = 500
X = pad_sequences(sequences, maxlen=max_len)
y = df['label'].values

embedding_dim = 300
embedding_index = {}
with open("model/glove.6B.300d.txt", encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

word_index = tokenizer.word_index
num_words = min(15000, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i < num_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

input_layer = Input(shape=(max_len,))
embedding = Embedding(input_dim=num_words, output_dim=embedding_dim,
                      weights=[embedding_matrix], trainable=True)(input_layer)
bilstm1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(embedding)
bilstm2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(bilstm1)
attention = AttentionLayer()(bilstm2)
dense1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005))(attention)
ln = LayerNormalization()(dense1)
drop = Dropout(0.5)(ln)
dense2 = Dense(32, activation='relu')(drop)
output = Dense(1, activation='sigmoid')(dense2)

model = Model(inputs=input_layer, outputs=output)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-4),
    metrics=['accuracy']
)

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
class_weights = dict(enumerate(class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train)))

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=[es],
    class_weight=class_weights
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.grid(True)
plt.show()

y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

model.save("fakenewsnet_final_model_with_kaggle.h5")
with open("fakenewsnet_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model trained and saved.")