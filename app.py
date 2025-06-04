from flask import Flask, render_template, request
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import re
import nltk
from nltk.corpus import stopwords

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

nltk.download('stopwords')

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

model = tf.keras.models.load_model(
    'fakenewsnet_final_model_with_kaggle.h5',
    custom_objects={'AttentionLayer': AttentionLayer}
)

with open('fakenewsnet_tokenizer1.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 500

app = Flask(__name__)

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None

    if request.method == 'POST':
        input_text = request.form['news_text']
        cleaned = clean_text(input_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len)

        pred_prob = model.predict(padded)[0][0]
        prediction = 'Real' if pred_prob >= 0.5 else 'Fake'
        confidence = round(pred_prob * 100 if prediction == 'Real' else (1 - pred_prob) * 100, 2)

    return render_template('index.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
