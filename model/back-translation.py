from deep_translator import GoogleTranslator
import pandas as pd
import time

# Load dataset with 'source' column
df = pd.read_csv("model/fakenewsnet_combined2.csv")
df = df[df['label'] == 1]  # Only real news

# Drop missing and sample 200 entries
real_news = df.dropna(subset=['text']).astype(str).sample(n=200, random_state=42)

augmented_data = []

for i, row in real_news.iterrows():
    try:
        original_text = row['text']
        source = row['source']  # Keep the source

        # Translate EN -> FR -> EN
        fr = GoogleTranslator(source='en', target='fr').translate(original_text)
        back = GoogleTranslator(source='fr', target='en').translate(fr)

        augmented_data.append({
            'text': back,
            'label': 1,
            'source': source
        })

        print(f"{len(augmented_data)}/200 done")
        time.sleep(1.5)  # To avoid rate limits

    except Exception as e:
        print(f"Failed at {i}: {e}")
        continue

# Save as DataFrame
aug_df = pd.DataFrame(augmented_data)
aug_df.to_csv("model/real_news_augmented2.csv", index=False)
print("Augmented file with source saved as 'real_news_augmented2.csv'")
