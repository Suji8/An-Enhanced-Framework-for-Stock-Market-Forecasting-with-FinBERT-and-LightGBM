import pandas as pd
import re
import os

def clean_text(text):
    """Cleans and normalizes a given news text."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove numbers/punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # normalize spaces
    return text

def preprocess_stocknews(input_path, output_path):
    print("🚀 Starting preprocessing...")

    # Load dataset
    df = pd.read_csv(input_path)
    print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Ensure 'Date' column exists
    if 'Date' not in df.columns:
        raise ValueError("❌ No 'Date' column found. Please check your dataset file.")

    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Combine all Top1–Top25 columns
    news_columns = [col for col in df.columns if col.startswith('Top')]
    if not news_columns:
        raise ValueError("❌ No columns starting with 'Top' found. Expected Top1–Top25.")
    
    print(f"📰 Found {len(news_columns)} news columns: {news_columns[:3]}...")

    df['Combined_News'] = df[news_columns].apply(
        lambda x: ' '.join(str(i) for i in x if pd.notna(i)), axis=1
    )

    # Clean text
    df['Clean_News'] = df['Combined_News'].apply(clean_text)

    # Keep only necessary columns
    df_final = df[['Date', 'Label', 'Clean_News']]
    print(f"✅ Preprocessing complete. Final shape: {df_final.shape}")

    # Save cleaned file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"💾 Cleaned dataset saved at: {output_path}")

if __name__ == "__main__":
    # Change these paths as per your folder
    input_csv = r"E:\archive (5)\Combined_News_DJIA.csv"
    output_csv = r"E:\archive (5)\cleaned_stocknews.csv"

    preprocess_stocknews(input_csv, output_csv)
