import pandas as pd
from datasets import Dataset, DatasetDict

# URL file mentah dari repo resmi IndoNLU (format TSV, tanpa header)
BASE_URL = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/"

urls = {
    "train": BASE_URL + "train_preprocess.tsv",
    "validation": BASE_URL + "valid_preprocess.tsv",
    "test": BASE_URL + "test_preprocess.tsv",
}

# Mapping label teks -> angka (urutan sesuai konvensi IndoNLU)
label2id = {"positive": 0, "neutral": 1, "negative": 2}
id2label = {v: k for k, v in label2id.items()}

def load_split(url):
    df = pd.read_csv(url, sep="\t", header=None, names=["text", "label"])
    df["label"] = df["label"].map(label2id)
    return df

print("Mengunduh dan memuat data...")
train_df = load_split(urls["train"])
valid_df = load_split(urls["validation"])
test_df = load_split(urls["test"])

# Konversi ke format HuggingFace Dataset supaya kompatibel dengan fine-tuning nanti
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(valid_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
})

print(dataset)

print("\n--- Contoh data train ---")
print(dataset["train"][0])
print(dataset["train"][1])

print("\n--- Distribusi label (train) ---")
print(train_df["label"].map(id2label).value_counts())

# Simpan ke disk lokal supaya tidak perlu download ulang setiap kali running script lain
dataset.save_to_disk("./smsa_dataset")
print("\nDataset berhasil disimpan ke folder ./smsa_dataset")