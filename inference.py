from transformers import pipeline

model_path = "./model-sentiment-indo"

classifier = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path
)

# Beberapa contoh kalimat untuk diuji
test_sentences = [
    "Produk ini sangat bagus dan pengirimannya cepat!",
    "Kecewa banget, barang datang rusak dan tidak sesuai deskripsi.",
    "Biasa aja sih, tidak ada yang spesial.",
    "Pelayanan customer service nya ramah dan responsif sekali.",
    "Harga mahal tapi kualitas mengecewakan.",
    "Hari ini cuaca di Medan cukup panas.",
]

print("=" * 70)
print("HASIL PREDIKSI SENTIMENT")
print("=" * 70)

for text in test_sentences:
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]
    print(f"\nTeks   : {text}")
    print(f"Prediksi: {label} (confidence: {score:.2%})")

print("\n" + "=" * 70)
print("Coba kalimat Anda sendiri (ketik 'exit' untuk keluar)")
print("=" * 70)

while True:
    user_input = input("\nMasukkan kalimat: ")
    if user_input.lower() == "exit":
        break
    result = classifier(user_input)[0]
    print(f"Prediksi: {result['label']} (confidence: {result['score']:.2%})")