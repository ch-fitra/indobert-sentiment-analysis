\# Sentiment Analysis Bahasa Indonesia dengan IndoBERT



Fine-tuning model \*\*IndoBERT\*\* untuk klasifikasi sentimen teks Bahasa Indonesia (positif/netral/negatif), menggunakan PyTorch dan Hugging Face Transformers.



\## 📊 Hasil



| Metrik | Validation | Test |

|---|---|---|

| Accuracy | 94.13% | 91.2% |

| F1-Macro | 91.72% | 89.04% |



\## 🛠️ Tech Stack



\- \*\*Model\*\*: \[IndoBERT base-p1](https://huggingface.co/indobenchmark/indobert-base-p1)

\- \*\*Framework\*\*: PyTorch, Hugging Face Transformers

\- \*\*Dataset\*\*: \[SmSA (IndoNLU)](https://github.com/IndoNLP/indonlu) — 11.000 data training, 3 kelas (positive/neutral/negative)

\- \*\*Hardware\*\*: NVIDIA RTX 4060 Laptop GPU (8GB VRAM), training \~4 menit untuk 3 epoch



\## 📁 Struktur Project

