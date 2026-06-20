\# Sentiment Analysis Bahasa Indonesia dengan IndoBERT



Fine-tuning model \*\*IndoBERT\*\* untuk klasifikasi sentimen teks Bahasa Indonesia (positif/netral/negatif), menggunakan PyTorch dan Hugging Face Transformers.



\### 1. Setup environment



```bash

python -m venv venv

venv\\Scripts\\activate          # Windows

\# source venv/bin/activate     # Mac/Linux



\# Install PyTorch (sesuaikan CUDA version, lihat https://pytorch.org/get-started/locally/)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128



\# Install dependency lainnya

pip install -r requirements.txt

```



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

