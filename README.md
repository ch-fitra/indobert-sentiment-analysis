\# Sentiment Analysis Bahasa Indonesia dengan IndoBERT



!\[Python](https://img.shields.io/badge/Python-3.14-blue?logo=python\&logoColor=white)

!\[PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch\&logoColor=white)

!\[Transformers](https://img.shields.io/badge/🤗%20Transformers-5.12-yellow)

!\[License](https://img.shields.io/badge/License-MIT-green)

!\[Accuracy](https://img.shields.io/badge/Accuracy-91.2%25-success)



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

.



├── load\_data.py      # Download \& siapkan dataset dari GitHub IndoNLU



├── train.py           # Fine-tuning IndoBERT



├── inference.py        # Test model dengan kalimat custom



├── check\_gpu.py        # Verifikasi GPU/CUDA terdeteksi



├── requirements.txt    # Dependencies



└── README.md



\## 🚀 Cara Menjalankan



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



\### 2. Download \& siapkan dataset



```bash

python load\_data.py

```



\### 3. Training model



```bash

python train.py

```



\### 4. Test inference



```bash

python inference.py

```



\## 📈 Detail Pendekatan



\- \*\*Mixed precision training (fp16)\*\* untuk mempercepat training di GPU

\- \*\*F1-macro\*\* dipakai sebagai metrik utama (bukan hanya accuracy) karena dataset imbalanced (58% positive, 31% negative, 10% neutral)

\- Model checkpoint terbaik dipilih otomatis berdasarkan F1-macro tertinggi di validation set



\## ⚠️ Known Limitations



\- Performa lebih rendah pada kelas \*\*neutral\*\* akibat data imbalance (hanya \~10% dari total data training)

\- Kalimat faktual netral (tanpa opini) kadang salah diklasifikasikan sebagai positive/negative



\## 📜 Lisensi



Project ini menggunakan \[MIT License](LICENSE).

Dataset SmSA berasal dari \[IndoNLU](https://github.com/IndoNLP/indonlu). Model base dari \[IndoBenchmark](https://huggingface.co/indobenchmark).

