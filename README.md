# 🇯🇵 LexiAI - Asisten Belajar Bahasa Jepang Pribadi

![LexiAI Banner](https://via.placeholder.com/800x200/FF6B6B/FFFFFF?text=LexiAI+-+Asisten+Belajar+Bahasa+Jepang+Pribadi)

LexiAI adalah asisten tutor bahasa Jepang pribadi yang dirancang untuk membantu pengguna menguasai kosakata, tata bahasa, dan nuansa dalam konteks pembelajaran. Aplikasi ini didukung oleh **Gemini 2.5 Flash** dan **LangChain** untuk memberikan pengalaman belajar yang interaktif dan personal.

## ✨ Fitur Utama

### 🎯 Pembelajaran Personal
- **Manajemen Kosakata Pribadi**: Tambah, lihat, dan hapus kosakata dalam database pribadi Anda
- **Progress Tracking**: Pantau perkembangan belajar dengan statistik kosakata yang dikuasai
- **Kosakata Acak**: Tinjau ulang kosakata yang sudah dipelajari dengan fitur random

### 🔍 Alat Belajar Cerdas
- **Kamus Jepang-Indonesia**: Cari arti, terjemahan, dan informasi kosakata secara instan
- **RAG (Retrieval-Augmented Generation)**: Sistem pencarian cerdas dari kamus terstruktur
- **Context-Aware Responses**: AI memahami konteks pembelajaran Anda

### 💬 Interaksi Natural
- **Chat Interface**: Berinteraksi dengan AI seperti berbicara dengan tutor manusia
- **Tool Integration**: AI secara otomatis menggunakan tools yang relevan
- **Multi-modal Learning**: Kombinasi teks, statistik, dan visualisasi

## 🚀 Cara Memulai

### Prasyarat
- Python 3.8 atau lebih tinggi
- Google API Key untuk Gemini AI
- Database SQLite (otomatis dibuat)

### Instalasi

1. **Clone atau download project ini**
```bash
git clone [repository-url]
cd LexiAI
```

2. **Install dependencies**
```bash
pip install streamlit sqlite3 langchain google-generativeai faiss-cpu pymupdf pandas
```

3. **Setup Google API Key**
   - Dapatkan API key dari [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Masukkan API key di sidebar aplikasi

4. **Jalankan aplikasi**
```bash
streamlit run app.py
```

## 📖 Panduan Penggunaan

### 🏠 Dashboard Utama

Aplikasi terbagi menjadi dua bagian utama:

**Kolom Kiri - Chat Interface**
- Berinteraksi dengan LexiAI seperti chatbot
- Ajukan pertanyaan tentang bahasa Jepang
- Minta bantuan memahami tata bahasa
- Diskusikan kosakata baru

**Kolom Kanan - Progress & Tips**
- Lihat statistik kosakata yang sudah dikuasai
- Progress bar target pembelajaran
- Tips belajar efektif

### 📚 Manajemen Kosakata (Sidebar)

#### ➕ Tambah Kosakata Baru
1. Buka tab "Tambah Kosakata"
2. Masukkan kata dalam bahasa Jepang
3. Masukkan arti dalam bahasa Indonesia
4. Klik "Tambahkan ke Kosakata"
5. **Form otomatis terkosongkan** setelah berhasil ditambahkan

#### 📖 Lihat Kosakata
- Tampilkan daftar 15 kosakata terbaru
- Data ditampilkan dalam format tabel rapi
- Sortir berdasarkan tanggal ditambahkan

#### 🗑️ Hapus Kosakata
- Pilih kata dari dropdown
- Konfirmasi penghapusan
- Data langsung terupdate

### ⚡ Aksi Cepat

#### 🔄 Refresh Data
- Memperbarui semua tampilan data terbaru

#### 🎲 Kosakata Acak
- **Langsung menampilkan 5 kosakata acak** tanpa prompt tambahan
- Perfect untuk sesi review kilat

### 💬 Interaksi dengan AI

#### Contoh Pertanyaan yang Bisa Diajukan:

**Untuk Menambah Kosakata:**
```
"Tambahkan kata '綺麗' dengan arti 'cantik' ke database saya"
```

**Untuk Mencari Arti:**
```
"Apa arti dari '頑張って'?"
"Jelaskan penggunaan partikel 'は' dan 'が'"
```

**Untuk Review:**
```
"Tampilkan kosakata yang sudah saya pelajari"
"Berikan saya 3 kata acak untuk direview"
```

**Untuk Belajar Konsep:**
```
"Jelaskan perbedaan antara huruf hiragana, katakana, dan kanji"
"Bagaimana cara membentuk kalimat lampau dalam bahasa Jepang?"
```

## 🔧 Tips Memaksimalkan Penggunaan

### 🎯 Strategi Belajar Efektif

1. **Konsistensi Harian**
   - Gunakan fitur "Kosakata Acak" setiap hari untuk review
   - Targetkan menambah 5-10 kata baru per hari

2. **Active Recall**
   - Coba ingat arti kata sebelum melihat jawaban
   - Gunakan kosakata dalam kalimat praktis

3. **Contextual Learning**
   - Minta AI untuk memberikan contoh kalimat
   - Pelajari kosakata dalam konteks percakapan

### 💡 Fitur Lanjutan

**Manfaatkan Tool Integration:**
- AI otomatis menggunakan kamus ketika Anda bertanya tentang kosakata
- Sistem penyimpanan memastikan kosakata tidak duplikat
- Riwayat chat menjaga konteks pembelajaran

**Progress Tracking:**
- Set target pribadi (default: 100 kosakata)
- Monitor kemajuan dengan progress bar
- Gunakan statistik sebagai motivasi

## 🗂️ Struktur Data

### Database Schema
```
vocabulary:
- id (INTEGER PRIMARY KEY)
- word (TEXT)
- meaning (TEXT)
- example (TEXT)
- date_added (TEXT)

memory:
- kata (TEXT)
- arti (TEXT)
- inserted_at (TEXT)
```

### File Konfigurasi
- `memory.db` - Database SQLite untuk penyimpanan kosakata
- Vector database untuk pencarian kamus yang cepat

## 🛠️ Troubleshooting

### Masalah Umum dan Solusi

**API Key Error:**
- Pastikan Google API Key valid
- Cek quota penggunaan di Google Cloud Console

**Database Issues:**
- File `memory.db` akan dibuat otomatis
- Pastikan folder memiliki permission write

**Performance:**
- Kosakata besar mungkin mempengaruhi kecepatan
- Gunakan fitur hapus untuk menjaga database optimal

## 📞 Dukungan

Jika Anda mengalami masalah atau memiliki saran:
1. Periksa bagian troubleshooting di atas
2. Pastikan semua dependencies terinstall dengan benar
3. Cek koneksi internet untuk API calls

## 🎯 Roadmap Fitur Mendatang

- [ ] Sistem spaced repetition
- [ ] Latihan kuis interaktif
- [ ] Audio pronunciation
- [ ] Multiple user support
- [ ] Export/import data
- [ ] Mobile app version

---

**LexiAI** - Membuat belajar bahasa Jepang menjadi lebih mudah, personal, dan menyenangkan! 🎉

*Dibangun dengan ❤️ menggunakan Streamlit, Gemini AI, dan LangChain*