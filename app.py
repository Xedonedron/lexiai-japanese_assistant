'''
LexiAI: Dokumentasi Asisten Belajar Bahasa Jepang Berbasis Tools
======================================================================================
LexiAI adalah asisten tutor bahasa Jepang pribadi yang dirancang untuk membantu pengguna menguasai kosakata, tata bahasa, dan nuansa dalam konteks pembelajaran.
Aplikasi ini didukung oleh Gemini 2.5 Flash dan LangChain.

TUJUAN UTAMA:
1. Menjelaskan kosakata, tata bahasa, dan nuansa secara bilingual (Indonesia/Jepang).
2. Mendorong retensi dan penerapan kata-kata baru melalui pembelajaran kontekstual.
3. Memberikan contoh kalimat dalam bahasa Jepang beserta artinya dalam Bahasa Indonesia.

KOMPONEN UTAMA & FUNGSI SETUP:
-----------------------------
1. init_db()
   - Tujuan: Menginisialisasi database SQLite lokal yang terletak di 'memory.db'.
   - Tabel yang dibuat: 
     - 'vocabulary': Untuk menyimpan kamus utama atau data RAG (kata, arti, contoh).
     - 'memory': Untuk menyimpan kata-kata yang dipelajari pengguna.

2. load_vector_db()
   - Tujuan: Memuat atau membuat Vector Database FAISS untuk keperluan RAG.
   - Mekanisme: Mencoba memuat index dari folder 'extension'. Untuk load file index.faiss dan index.pkl.

FUNGSI TOOLING (Digunakan oleh LLM untuk mengambil tindakan):
-------------------------------------------------------------
Ketiga fungsi ini didaftarkan sebagai tools untuk LLM agar dapat merespons permintaan spesifik.

1. f_memory_update(word: str, meaning: str) -> str
   - Tujuan: Menyimpan kosakata baru beserta artinya ke tabel 'memory' pengguna.
   - Pengecekan: Memastikan kata belum ada di database sebelum menambahkan.
   - Pemicu: Permintaan pengguna untuk "Add this word to my vocabulary" atau "Tambahkan kata inu sebagai anjing".

2. f_memory_query(rows_query: int = 5) -> str
   - Tujuan: Mengambil sejumlah kata (default 5) secara acak dari database 'memory' pengguna untuk ditinjau.
   - Output: Mengembalikan daftar kosakata yang diformat atau pesan "Database kosong".
   - Pemicu: Permintaan pengguna seperti "Show me my vocabulary" atau meninjau kata yang dipelajari.

3. f_lookup_dictionary(question: str) -> str
   - Tujuan: Melakukan pencarian kosakata berbasis RAG (Retrieval-Augmented Generation) 
     dari Vector DB (FAISS) berdasarkan pertanyaan pengguna.
   - Proses: Melakukan `similarity_search` pada `vector_db`, lalu menggunakan hasil 
     (konteks) tersebut bersama dengan prompt RAG untuk menghasilkan jawaban kontekstual menggunakan LLM.
   - Pemicu: Permintaan pencarian kosakata, misalnya "What is X in Japanese?".

Penyempurnaan berikutnya:
    - Saat ini fungsi `f_memory_update` masih hanya ada mekanisme validasi untuk tidak memasukan kata duplikat
      namun belum ada validasi untuk melihat apakah query dari user sesuai dengan terjemahan atau tidak.
    - Diperlukan eksperimen lebih lanjut untuk menentukan model dalam proses embedding.
      Model yang digunakan saat ini adalah model `intfloat/multilingual-e5-large` dari Huggingface.
    - RAG belum bersifat dinamis karena menggunakan file PDF sebagai kamus, fungsi f_lookup_dictionary dapat dikembangkan
      dengan mengakses API dari website [Jisho](jisho.org).
    - Belum banyak penanganan error ketika fungsi ataupun fitur dari Agent jika mengalami kegagalan, hanya ada exception umum.
    - Perlu ada pembaruan halaman Streamlit agar lebih interaktif, seperti menampilkan isi kamus di sidebar halaman,
      membuat halaman khusus untuk menambahkan kosakata dengan form dan button.
'''

import streamlit as st
import sqlite3
import os
import json
from datetime import datetime
import fitz
import faiss
import pickle
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from uuid import uuid4
from langchain_core.documents import Document

# ----------------------------------------------------
# 1. KONFIGURASI UMUM & SETUP DATABASE
# ----------------------------------------------------

# Pastikan API Key ada
try:
    if "GOOGLE_API_KEY" not in os.environ:
        api_key = st.sidebar.text_input("Masukkan Google API Key Anda:", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            st.warning("Silakan masukkan Google API Key untuk melanjutkan.")
            st.stop()
except Exception as e:
    pass

# Path Database
DB_PATH = 'memory.db'

def init_db():
    """Inisialisasi database dan tabel memory serta vocabulary."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS vocabulary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT,
            meaning TEXT,
            example TEXT,
            date_added TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            kata TEXT,
            arti TEXT,
            inserted_at TEXT DEFAULT (datetime('now'))
        )
    ''')
    conn.commit()
    conn.close()
    
init_db()

def extract_and_chunk_dictionary(pdf_path):
    """
    Mengekstrak teks dari PDF kamus dan memotongnya menjadi Document chunks.
    (Berdasarkan implementasi di [28] dan [29])
    """
    try:
        doc = fitz.open(pdf_path) # Menggunakan fitz/PyMuPDF [28]
        text = "\n".join([page.get_text() for page in doc])
        doc.close()

        lines = text.splitlines()
        clean_lines = [l.strip() for l in lines if len(l.strip()) > 0]

        # Filter baris yang memiliki pola kamus (asumsi ada tanda kurung) [29]
        chunk_documents = [Document(page_content=line) 
                           for line in clean_lines 
                           if '(' in line or '（' in line]
        
        return chunk_documents
        
    except FileNotFoundError:
        st.error(f"File PDF tidak ditemukan: {pdf_path}")
        return []
    except Exception as e:
        st.error(f"Gagal mengekstrak dari PDF: {e}")
        return []
    
# ----------------------------------------------------
# 2. DEFINISI PROMPT & MODEL LLM
# ----------------------------------------------------

SYSTEM_PROMPT = """
You are LexiAI, a personal AI language learning assistant designed to help users master Japanese.
You act as a bilingual tutor who explains Japanese vocabulary, grammar, and nuance in both Indonesian and Japanese.
Your goal is to help the user understand, retain, and apply new words naturally through contextual learning.

- When explaining vocabulary, provide hiragana, and also meaning.
- Keep explanations concise but informative, focusing on practical understanding.
- Avoid giving full sentence translations unless requested.
- Always provide at least one example sentence in Japanese with its Indonesian meaning.
- Track newly learned words internally for memory updates.
- Encourage active recall and self-reflection using short follow-up questions like “Apakah anda mengerti?” and something else.

When the user is about requests to:
- "Add this word to my vocabulary", "Tambahkan kata inu ke dalam vocabulary" → call function `f_memory_update`
- For vocabulary/kamus search (e.g., "What is X in Japanese?", "Apa bahasa jepang nya sarapan?, "Apa arti asagohan?", "Apa bahasa Jepang nya kelas dari kamus?") → call function `f_lookup_dictionary`
- To review learned words (e.g., "Show me my vocabulary", "Apa saja kosakata yang sudah anda simpan?", "Apa saja kosakata yang sudah saya simpan?") → call function `f_memory_query`
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{input}")
])

# ----------------------------------------------------
# 3. SETUP RAG (Vector DB) untuk f_lookup_dictionary
# ----------------------------------------------------

FAISS_FILE = "extension\index.faiss"
DOCSTORE_FILE = "extension\index.pkl"

def load_vector_db():
    # Cek keberadaan file FAISS dan metadata
    if os.path.exists(FAISS_FILE) and os.path.exists(DOCSTORE_FILE):
        index = faiss.read_index(FAISS_FILE)

        # Load docstore
        with open(DOCSTORE_FILE, "rb") as f:
            docstore_dict = pickle.load(f)
        docstore = InMemoryDocstore(docstore_dict)

        # Bungkus ke object LangChain FAISS agar kompatibel dengan API LangChain
        vector_db = FAISS(
            index=index,
            embedding_function=None,  # embedding_function tidak diperlukan
            docstore=docstore,
            index_to_docstore_id={i: i for i in range(len(docstore_dict))}
        )
        return vector_db
    else:
        st.error(f"FAISS index atau docstore tidak ditemukan.")
        return None

vector_db = load_vector_db()

# ----------------------------------------------------
# 4. DEFINISI FUNGSI TOOLS
# ----------------------------------------------------

@tool
def f_memory_update(word: str, meaning: str) -> str:
    """Menyimpan kosakata ke database SQLite."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT 1 FROM memory WHERE kata = ?", (word.lower(),))
    exists = c.fetchone()
    
    if exists:
        conn.close()
        return f"Kata '{word}' sudah ada di database, tidak ditambahkan ulang."
        
    else:
        c.execute('''
        INSERT INTO memory (kata, arti)
        VALUES (?, ?)
    ''', (word, meaning))
    conn.commit()
    conn.close()
    return f"Berhasil menambahkan '{word}' ({meaning}) ke kosakata Anda"

@tool
def f_memory_query(rows_query: int = 5) -> str:
    """Mengambil beberapa baris data dari database 'memory'."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT kata, arti FROM memory ORDER BY RANDOM() LIMIT ?", (rows_query,))
    results = c.fetchall()
    conn.close()

    if not results:
        return "Database kosong."
        
    formatted = "\n".join([f"- {w} ({m})" for w, m in results])
    return f"Kosakata acak yang tersimpan:\n{formatted}"

@tool
def f_lookup_dictionary(question: str) -> str:
    """
    Cari kosakata dari kamus Jepang (RAG/Vector DB) berdasarkan pertanyaan pengguna.
    """ # Berdasarkan
    
    # Mencari dokumen yang relevan dari Vector DB
    relevant_docs = vector_db.similarity_search(question, k=5)
    docs_content = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Menggunakan prompt RAG untuk menghasilkan jawaban kontekstual
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Anda adalah bagian RAG, jawab pertanyaan berdasarkan konteks yang diberikan."),
        ("user", f"Context: {docs_content}\nQuestion: {question}")
    ])
    
    messages = rag_prompt.invoke({"question": question, "context": docs_content})
    response = llm.invoke(messages)
    
    # Return string plain (bukan Markdown object)
    return response.content

llm_with_tools = llm.bind_tools([f_memory_update, f_memory_query, f_lookup_dictionary])

# ----------------------------------------------------
# 5. FUNGSI BANTU UNTUK INTERAKTIVITAS
# ----------------------------------------------------

def get_vocabulary_count():
    """Mendapatkan jumlah kosakata dalam database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM memory")
    count = c.fetchone()[0]
    conn.close()
    return count

def get_recent_vocabulary(limit=10):
    """Mendapatkan kosakata terbaru."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT kata, arti, inserted_at FROM memory ORDER BY inserted_at DESC LIMIT ?", (limit,))
    results = c.fetchall()
    conn.close()
    return results

def add_vocabulary_manual(word, meaning):
    """Menambahkan kosakata secara manual."""
    return f_memory_update.invoke({"word": word, "meaning": meaning})

def delete_vocabulary(word):
    """Menghapus kosakata dari database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM memory WHERE kata = ?", (word,))
    conn.commit()
    conn.close()
    return f"Kata '{word}' berhasil dihapus"

# ----------------------------------------------------
# 6. ANTARMUKA STREAMLIT YANG LEBIH INTERAKTIF
# ----------------------------------------------------

st.set_page_config(
    page_title="LexiAI - Asisten Belajar Bahasa Jepang",
    page_icon="🇯🇵",
    layout="wide"
)

url = "https://www.linkedin.com/in/nixon-hutahaean/"
st.title("🇯🇵 LexiAI: Asisten Belajar Bahasa Jepang Pribadi")
st.markdown("Dibuat oleh: [Nixon Daniel Hutahaean](%s)" % url)

# Fungsi untuk menangani kosakata acak
def handle_random_vocabulary():
    """Menangani permintaan kosakata acak secara langsung"""
    with st.spinner("Mengambil kosakata acak..."):
        # Langsung panggil tool f_memory_query
        result = f_memory_query.invoke({"rows_query": 5})
        
        # Format hasil untuk ditampilkan
        formatted_result = f"**🎲 Kosakata Acak dari Database Anda:**\n\n{result}"
        
        # Tambahkan ke session_state messages
        if "messages" not in st.session_state:
            st.session_state.messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                AIMessage(content=(
                    "Halo! Saya LexiAI, asisten belajar bahasa Jepang pribadi Anda.\n\n"
                    "Saya memiliki tiga fitur utama untuk membantu Anda belajar bahasa Jepang secara efektif:\n"
                    "1. **Menambahkan Kosakata Baru:** Anda dapat menambahkan kata baru beserta artinya ke daftar kosakata Anda.\n"
                    "2. **Mencari Arti Kata atau Terjemahan:** Saya bisa menampilkan arti kata, terjemahan, dan informasi terkait kosakata.\n"
                    "3. **Meninjau Kosakata yang Sudah Dipelajari:** Anda bisa meminta saya menampilkan kembali kata-kata yang telah ditambahkan.\n\n"
                    "Apa yang ingin Anda lakukan hari ini?"
                ))
            ]
        
        # Tambahkan pesan user dan assistant
        st.session_state.messages.append(HumanMessage(content="Tampilkan 5 kosakata acak yang sudah saya pelajari"))
        st.session_state.messages.append(AIMessage(content=formatted_result))

# Sidebar dengan fitur interaktif
with st.sidebar:
    st.header("📚 Manajemen Kosakata")
    
    # Statistik kosakata
    vocab_count = get_vocabulary_count()
    st.metric("Total Kosakata", vocab_count)
    
    # Tab untuk berbagai fitur
    tab1, tab2, tab3 = st.tabs(["Tambah Kosakata", "Lihat Kosakata", "Hapus Kosakata"])
    
    with tab1:
        st.subheader("➕ Tambah Kosakata Baru")
        
        # Gunakan form dengan key yang unik
        with st.form("add_vocab_form", clear_on_submit=True):
            word_input = st.text_input(
                "Kata (Bahasa Jepang):", 
                placeholder="contoh: neko"
            )
            meaning_input = st.text_input(
                "Arti (Bahasa Indonesia):", 
                placeholder="contoh: kucing"
            )
            submit_button = st.form_submit_button("Tambahkan ke Kosakata")
            
            if submit_button:
                if word_input and meaning_input:
                    result = add_vocabulary_manual(word_input, meaning_input)
                    st.success(result)
                    # Tidak perlu st.rerun() karena clear_on_submit=True sudah menangani
                else:
                    st.error("Harap isi kedua field terlebih dahulu!")
    
    with tab2:
        st.subheader("📖 Kosakata Saya")
        # Pastikan memanggil get_recent_vocabulary di setiap tab
        recent_vocab_tab2 = get_recent_vocabulary(15)
        
        if recent_vocab_tab2:
            df = pd.DataFrame(recent_vocab_tab2, columns=["Kata", "Arti", "Tanggal Ditambahkan"])
            st.dataframe(df[["Kata", "Arti"]], use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada kosakata yang disimpan.")
    
    with tab3:
        st.subheader("🗑️ Hapus Kosakata")
        # Pastikan memanggil get_recent_vocabulary di setiap tab
        recent_vocab_tab3 = get_recent_vocabulary(15)
        
        if recent_vocab_tab3:
            words_to_delete = [f"{word} - {meaning}" for word, meaning, _ in recent_vocab_tab3]
            selected_vocab = st.selectbox("Pilih kata untuk dihapus:", words_to_delete)
            
            if selected_vocab and st.button("Hapus Kata Terpilih", type="primary"):
                word_to_delete = selected_vocab.split(" - ")[0]
                result = delete_vocabulary(word_to_delete)
                st.success(result)
                st.rerun()
        else:
            st.info("Tidak ada kosakata untuk dihapus.")

    # Quick actions
    st.divider()
    st.subheader("⚡ Aksi Cepat")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    with col2:
        # Tombol Kosakata Acak yang diperbaiki
        if st.button("🎲 Kosakata Acak"):
            handle_random_vocabulary()
            st.rerun()

# Area chat utama
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Chat dengan LexiAI")
    
    # Inisialisasi riwayat chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            AIMessage(content=(
                "Halo! Saya LexiAI, asisten belajar bahasa Jepang pribadi Anda.\n\n"
                "Saya memiliki tiga fitur utama untuk membantu Anda belajar bahasa Jepang secara efektif:\n"
                "1. **Menambahkan Kosakata Baru:** Anda dapat menambahkan kata baru beserta artinya ke daftar kosakata Anda.\n"
                "2. **Mencari Arti Kata atau Terjemahan:** Saya bisa menampilkan arti kata, terjemahan, dan informasi terkait kosakata.\n"
                "3. **Meninjau Kosakata yang Sudah Dipelajari:** Anda bisa meminta saya menampilkan kembali kata-kata yang telah ditambahkan.\n\n"
                "Apa yang ingin Anda lakukan hari ini?"
            ))
        ]

    # Tampilkan riwayat chat
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Fungsi untuk memproses input pengguna
    def handle_chat_input(prompt):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("LexiAI sedang memproses..."):
            ai_msg = llm_with_tools.invoke(st.session_state.messages)
            st.session_state.messages.append(ai_msg)

        final_response_content = ""

        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            tool_results = []
            
            for call in ai_msg.tool_calls:
                tool_name = call["name"]
                tool_id = call["id"]
                
                try:
                    args = json.loads(call["args"]) if isinstance(call["args"], str) else call["args"]
                    
                    if tool_name == "f_memory_update":
                        result = f_memory_update.invoke(args)
                    elif tool_name == "f_memory_query":
                        result = f_memory_query.invoke(args)
                    elif tool_name == "f_lookup_dictionary":
                        result = f_lookup_dictionary.invoke(args)
                    else:
                        result = f"Tool tidak dikenal: {tool_name}"
                        
                    tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_id))

                except Exception as e:
                    error_msg = f"Error saat menjalankan tool {tool_name}: {e}"
                    tool_results.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
            
            st.session_state.messages.extend(tool_results)
            
            with st.spinner("Mengintegrasikan hasil tool..."):
                final_response = llm_with_tools.invoke(st.session_state.messages)
                final_response_content = final_response.content
                st.session_state.messages.append(final_response)

        else:
            final_response_content = ai_msg.content

        with st.chat_message("assistant"):
            st.markdown(final_response_content)

with col2:
    st.header("📊 Progress Belajar")
    
    # Progress ring
    st.subheader("Kemajuan Kosakata")
    
    target_vocab = 100
    progress = min(vocab_count / target_vocab, 1.0)
    
    st.metric("Kosakata Dikuasai", f"{vocab_count}/{target_vocab}")
    st.progress(progress)
    
    # Tips belajar
    st.divider()
    st.subheader("💡 Tips Belajar")
    
    tips = [
        "Pelajari 5 kata baru setiap hari",
        "Ulangi kosakata kemarin sebelum belajar yang baru",
        "Gunakan kata dalam kalimat sederhana",
        "Dengarkan pengucapan yang benar",
        "Buat flashcards digital"
    ]
    
    for tip in tips:
        st.write(f"• {tip}")

# Input chat - SELALU DI BAWAH seperti chatbot biasa
prompt = st.chat_input("Tulis pesan Anda di sini...")
if prompt:
    handle_chat_input(prompt)