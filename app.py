# ============================================================
# FlightBot v3 — Chatbot Tiket Pesawat
# Deploy-ready untuk Streamlit Cloud
# Dataset: penjualan_tiket_pesawat.csv
# Maskapai : Garuda, Citilink, Lion Air
# Kota     : Jakarta, Medan, Surabaya
# Pembayaran: E-Wallet, Transfer Bank, Credit Card
# ============================================================

import re
import random
import warnings
import pandas as pd
import numpy as np
import nltk
import streamlit as st

warnings.filterwarnings("ignore")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('stopwords', quiet=True)


# ============================================================
# 1. DATASET Q&A
# ============================================================
flight_qa_data = [
    # ── HARGA ──
    {"category": "harga", "question": "berapa harga tiket pesawat dari jakarta",
     "answer": "Berdasarkan data penjualan, rata-rata harga tiket pesawat dari Jakarta sekitar Rp1.206.583. Harga bervariasi tergantung maskapai dan waktu pemesanan."},
    {"category": "harga", "question": "berapa harga tiket pesawat dari medan",
     "answer": "Berdasarkan data penjualan, rata-rata harga tiket pesawat dari Medan sekitar Rp1.247.802. Tersedia maskapai Garuda, Citilink, dan Lion Air."},
    {"category": "harga", "question": "berapa harga tiket pesawat dari surabaya",
     "answer": "Berdasarkan data penjualan, rata-rata harga tiket pesawat dari Surabaya sekitar Rp1.294.778. Harga tergantung maskapai dan tanggal keberangkatan."},
    {"category": "harga", "question": "berapa harga tiket garuda indonesia",
     "answer": "Harga tiket Garuda Indonesia rata-rata sekitar Rp1.263.633. Garuda merupakan maskapai full service dengan layanan premium."},
    {"category": "harga", "question": "berapa harga tiket citilink",
     "answer": "Harga tiket Citilink rata-rata sekitar Rp1.238.239. Citilink adalah maskapai low cost carrier (LCC) yang terjangkau."},
    {"category": "harga", "question": "berapa harga tiket lion air",
     "answer": "Harga tiket Lion Air rata-rata sekitar Rp1.249.858. Lion Air adalah maskapai low cost carrier populer di Indonesia."},
    {"category": "harga", "question": "tiket pesawat murah jakarta medan surabaya",
     "answer": "Tiket pesawat antara Jakarta, Medan, dan Surabaya tersedia mulai sekitar Rp509.210. Pilih Citilink atau Lion Air untuk harga lebih terjangkau."},
    {"category": "harga", "question": "bagaimana cara mendapatkan tiket pesawat paling murah",
     "answer": "Tips mendapatkan tiket murah: (1) Pesan 1-2 bulan sebelum keberangkatan, (2) Pilih maskapai LCC seperti Citilink atau Lion Air, (3) Manfaatkan promo flash sale."},

    # ── MASKAPAI ──
    {"category": "maskapai", "question": "maskapai apa saja yang tersedia",
     "answer": "Maskapai yang tersedia: Garuda Indonesia (full service), Citilink (low cost), dan Lion Air (low cost). Semua melayani rute Jakarta, Medan, dan Surabaya."},
    {"category": "maskapai", "question": "maskapai terbaik garuda citilink lion air mana yang bagus",
     "answer": "Dari data penjualan: Garuda (184 transaksi) unggul di layanan premium, Citilink (154 transaksi) dan Lion Air (162 transaksi) populer sebagai pilihan ekonomis."},
    {"category": "maskapai", "question": "perbedaan garuda dan citilink dan lion air",
     "answer": "Garuda Indonesia: maskapai full service, harga premium. Citilink & Lion Air: maskapai low cost carrier (LCC), harga lebih terjangkau namun fasilitas lebih terbatas."},
    {"category": "maskapai", "question": "apa itu low cost carrier maskapai murah",
     "answer": "Low Cost Carrier (LCC) adalah maskapai dengan harga tiket lebih murah, biasanya tanpa fasilitas tambahan seperti bagasi gratis dan makanan. Contoh: Citilink dan Lion Air."},

    # ── RUTE ──
    {"category": "rute", "question": "rute penerbangan yang tersedia jakarta medan surabaya",
     "answer": "Rute penerbangan yang tersedia mencakup kota Jakarta, Medan, dan Surabaya. Tersedia penerbangan antar ketiga kota tersebut dengan Garuda, Citilink, dan Lion Air."},
    {"category": "rute", "question": "penerbangan dari jakarta ke medan",
     "answer": "Penerbangan dari Jakarta ke Medan tersedia melalui Garuda, Citilink, dan Lion Air. Rata-rata harga tiket sekitar Rp1.200.000 - Rp1.300.000."},
    {"category": "rute", "question": "penerbangan dari jakarta ke surabaya",
     "answer": "Penerbangan dari Jakarta ke Surabaya tersedia melalui Garuda, Citilink, dan Lion Air. Rata-rata harga tiket sekitar Rp1.200.000 - Rp1.300.000."},
    {"category": "rute", "question": "penerbangan dari medan ke surabaya",
     "answer": "Penerbangan dari Medan ke Surabaya tersedia dengan beberapa pilihan maskapai. Cek ketersediaan di website maskapai atau aplikasi travel."},

    # ── PEMBAYARAN ──
    {"category": "pembayaran", "question": "metode pembayaran apa yang tersedia untuk beli tiket",
     "answer": "Metode pembayaran yang tersedia: E-Wallet (paling populer, 187 transaksi), Credit Card (157 transaksi), dan Transfer Bank (156 transaksi)."},
    {"category": "pembayaran", "question": "bisa bayar tiket pesawat pakai e-wallet",
     "answer": "Ya! E-Wallet adalah metode pembayaran paling populer untuk tiket pesawat, mencakup sekitar 37% dari total transaksi."},
    {"category": "pembayaran", "question": "bisa bayar tiket pesawat pakai kartu kredit credit card",
     "answer": "Ya, pembayaran dengan kartu kredit (Credit Card) tersedia dan mencakup sekitar 31% transaksi. Kadang ada promo cashback khusus kartu kredit tertentu."},
    {"category": "pembayaran", "question": "bisa bayar tiket pesawat pakai transfer bank",
     "answer": "Ya, Transfer Bank tersedia sebagai metode pembayaran (sekitar 31% transaksi). Pastikan transfer dalam waktu yang ditentukan agar pemesanan tidak dibatalkan."},

    # ── JADWAL ──
    {"category": "jadwal", "question": "jadwal penerbangan garuda indonesia",
     "answer": "Jadwal Garuda Indonesia bervariasi setiap hari. Silakan cek website resmi Garuda atau aplikasi travel untuk jadwal terbaru sesuai rute yang diinginkan."},
    {"category": "jadwal", "question": "jadwal penerbangan lion air hari ini",
     "answer": "Jadwal Lion Air berbeda setiap hari. Silakan cek website resmi atau aplikasi untuk jadwal terbaru sesuai rute yang diinginkan."},
    {"category": "jadwal", "question": "jadwal penerbangan citilink",
     "answer": "Jadwal Citilink dapat dilihat di website citilink.co.id atau aplikasi travel. Citilink melayani rute Jakarta, Medan, dan Surabaya."},

    # ── BOOKING ──
    {"category": "booking", "question": "cara pesan tiket pesawat online langkah langkah",
     "answer": "Cara booking tiket: (1) Pilih rute dan tanggal, (2) Isi data penumpang, (3) Pilih kursi (opsional), (4) Pembayaran via E-Wallet/Kartu Kredit/Transfer Bank, (5) E-ticket dikirim via email."},
    {"category": "booking", "question": "apakah bisa pilih kursi saat booking tiket",
     "answer": "Ya, beberapa maskapai memungkinkan pemilihan kursi saat booking, namun bisa dikenakan biaya tambahan."},

    # ── BAGASI ──
    {"category": "bagasi", "question": "berapa kg bagasi gratis pesawat domestik",
     "answer": "Bagasi gratis biasanya 15-20 kg untuk penerbangan domestik. Garuda biasanya 20 kg, LCC seperti Citilink dan Lion Air bisa berbeda atau tidak termasuk bagasi."},
    {"category": "bagasi", "question": "apa itu bagasi kabin pesawat",
     "answer": "Bagasi kabin adalah barang bawaan yang dibawa masuk ke pesawat, biasanya maksimal 7 kg dengan dimensi tertentu."},

    # ── REFUND ──
    {"category": "refund", "question": "cara refund tiket pesawat",
     "answer": "Refund tiket dapat dilakukan melalui aplikasi atau website tempat pembelian. Syarat dan ketentuan tergantung kebijakan maskapai (Garuda, Citilink, atau Lion Air)."},
    {"category": "refund", "question": "apakah tiket pesawat bisa di reschedule ubah jadwal",
     "answer": "Ya, tiket bisa diubah jadwal (reschedule) tergantung jenis tiket dan mungkin dikenakan biaya tambahan. Cek kebijakan masing-masing maskapai."},

    # ── PROMO ──
    {"category": "promo", "question": "promo diskon tiket pesawat murah",
     "answer": "Promo tiket pesawat biasanya tersedia saat Harbolnas, flash sale maskapai, atau promo kartu kredit. Pantau website Garuda, Citilink, dan Lion Air secara rutin."},
    {"category": "promo", "question": "cara dapat diskon tiket pesawat promo",
     "answer": "Tips diskon: Gunakan kode promo, bayar dengan kartu kredit yang punya cashback, atau booking lewat aplikasi travel yang sering ada promo."},

    # ── BANDARA ──
    {"category": "bandara", "question": "bandara di jakarta apa saja",
     "answer": "Bandara utama di Jakarta adalah Soekarno-Hatta (CGK) dan Halim Perdanakusuma (HLP). Garuda, Citilink, dan Lion Air beroperasi di Soekarno-Hatta."},
    {"category": "bandara", "question": "berapa jam sebelum penerbangan harus di bandara",
     "answer": "Disarankan tiba di bandara 2 jam sebelum keberangkatan untuk penerbangan domestik, agar cukup waktu untuk check-in dan pemeriksaan keamanan."},

    # ── UMUM ──
    {"category": "umum", "question": "apa itu e ticket pesawat",
     "answer": "E-ticket adalah tiket elektronik yang berisi detail penerbangan dan digunakan saat check-in. Bisa ditunjukkan lewat hp atau dicetak."},
    {"category": "umum", "question": "apa yang harus dibawa saat naik pesawat dokumen",
     "answer": "Dokumen penting: KTP/Paspor, e-ticket, dan bagasi sesuai aturan maskapai (Garuda, Citilink, atau Lion Air)."},
    {"category": "umum", "question": "topik apa saja yang bisa ditanyakan capability chatbot",
     "answer": "Chatbot ini membantu topik: harga tiket, maskapai (Garuda/Citilink/Lion Air), rute (Jakarta/Medan/Surabaya), pembayaran (E-Wallet/Kartu Kredit/Transfer Bank), jadwal, booking, bagasi, refund, dan promo."},
    {"category": "umum", "question": "tidak tahu tidak mengerti pertanyaan tidak relevan",
     "answer": "Maaf, saya tidak menemukan informasi yang sesuai. Silakan tanya seputar tiket pesawat, maskapai, harga, jadwal, atau booking."},
]


# ============================================================
# 2. PREPROCESSING
# ============================================================
lemmatizer = WordNetLemmatizer()

indonesian_stopwords = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'atau', 'juga', 'dalam', 'tidak', 'akan', 'ada',
    'saya', 'kamu', 'anda', 'ia', 'mereka', 'kami', 'kita',
    'apa', 'bagaimana', 'berapa', 'kapan', 'dimana', 'siapa', 'apakah', 'cara',
    'lebih', 'sangat', 'dapat', 'nya', 'pun', 'lagi', 'belum',
    'telah', 'namun', 'tapi', 'serta', 'meski', 'agar', 'supaya', 'hal',
    'mau', 'ingin', 'cari', 'tanya', 'info', 'informasi',
    'dong', 'nih', 'ya', 'aja', 'kok',
    'the', 'is', 'are', 'was', 'what', 'how', 'why', 'when', 'where'
}
english_stopwords = set(stopwords.words('english'))
all_stopwords = indonesian_stopwords.union(english_stopwords)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df = pd.DataFrame(flight_qa_data)
df['processed_question'] = df['question'].apply(preprocess_text)


# ============================================================
# 3. CHATBOT ENGINE
# ============================================================
class FlightChatbotEngineV3:

    def __init__(self, dataframe, threshold=0.15, top_k=3):
        self.df = dataframe
        self.threshold = threshold
        self.top_k = top_k
        self.conversation_history = []
        self._build_index()
        self._define_rules()

    def _build_index(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df['processed_question']
        )

    def _define_rules(self):
        self.rules = [
            ('rute populer',       'rute',     r'(jakarta.*medan|jakarta.*surabaya|medan.*surabaya|rute penerbangan)'),
            ('harga tiket',        'harga',    r'(harga|murah|termurah|biaya|berapa|pesan tiket)'),
            ('jadwal penerbangan', 'jadwal',   r'(jadwal|ada penerbangan|penerbangan pagi|penerbangan siang|penerbangan malam|jam berangkat)'),
            ('maskapai terbaik',   'maskapai', r'(maskapai|garuda|citilink|lion air|terbaik|lebih nyaman|lebih bagus)'),
            ('promo & diskon',     'promo',    r'(promo|diskon|sale|cashback|akhir pekan)'),
            ('greeting',           None,       r'(halo|hai|\bhi\b|hello|selamat pagi|selamat siang|selamat malam)'),
            ('capability',         None,       r'(bisa apa|fitur|topik apa|apa saja yang bisa|yang bisa dicari)'),
        ]

    def _check_rules(self, text):
        text_lower = text.lower()
        for label, df_category, pattern in self.rules:
            if re.search(pattern, text_lower):
                if label == 'greeting':
                    return "👋 Halo! Mau cari tiket ke mana hari ini?"
                if label == 'capability':
                    cap = self.df[
                        (self.df['category'] == 'umum') &
                        (self.df['question'].str.contains('topik|capability', na=False))
                    ]
                    if len(cap) > 0:
                        return f"[✈️ umum | RULE]\n\n{cap.iloc[0]['answer']}"
                    return "[✈️ umum | RULE]\n\nSaya bisa bantu cari tiket, harga, jadwal, maskapai, booking, promo, dan bagasi."
                filtered = self.df[self.df['category'] == df_category]
                if len(filtered) > 0:
                    row = filtered.sample(1).iloc[0]
                    return f"[✈️ {label} | RULE]\n\n{row['answer']}"
        return None

    def _search_tfidf(self, query):
        processed = preprocess_text(query)
        vec = self.vectorizer.transform([processed])
        scores = cosine_similarity(vec, self.tfidf_matrix).flatten()
        top_results = np.argsort(scores)[::-1][:self.top_k]
        return [(idx, scores[idx]) for idx in top_results]

    def get_response(self, user_input):
        if not user_input.strip():
            return "Masukkan pertanyaan."
        rule_response = self._check_rules(user_input)
        if rule_response:
            return rule_response
        results = self._search_tfidf(user_input)
        best_idx, best_score = results[0]
        row = self.df.iloc[best_idx]
        if best_score < self.threshold or row['category'] == 'umum':
            return "🤔 Maaf, aku belum menemukan info yang cocok. Coba tanya seputar harga, jadwal, maskapai, atau rute."
        self.conversation_history.append(user_input)
        return (
            f"[✈️ {row['category']} | TF-IDF]\n\n"
            f"{row['answer']}\n\n"
            f"─────────────────\n"
            f"💡 Tips: Cek harga di beberapa hari untuk dapat tiket termurah."
        )


# ============================================================
# 4. STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="FlightBot v3",
    page_icon="✈️",
    layout="centered"
)

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1f5e, #2d3a8c, #1565c0);
            padding: 20px 24px; border-radius: 15px; color: white; margin-bottom: 20px;">
    <h2 style="margin:0;">✈️ FlightBot v3 — Flight Assistant</h2>
    <p style="margin:4px 0 0 0; opacity:0.8;">🟢 Online · Chatbot Tiket Pesawat | Jakarta · Medan · Surabaya</p>
</div>
""", unsafe_allow_html=True)

# Inisialisasi bot & chat history di session state
if "bot" not in st.session_state:
    st.session_state.bot = FlightChatbotEngineV3(df)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Halo! Saya FlightBot ✈️\nSilakan tanya seputar harga tiket, jadwal, maskapai, atau rute penerbangan!"}
    ]

# Tampilkan chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Quick chips
st.markdown("**Quick questions:**")
cols = st.columns(5)
chips = [
    ("Jakarta ke Surabaya", "tiket jakarta ke surabaya"),
    ("Tiket murah",         "tiket murah"),
    ("Jadwal",              "jadwal penerbangan"),
    ("Maskapai",            "maskapai terbaik"),
    ("Harga tiket",         "harga tiket"),
]
for col, (label, query) in zip(cols, chips):
    if col.button(label, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": label})
        response = st.session_state.bot.get_response(query)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Input chat
if user_input := st.chat_input("Ketik pertanyaan, misal: harga tiket dari jakarta..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = st.session_state.bot.get_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Tombol clear
if st.button("🗑️ Clear Chat", use_container_width=False):
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Halo! Saya FlightBot ✈️\nSilakan tanya seputar harga tiket, jadwal, maskapai, atau rute penerbangan!"}
    ]
    st.session_state.bot.conversation_history = []
    st.rerun()
