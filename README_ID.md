# Sayaka-Nao

Sayaka-Nao adalah chatbot AI berbasis web dengan Flask yang berperan sebagai pasangan Tsuki. Dia memiliki memori untuk mengingat percakapan santai sehari-hari maupun diskusi serius guna membantu proses belajar. Untuk mendukung pembelajaran, Sayaka-Nao juga dapat menerima unggahan file PDF untuk memperluas wawasan dan memberikan respons yang lebih bermakna.

---

## Status Proyek

⚠️ **Saat ini Sayaka-Nao masih dalam tahap pengembangan.**
Proyek ini masih dalam tahap pengembangan aktif.  
Beberapa fitur belum tersedia atau masih dalam proses penyelesaian.  
Silakan berkontribusi atau laporkan isu jika ada saran dan perbaikan.

---

## Fitur Utama

* **Daily Plan**: Fitur obrolan santai sehari-hari untuk menemani dan ngobrol soal aktivitasmu.
* **Learn Plan**: Membantu proses belajar dengan interaksi dan latihan bersama Sayaka-Nao.
* **Obrolan Interaktif**: Bisa ngobrol bebas dengan Sayaka-Nao.
* **Upload PDF**: Menambah wawasan AI dengan mengimpor materi dari file PDF, memperkaya "otak" AI agar makin cerdas dan informatif.

---

## Teknologi

* **Flask** sebagai backend web framework.
* **Gemma** sebagai kerangka AI utama.
* Bahasa **Python**.
* Dependencies bisa dilihat pada file `requirements.txt`.

---

## Instalasi

1. Clone repo:

   ```bash
   git clone https://github.com/PurnamaRidzkyN/Sayaka-Nao.git
   cd Sayaka-Nao
   ```

2. Buat virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan aplikasi:

   ```bash
   python run.py
   ```

5. Akses lewat browser di `http://localhost:5000`.

---

## Catatan Penting

* Folder `app/memory` dan file `app/persona.py` tidak termasuk dalam repositori karena **.gitignore**, namun sangat penting untuk fungsi AI.
* **app/memory** berisi data memori AI, menyimpan informasi dan konteks percakapan yang membuat AI bisa "ingat" dan belajar.
* **app/persona.py** berisi profil atau karakteristik persona AI yang mengatur cara AI berinteraksi dan merespons pengguna.
* Pastikan kamu buat atau backup file dan folder tersebut agar AI berjalan dengan optimal.

---

## Cara Menggunakan

* Chat dengan Sayaka-Nao langsung di halaman utama.
* Buat rencana harian dan belajar bersama dengan fitur yang tersedia.
* Upload PDF untuk menambah wawasan AI.
* AI akan memproses materi PDF dan menyesuaikan responsnya berdasarkan wawasan baru.

---

## Kontribusi

Fork, modifikasi, dan buat pull request. Pastikan kode rapi dan terdokumentasi.

---

## Lisensi

Proyek ini dilindungi oleh **Purnama License (Non-Commercial Use)**:

* ✅ Bebas digunakan untuk keperluan pribadi, pembelajaran, dan non-komersial.
* 🚫 Dilarang digunakan untuk tujuan komersial (menjual, monetisasi, layanan berbayar, dsb) tanpa izin dari pembuat.

Untuk kerja sama atau izin komersial, hubungi:

* 📧 Email: [purnamanugraha492@gmail.com](mailto:purnamanugraha492@gmail.com)
* 📱 Instagram: [@purnama_ridzkyn\_](https://instagram.com/purnama_ridzkyn)
---

## Kontak

Purnama Ridzky N
GitHub: [https://github.com/PurnamaRidzkyN](https://github.com/PurnamaRidzkyN)

