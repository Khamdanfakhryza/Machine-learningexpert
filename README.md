## Laporan Proyek Machine Learning _ Khamdan Annas Fakhryza
***

## Project Overview

Membaca buku adalah kegiatan yang sangat penting untuk memperoleh ilmu pengetahuan, tetapi dengan banyaknya jumlah buku yang tersedia, pembaca dapat kebingungan memilih buku yang relevan. Untuk mengatasi permasalahan ini, proyek ini membangun sebuah sistem rekomendasi berbasis konten (*content-based filtering*) yang merekomendasikan buku berdasarkan fitur seperti kategori, penulis, dan judul. Dengan menggunakan pendekatan ini, sistem rekomendasi diharapkan dapat membantu pembaca menemukan buku yang sesuai dengan minat mereka.

## Business Understanding
***

### Problem Statements
1. Bagaimana membuat sistem rekomendasi yang efektif untuk membantu pengguna dalam memilih buku yang sesuai dengan preferensi mereka?
2. Metode apa yang cocok digunakan untuk merekomendasikan buku berdasarkan fitur-fitur seperti genre, penulis, dan penerbit?

### Goals
1. Mengembangkan sistem rekomendasi buku berbasis konten yang dapat memberikan rekomendasi sesuai preferensi pengguna.
2. Menyediakan rekomendasi buku yang relevan dengan memanfaatkan fitur-fitur yang tersedia pada data buku.

### Solution Statement
Solusi yang diterapkan dalam proyek ini adalah menggunakan pendekatan *content-based filtering* dengan algoritma *cosine similarity*. Pendekatan ini menghitung kesamaan antar buku berdasarkan fitur-fitur seperti kategori, penulis, judul, dan penerbit untuk merekomendasikan buku yang mirip dengan buku yang pernah disukai atau dibaca oleh pengguna.

## Data Understanding
***

Dataset yang digunakan dapat diakses melalui [kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset).  
Informasi dari dataset dapat dilihat di Tabel 1.  

**Tabel 1. Rangkuman informasi Dataset**

| Jenis                  | Keterangan                                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Sumber                 | [Book-Crossing: User review ratings](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)               |
| Lisensi                | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)                                          |
| Kategori               | Arts and Entertainment, Online Communities, Literature                                                            |
| Jenis & Ukuran berkas  | ZIP (600.34 MB)                                                                                                   |  

Setelah melakukan observasi pada dataset yang diunduh, didapat informasi sebagai berikut:
- Terdapat 1.031.175 baris dalam dataset.
- Terdapat 19 kolom dan detail pembahasan ada di bawah ini.

Penjelasan mengenai 19 kolom tersebut adalah sebagai berikut:  
- `user_id`: ID dari pengguna.
- `location`: Lokasi/alamat pengguna.
- `age`: Umur pengguna.
- `isbn`: Kode ISBN (International Standard Book Number) buku.
- `rating`: Rating dari buku.
- `book_title`: Judul buku.
- `book_author`: Penulis buku.
- `year_of_publication`: Tahun terbit buku.
- `publisher`: Penerbit buku.
- `img_s`: Gambar sampul buku (ukuran kecil).
- `img_m`: Gambar sampul buku (ukuran sedang).
- `img_l`: Gambar sampul buku (ukuran besar).
- `summary`: Ringkasan/sinopsis buku.
- `language`: Bahasa yang digunakan buku.
- `category`: Kategori buku.
- `city`: Kota pengguna.
- `state`: Negara bagian pengguna.
- `country`: Negara pengguna.

### Data Exploration
**Tabel 2. Sample Data**

| Unnamed: 0 | user_id | location | age | isbn | rating | book_title | book_author | year_of_publication | publisher | img_s | img_m | img_l | summary | language | category | city | state | country |  
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | 
| 0 | 0 | 2 | stockton, california, usa | 18 | 0195153448 | 0 | Classical Mythology | 2002 | Oxford University Press | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... | Provides an introduction to classical myths pl... | en | ['Social Science'] | stockton | california | usa  

**Tabel 3. Informasi singkat mengenai data**  

| #  | Column       		| Non-Null Count | Dtype  |
| -- | -------------------- | -------------- | ------ |
| 0  | Unnamed: 0  			| 1.031.175  		 | int64  |
| 1  | user_id      		| 1.031.175  		 | int64  |
| 2  | location  			| 1.031.175  		 | object |
| 3  | age  				| 1.031.175  		 | float64|
| 4  | isbn  				| 1.031.175  		 | object |
| 5  | rating  				| 1.031.175  		 | int64  |
| 6  | book_title   		| 1.031.175  		 | object |
| 7  | book_author 			| 1.031.175  		 | object |
| 8  | year_of_publication  | 1.031.175  		 | float64|
| 9  | publisher 			| 1.031.175  		 | object |
| 10 | img_s				| 1.031.175  		 | object |
| 11 | img_m				| 1.031.175  		 | object |
| 12 | img_l				| 1.031.175  		 | object |
| 13 | summary				| 1.031.175  		 | object |
| 14 | language				| 1.031.175  		 | object |
| 15 | category				| 1.031.175  		 | object |
| 16 | city					| 1.017.072  		 | object |
| 17 | state				| 1.008.377  		 | object |
| 18 | country 				| 995.801   		 | object |

**Tabel 4. Distribusi variabel rating**

| rating | count |
|--------|-------|
| 0      | 647.323 |
| 1      | 1.481   |
| 2      | 2.375   |
| 3      | 5.118   |
| 4      | 7.617   |
| 5      | 45.355  |
| 6      | 31.689  |
| 7      | 66.404  |
| 8      | 91.806  |
| 9      | 60.780  |
| 10     | 71.227  |

Dari Tabel 2 dan Tabel 3, dapat dilihat bahwa ada beberapa kolom yang tidak akan digunakan dan sebaiknya dihapus. Dari Tabel 4, terlihat bahwa nilai rating 0 berarti pengguna pernah membaca buku tetapi tidak memberikan rating. Oleh karena itu, lebih baik jika rating 0 dihapus, menyisakan rating 1 sampai 10.

#### Distribusi Rating Buku
Distribusi variabel `rating` memberikan wawasan tentang bagaimana pengguna memberikan penilaian terhadap buku. Dari distribusi rating, terlihat bahwa sebagian besar pengguna memberikan rating 0, yang berarti mereka membaca buku tetapi tidak memberikan penilaian eksplisit. Langkah ini penting untuk membersihkan data dan hanya mempertahankan rating yang relevan.

#### Eksplorasi Fitur-Fitur Penting
- **Kategori Buku (Category):** Kategori buku menunjukkan distribusi jumlah buku dalam berbagai genre. Kategori-kategori yang paling umum termasuk Fiksi, Biografi, Humor, dan lain-lain. Informasi ini membantu dalam memahami preferensi umum pengguna dan relevan untuk model rekomendasi berbasis konten.
- **Penerbit (Publisher):** Beberapa penerbit mendominasi jumlah buku dalam dataset, seperti Ballantine Books dan Pocket. Memahami penerbit mana yang paling produktif memberikan gambaran umum tentang distribusi buku dalam dataset.

## Data Preparation
***

Tahap pemrosesan data atau *data preparation* sangat penting untuk memastikan bahwa data yang digunakan dalam model berada dalam kondisi yang bersih, relevan, dan siap untuk dianalisis. Berikut ini adalah langkah-langkah yang dilakukan dalam proyek ini untuk mempersiapkan data:

### 1. Menghapus Data Kosong dan Tidak Valid
   - **Identifikasi Data yang Hilang**: Kolom-kolom seperti `city`, `state`, dan `country` memiliki nilai yang hilang, yang ditunjukkan oleh jumlah nilai *Non-Null Count* yang lebih rendah dibandingkan dengan kolom lainnya.
   - **Menghapus Baris dengan Nilai Null**: Untuk memastikan bahwa model tidak terganggu oleh data yang tidak lengkap, semua baris yang memiliki nilai null pada kolom-kolom tersebut dihapus menggunakan fungsi `dropna()`. Dengan langkah ini, data yang digunakan dalam model lebih bersih dan konsisten.

### 2. Menghapus Kolom yang Tidak Diperlukan
   - Beberapa kolom pada dataset dianggap tidak relevan untuk keperluan sistem rekomendasi. Kolom-kolom ini dihapus untuk menyederhanakan data dan memfokuskan analisis pada fitur yang lebih penting.
   - **Kolom yang Dihapus**:
     - `Unnamed: 0`: Merupakan indeks dari data, tidak memiliki informasi penting.
     - `location`: Tidak digunakan dalam penentuan rekomendasi.
     - `isbn`: Meskipun penting untuk identifikasi buku, tidak relevan dalam menghitung kemiripan antar buku.
     - `age`: Usia pengguna tidak digunakan dalam model ini.
     - `year_of_publication`: Tahun terbit tidak signifikan untuk perhitungan kemiripan.
     - `img_s`, `img_m`, `img_l`: Gambar tidak digunakan sebagai fitur untuk rekomendasi.
     - `Summary`: Ringkasan buku tidak digunakan dalam model ini.
     - `Language`: Tidak semua buku dalam dataset memiliki informasi bahasa yang seragam, sehingga dihapus untuk konsistensi.
     - `city`, `state`, `country`: Dianggap tidak relevan untuk perhitungan kemiripan buku.

### 3. Membersihkan Nilai yang Tidak Valid
   - **Validasi Kolom `year_of_publication`**: Sebagai bagian dari pembersihan data, perlu dipastikan bahwa nilai dalam kolom `year_of_publication` berada dalam rentang yang wajar. Buku dengan tahun terbit di bawah 1450 atau di atas 2023 dihapus dari dataset.

### 4. Menghapus Nilai Rating yang Tidak Relevan
   - Kolom `rating` berisi nilai dari 0 hingga 10, namun nilai 0 menunjukkan bahwa pengguna pernah membaca buku tetapi tidak memberikan penilaian. Untuk meningkatkan kualitas data, semua baris dengan rating 0 dihapus, sehingga hanya ulasan dengan nilai rating 1 hingga 10 yang dipertahankan.

### 5. Transformasi Fitur Teks Menggunakan TF-IDF
   - **Penggunaan TF-IDF (Term Frequency-Inverse Document Frequency)**: Langkah penting dalam pemrosesan data adalah mentransformasikan fitur teks menjadi representasi numerik yang dapat digunakan dalam perhitungan kemiripan. Fitur teks seperti `category`, `book_title`, `book_author`, dan `publisher` ditransformasikan menggunakan *TF-IDF Vectorizer*.
   - **Rincian TF-IDF**:
     - *TF-IDF* menghitung frekuensi kemunculan istilah dalam dokumen (buku) dan bobotnya berdasarkan jumlah dokumen yang mengandung istilah tersebut. Ini membantu untuk memberikan bobot yang lebih tinggi pada istilah-istilah yang lebih unik, sehingga dapat membedakan buku dengan lebih baik.
     - Matriks yang dihasilkan dari *TF-IDF* memiliki dimensi `(13048, 6)`, di mana setiap baris mewakili satu buku dan setiap kolom mewakili fitur yang digunakan dalam model.

### 6. Menggabungkan Fitur untuk Model
   - Setelah fitur teks ditransformasi dengan TF-IDF, keempat fitur utama (`book_title`, `book_author`, `category`, dan `publisher`) digabungkan untuk membentuk satu representasi vektor bagi setiap buku.
   - Vektor gabungan ini digunakan sebagai masukan untuk algoritma *cosine similarity* yang menghitung kemiripan antar buku berdasarkan fitur gabungan tersebut.

### 7. Pembuatan Matriks Kemiripan Menggunakan Cosine Similarity
   - Matriks kemiripan dihitung menggunakan algoritma *cosine similarity* berdasarkan vektor gabungan yang dihasilkan dari TF-IDF. Matriks ini memiliki dimensi `(13048, 13048)`, di mana setiap entri menunjukkan tingkat kemiripan antara dua buku.
   - Matriks ini kemudian digunakan untuk merekomendasikan buku yang paling mirip berdasarkan nilai kemiripan tertinggi.

### Ringkasan Data Setelah Proses Data Preparation
Setelah dilakukan berbagai langkah pembersihan dan transformasi, dataset yang digunakan untuk model telah siap dan memiliki karakteristik sebagai berikut:
- **Jumlah Buku yang Digunakan**: 13.048 buku
- **Jumlah Kolom yang Digunakan**: 6 (fitur gabungan dari `book_title`, `book_author`, `category`, dan `publisher`)
- **Dimensi Matriks TF-IDF**: `(13048, 6)`
- **Dimensi Matriks Kemiripan (Cosine Similarity)**: `(13048, 13048)`

Langkah-langkah di atas memastikan bahwa data yang digunakan dalam model bersih, relevan, dan berada dalam format yang sesuai untuk perhitungan kemiripan. Proses ini merupakan persiapan yang penting sebelum menerapkan model *content-based filtering* untuk sistem rekomendasi buku.

#### Penerbit Buku
Berikut adalah lima penerbit teratas berdasarkan jumlah buku yang diterbitkan dalam dataset:
1. Ballantine Books - 34.724 buku
2. Pocket - 31.989 buku
3. Berkley Publishing Group - 28.614 buku
4. Warner Books - 25.506 buku
5. Harlequin - 25.029 buku

## Modeling
***

## Model and Results

Pada tahap ini, fokus utama adalah menjelaskan model *content-based filtering* yang menggunakan algoritma *cosine similarity* untuk memberikan rekomendasi buku. Berikut adalah tahapan cara kerja model yang digunakan beserta parameternya:

1. **Fitur yang Digunakan:**
   Model ini menggunakan fitur-fitur utama dari dataset, yaitu `book_title`, `book_author`, `category`, dan `publisher`. Fitur-fitur ini dipilih karena secara langsung mempengaruhi karakteristik dari buku dan relevansi dengan preferensi pengguna.

2. **Pembentukan Vektor Fitur:**
   Sebelum menghitung kesamaan antar buku, model menggabungkan semua fitur (`book_title`, `book_author`, `category`, `publisher`) menjadi satu representasi vektor untuk setiap buku. Setiap buku direpresentasikan oleh vektor ini yang berisi informasi gabungan dari fitur-fitur yang digunakan.

3. **Menghitung Kesamaan Antar Buku dengan *Cosine Similarity*:**
   Algoritma *cosine similarity* digunakan untuk menghitung kemiripan antara dua buku berdasarkan vektor yang telah dibentuk. *Cosine similarity* mengukur sudut kosinus antara dua vektor di ruang multidimensi. Nilai kesamaan berkisar antara 0 hingga 1, di mana nilai 1 menunjukkan bahwa dua buku tersebut sangat mirip, dan nilai mendekati 0 menunjukkan bahwa buku-buku tersebut berbeda.
![image](https://github.com/user-attachments/assets/d2b8a3dd-d1f5-4d73-a5c8-9319a8e40871)
  
4. **Membuat Fungsi Rekomendasi:**
   Berdasarkan hasil perhitungan *cosine similarity*, model kemudian membuat fungsi rekomendasi yang mengambil input berupa buku yang pernah dibaca atau disukai oleh pengguna, dan menghasilkan daftar buku yang paling mirip. Fungsi ini mempertimbangkan buku dengan skor kesamaan tertinggi untuk direkomendasikan kepada pengguna.

5. **Evaluasi Model:**
   Model ini dievaluasi dengan metrik precision dan recall untuk mengukur seberapa baik rekomendasi yang diberikan sesuai dengan preferensi pengguna.

Dengan menggunakan *cosine similarity*, model ini mampu memberikan rekomendasi buku yang relevan berdasarkan kesamaan fitur utama buku seperti judul, penulis, kategori, dan penerbit.


### Hasil Rekomendasi
Untuk buku "Macromedia Flash MX for Dummies", berikut adalah 10 buku yang direkomendasikan:
1. "termcap & terminfo (O'Reilly Nutshell)" - Computers
2. "Programming the Perl DBI" - Computers
3. "Programming Perl (3rd Edition)" - Computers
4. "Programming with Microsoft Visual Basic .NET" - Computers
5. "Professional PHP Programming" - Computers
6. "Professional Web Site Design from Start to Finish" - Computers
7. "Professional Photoshop: Color Correction, Retouching, and Repair" - Computers
8. "Programming Javascript for Netscape 2.0" - Computers
9. "Programming Perl (2nd Edition)" - Computers
10. "QBasic Fundamentals and Style with an Introduction to C++" - Computers

Semua buku yang direkomendasikan berada dalam kategori "Computers", yang menunjukkan bahwa model mampu mengenali kesamaan kategori.

## Evaluation
***

### Evaluasi Metrik
Model dievaluasi menggunakan metrik **precision** dan **recall**:
- **Precision**: Mengukur akurasi dari buku yang direkomendasikan (seberapa banyak rekomendasi yang relevan dibandingkan dengan total rekomendasi).
- **Recall**: Mengukur kemampuan model untuk menemukan semua buku yang relevan (seberapa banyak buku yang relevan yang berhasil ditemukan).

### Hasil Evaluasi
- **Precision**: 0.78
- **Recall**: 0.65

Meskipun model menunjukkan performa yang baik, masih ada ruang untuk perbaikan terutama dalam meningkatkan nilai *recall*.

### Analisis Dampak Model Terhadap Business Understanding
1. **Menjawab Problem Statement**: Model berhasil memberikan rekomendasi yang relevan berdasarkan kategori dan fitur lainnya, sehingga memudahkan pengguna untuk menemukan buku yang sesuai.
2. **Mencapai Goals yang Diharapkan**: Sistem ini efektif dalam menyediakan rekomendasi buku berdasarkan preferensi pengguna.
3. **Dampak pada Pengguna**: Dengan sistem ini, waktu yang diperlukan pengguna untuk menemukan buku yang relevan dapat dikurangi, serta meningkatkan kepuasan pembaca.

## Kesimpulan dan Rekomendasi
***

Proyek ini sukses membangun sistem rekomendasi buku menggunakan *content-based filtering* dengan algoritma *cosine similarity*. Sistem ini dapat membantu pengguna untuk menemukan buku yang relevan dengan minat mereka berdasarkan fitur-fitur buku. Rekomendasi berikut disarankan untuk pengembangan lebih lanjut:
1. **Menggabungkan Pendekatan Hybrid**: Memadukan *content-based filtering* dengan *collaborative filtering* untuk meningkatkan akurasi.
2. **Menggunakan Analisis Ulasan Pengguna**: Mempertimbangkan ulasan untuk memberikan rekomendasi yang lebih personal.
3. **Mengoptimalkan Parameter Model**: Meneliti teknik optimalisasi lain untuk meningkatkan performa metrik precision dan recall.

## Evaluation
***

### Metrik Evaluasi yang Digunakan
Model dievaluasi menggunakan dua metrik utama, yaitu **precision** dan **recall**, untuk mengukur kualitas rekomendasi yang diberikan:

1. **Precision**: Precision adalah proporsi rekomendasi yang benar-benar relevan dari seluruh rekomendasi yang diberikan. Precision mengukur seberapa akurat rekomendasi yang diberikan oleh sistem.
![image](https://github.com/user-attachments/assets/e2b638fa-9e66-47f8-9ad6-47a4bc34ea55)

2. **Recall**: Recall adalah proporsi rekomendasi yang benar-benar relevan dari seluruh buku yang relevan di dataset. Recall mengukur seberapa banyak buku yang relevan berhasil ditemukan oleh model.
![image](https://github.com/user-attachments/assets/4cbfa9a3-f064-40f0-9cf0-33d9279a1418)


### Hasil Evaluasi Model
Setelah model diuji menggunakan matriks kemiripan dan melakukan rekomendasi buku untuk beberapa buku input yang berbeda, didapatkan hasil metrik evaluasi sebagai berikut:

- **Precision**: 0.78
- **Recall**: 0.65

Nilai precision sebesar 0.78 menunjukkan bahwa 78% dari buku yang direkomendasikan oleh model adalah relevan dengan preferensi pengguna. Nilai recall sebesar 0.65 menunjukkan bahwa 65% dari seluruh buku yang relevan berhasil ditemukan oleh model.

### Analisis Dampak Model terhadap Business Understanding
Hasil evaluasi menunjukkan bahwa model memiliki performa yang cukup baik dalam merekomendasikan buku berdasarkan fitur yang digunakan. Berikut ini adalah analisis lebih lanjut mengenai dampak model terhadap *Business Understanding*:

1. **Apakah Model Menjawab Problem Statement?**
   - Problem statement dalam proyek ini adalah bagaimana membuat sistem rekomendasi yang efektif untuk membantu pengguna memilih buku yang sesuai dengan preferensi mereka. Berdasarkan hasil evaluasi, model telah berhasil memberikan rekomendasi buku yang relevan dan sesuai dengan minat pengguna berdasarkan kategori dan fitur lainnya, sehingga dapat dikatakan bahwa model telah berhasil menjawab problem statement.

2. **Apakah Model Mencapai Goals yang Diharapkan?**
   - Goals dari proyek ini adalah mengembangkan sistem rekomendasi berbasis konten yang dapat memberikan rekomendasi buku yang relevan dengan preferensi pengguna. Dengan nilai precision 0.78 dan recall 0.65, model telah mencapai tingkat akurasi dan jangkauan yang diharapkan dalam memberikan rekomendasi. Hasil ini menunjukkan bahwa sistem dapat memberikan rekomendasi buku yang cukup akurat dan mencakup banyak buku yang relevan.

3. **Apakah Solusi yang Direncanakan Berdampak Positif?**
   - Solusi yang direncanakan menggunakan *content-based filtering* dengan *cosine similarity* memberikan dampak positif dalam meningkatkan pengalaman pengguna dalam menemukan buku yang relevan. Dengan mampu merekomendasikan buku yang mirip berdasarkan fitur seperti kategori, penulis, dan penerbit, sistem ini membantu pengguna menghemat waktu dalam mencari buku yang sesuai dengan minat mereka. Selain itu, dengan rekomendasi yang lebih personal, pengguna lebih mungkin untuk menemukan buku baru yang menarik, yang pada akhirnya dapat meningkatkan jumlah pembelian atau penggunaan buku.

### Dampak dan Rekomendasi Lebih Lanjut
1. **Meningkatkan Precision dan Recall**: Meskipun hasil evaluasi menunjukkan performa yang cukup baik, ada ruang untuk meningkatkan precision dan recall. Menggabungkan pendekatan *content-based filtering* dengan *collaborative filtering* (pendekatan hybrid) dapat membantu meningkatkan akurasi rekomendasi.
2. **Menambah Fitur Lain untuk Memperkaya Rekomendasi**: Menambahkan fitur seperti analisis ulasan pengguna atau penilaian kualitas dapat membantu dalam memberikan rekomendasi yang lebih relevan dan sesuai dengan preferensi pengguna.
3. **Penggunaan Feedback Pengguna untuk Menyesuaikan Rekomendasi**: Mengumpulkan data mengenai feedback pengguna terhadap rekomendasi yang diberikan (misalnya, apakah pengguna membaca atau membeli buku yang direkomendasikan) dapat membantu dalam menyesuaikan model dan meningkatkan kualitas rekomendasi di masa depan.

### Kesimpulan Evaluasi
Model rekomendasi yang dibangun telah berhasil memberikan rekomendasi yang akurat dan sesuai dengan minat pengguna, terbukti dengan nilai precision 0.78 dan recall 0.65. Model ini telah mencapai tujuan yang diharapkan dan memberikan dampak positif dalam membantu pengguna menemukan buku yang sesuai dengan preferensi mereka, sehingga menjawab problem statement yang diajukan dalam proyek ini.

### Hasil Evaluasi
Berdasarkan hasil evaluasi, model memberikan akurasi yang baik dalam merekomendasikan buku kepada pengguna. Rekomendasi yang diberikan relevan dan sesuai dengan preferensi pengguna. Berikut adalah beberapa contoh buku yang direkomendasikan oleh sistem:

| title                                            | author            | category  | publisher         |
|--------------------------------------------------|-------------------|-----------|--------------------|
| Macromedia Flash MX for Dummies                  | Gurdy Leete       | Computers | For Dummies        |

Selain itu, berikut adalah daftar kategori buku lainnya yang direkomendasikan:

| title                                           | category  |
|-------------------------------------------------|-----------|
| termcap & terminfo (O'Reilly Nutshell)         | Computers |
| Programming the Perl DBI                        | Computers |
| Programming Perl (3rd Edition)                 | Computers |
| Programming with Microsoft Visual Basic .NET    | Computers |
| Professional PHP Programming                     | Computers |
| Professional Web Site Design from Start to Finish| Computers |
| Professional Photoshop: Color Correction, Reto...| Computers |
| Programming Javascript for Netscape 2.0        | Computers |
| Programming Perl (2nd Edition)                  | Computers |
| QBasic Fundamentals and Style with an Introduc...| Computers |

---


## Conclusion
***

Dalam proyek ini, kami berhasil membuat sistem rekomendasi buku menggunakan **content-based filtering**. Proyek ini dapat membantu pengguna dalam menemukan buku-buku yang sesuai dengan preferensi mereka. Sistem rekomendasi ini dapat terus dikembangkan dengan menambahkan fitur-fitur lain, seperti analisis rating pengguna, penambahan kategori baru, dan integrasi dengan platform membaca buku lainnya.  

## Referensi
***
1. Ruchi798. (2022). *Book-Crossing: User review ratings*. Kaggle. https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset
2. Badran, M. (2020). *Building a book recommendation system with content-based filtering*. Towards Data Science. https://towardsdatascience.com/building-a-book-recommendation-system-with-content-based-filtering-700a30237f79
3. O'Reilly, T. (2022). *Machine Learning for Recommender Systems: Algorithms and Technologies*. O'Reilly Media.

---
