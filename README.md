# Laporan Proyek Machine Learning - Khamdan Annas Fakhryza
## _Proyek Pertama_

# Domain Proyek
### Latar Belakang
Penyakit jantung merupakan salah satu penyebab utama kematian di seluruh dunia. Setiap tahunnya, jutaan orang mengalami gangguan kesehatan serius yang berkaitan dengan jantung, seperti serangan jantung dan gagal jantung. Di Indonesia, penyakit jantung menjadi salah satu penyebab kematian tertinggi, terutama pada kelompok usia lanjut, yang menunjukkan urgensi deteksi dini untuk faktor-faktor risiko penyakit jantung.

Faktor risiko seperti usia, tekanan darah, kadar kolesterol, kebiasaan merokok, indeks massa tubuh (BMI), dan riwayat diabetes, memiliki peran penting dalam memprediksi potensi terjadinya penyakit jantung. Mengingat banyaknya faktor risiko yang harus dipertimbangkan, memprediksi penyakit jantung secara manual menjadi tugas yang rumit dan rentan terhadap kesalahan.

Dengan kemajuan teknologi dalam kecerdasan buatan (AI) dan pembelajaran mesin (ML), terdapat peluang untuk meningkatkan kemampuan prediksi risiko penyakit jantung secara lebih cepat dan akurat. Penggunaan model machine learning dapat membantu tenaga medis dalam memberikan diagnosis yang lebih baik berdasarkan data kesehatan pasien, yang pada akhirnya dapat meningkatkan upaya pencegahan dan pengobatan.

Dalam proyek ini, dikembangkan sistem prediksi risiko penyakit jantung berbasis machine learning menggunakan berbagai model seperti K-Nearest Neighbor (KNN), Random Forest, Gradient Boosting, dan Support Vector Machine (SVM). Dengan memanfaatkan data medis seperti tekanan darah, kadar kolesterol, dan riwayat kesehatan pasien, sistem ini diharapkan dapat membantu mengidentifikasi risiko penyakit jantung secara lebih efektif, sehingga dapat mengurangi angka kematian dan meningkatkan kualitas hidup pasien melalui deteksi dan intervensi dini.

# Business Understanding
### Problem Statements
- Pernyataan Masalah 1:
  Meningkatnya prevalensi penyakit jantung menjadikannya salah satu penyebab utama kematian di seluruh dunia. Penting untuk mengidentifikasi individu yang berisiko tinggi secara dini untuk pencegahan yang efektif.
- Pernyataan Masalah 2:
  Tingginya variasi dalam data kesehatan pasien seringkali menyebabkan kesalahan dalam prediksi risiko penyakit jantung. Oleh karena itu, diperlukan pendekatan berbasis machine learning untuk meningkatkan akurasi prediksi risiko.
- Pernyataan Masalah 3:
  Diperlukan evaluasi performa berbagai model machine learning, seperti KNN, Random Forest, Adaboost, dan SVM, untuk menemukan model yang paling optimal dalam memprediksi risiko penyakit jantung dengan akurat.

### Goals
- Jawaban Pernyataan Masalah 1:
  Mengembangkan model prediksi risiko penyakit jantung untuk membantu pencegahan dini serta mendukung penanganan yang lebih efektif berdasarkan data kesehatan pasien.
- Jawaban Pernyataan Masalah 2:
  Menerapkan teknik machine learning untuk menangani data kesehatan pasien yang bervariasi, dengan tujuan meningkatkan akurasi prediksi risiko penyakit jantung.
- Jawaban Pernyataan Masalah 3:
  Membandingkan performa algoritma KNN, Random Forest, Adaboost, dan SVM, untuk menemukan model terbaik yang memberikan prediksi risiko penyakit jantung paling akurat dan andal.
  
### Solution Statements
- Solution 1: 
Menggunakan Beberapa Algoritma untuk Membuat Model Prediksi Untuk mencapai prediksi risiko stroke yang akurat, solusi pertama adalah menggunakan dan membandingkan empat algoritma berbeda: K-Nearest Neighbors (KNN), Random Forest, AdaBoost, dan Support Vector Machine (SVM). Masing-masing algoritma memiliki pendekatan unik untuk memprediksi hasil berdasarkan fitur kesehatan, sehingga memungkinkan untuk menemukan model terbaik yang dapat menghasilkan prediksi akurat.
    - KNN: Algoritma sederhana yang efektif untuk masalah klasifikasi, cocok untuk dataset yang memiliki hubungan non-linear.
    - Random Forest: Algoritma berbasis pohon keputusan yang kuat untuk mengatasi overfitting dan bekerja baik dengan dataset yang memiliki banyak fitur.
    - AdaBoost: Metode boosting yang memperkuat model lemah menjadi model yang lebih kuat, ideal untuk menangani data yang tidak seimbang.
    - SVM: Model klasifikasi yang bekerja baik dalam ruang dimensi tinggi dan digunakan untuk dataset yang kompleks dengan kelas yang dapat dipisahkan.

- Solution 2: 
Menggunakan _Mean squared error_ masing-masing algoritma pada data train dan test

# Data Understanding
Dataset ini dapat diakses menggunakan [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

- Jumlah Data:Dataset yang digunakan terdiri dari 588 baris dan 12 kolom setelah proses pembersihan dan penghapusan outlier.
- Kondisi Data:Data yang digunakan tidak memiliki nilai yang hilang (missing values) setelah proses pembersihan. Semua kolom memiliki tipe data yang sesuai dan tidak ada nilai yang ekstrem (outlier) yang dapat mempengaruhi analisis.
- Tautan Sumber Data:Data diambil dari file ZIP yang berjudul "Heart Failure Prediction.zip", yang berisi file CSV bernama "heart.csv". Tautan sumber data dapat disediakan jika tersedia secara publik.
- Uraian Seluruh Fitur pada Data:
1. Age: Usia pasien (numerik).
2. Sex: Jenis kelamin pasien (kategorikal).
3. ChestPainType: Tipe nyeri dada (kategorikal).
4. RestingBP: Tekanan darah istirahat (numerik).
5. Cholesterol: Kadar kolesterol (numerik).
6. FastingBS: Kadar glukosa puasa (numerik).
7. RestingECG: Hasil EKG istirahat (kategorikal).
8. MaxHR: Denyut jantung maksimum (numerik).
9. ExerciseAngina: Apakah ada angina saat berolahraga (kategorikal).
10. Oldpeak: Nilai ST depresiasi (numerik).
11. ST_Slope: Kemiringan ST (kategorikal).
12. HeartDisease: Target variabel (0 atau 1, menunjukkan ada tidaknya penyakit jantung).

### Data Loading
Data_Jantung = pd.read_csv perintah ini digunakan untuk membaca data dengan format csv. kemudian ditampikan dengan memanggil kelas Data_Jantung, juga bisa menggunakan Data_Jantung.head() jika hanya ingin menampilkan 5 baris data

![Screenshot 2024-10-15 001931](https://github.com/user-attachments/assets/60592abd-bb34-453c-8e2e-1657b3557dcc)

### Menampilkan informasi dari dataset

Pada Fungsi info() di pandas yang digunakan untuk menampilkan informasi dari dataset seperti jenis data tipe datanya

![Screenshot 2024-10-15 002725](https://github.com/user-attachments/assets/d0cd8d6b-e982-403b-8b6a-86fe633d6a57)

dari output pada gambar, dapat dilihat bahwa :
* Terdapat 5 kolom dengan tipe objek yaitu : Sex, ChestPainType, RestingECG, ExerciseAngina dan ST_Slope. kolom ini merupakan  categorical features (fitur non-numerik)
* Terdapat 6 kolom bertipe numerik dengan tipe data int64 yaitu Age, RestingBP, Cholesterol, FastingBS, MaxHR dan HeartDisease.
* Terdapat 1 kolom bertipe numerik dengan tipe data float64 yaitu OldPeak.

![Screenshot 2024-10-15 002913](https://github.com/user-attachments/assets/0bccff20-d548-4259-89ea-f1cf509c755e)

Pada fungsi describe() yang berfungsi untuk menampilkan statistik dari dataset dan deskripsi pada dataset
Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:
* Count  adalah jumlah sampel pada data.
* Mean adalah nilai rata-rata 
* Std adalah standar deviasi
* Min yaitu nilai minimum setiap kolom
* 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
* 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
* 75% adalah kuartil ketiga
* Max adalah nilai maksimum.

### Mencari missing value
Pada proyek ini  digunakan fungsi isnull().sum() yang berfungsi untuk menemukan nilai _missing value_ di masing  masing kolom dataset. _missing value_ sendiri dapat diartikan sebagai nilai atribut yang kosong pada objek data.

![image](https://github.com/user-attachments/assets/0c6fb89a-94b2-4bb9-af5b-8754c8fc7f7b)

Berdasarkan output pada gambar dapat dilihat bahwa tidak ditemukan _missing value_ pada masing masing kolom di dataset 

### Visualisasi Data
Visualisai ini digunakan untuk melihat apakah ada data yang terdapat indikasi outlier atau tidak

![image](https://github.com/user-attachments/assets/100a161b-9a67-467d-a7d9-5a85b6673b42)

menggunakan perintah berikut:
```sh
df_outlier=data_jantung.select_dtypes(exclude=['object'])
for column in df_outlier:
        plt.figure()
        sns.boxplot(data=df_outlier, x=column)
```

Di dalam gamabr, pada beberapa fitur numerik yang terdapat outliers. Untuk mengatasi outliers tersebut saya menggunakan metode IQR. Saya akan menggunakan metode IQR untuk mengidentifikasi outlier yang berada di luar Q1 dan Q3. Nilai apa pun yang berada di luar batas ini dianggap sebagai outlier. 

# Data Preparation
### Tahap Preparation :
- Pengolahan Fitur Kategorikal: Fitur kategorikal seperti jenis kelamin (Sex), jenis nyeri dada (ChestPainType), EKG saat istirahat (RestingECG), angina saat berolahraga (ExerciseAngina), dan kemiringan ST (ST_Slope) tidak dapat langsung diproses oleh algoritma machine learning yang memerlukan input numerik. Oleh karena itu, fitur-fitur ini dikonversi menjadi format numerik melalui teknik One-Hot Encoding, yang akan menghasilkan kolom biner untuk setiap kategori, sehingga model dapat mengenali perbedaan antar kategori tersebut.
- Penanganan Outlier: Untuk fitur-fitur numerik, outlier ditangani dengan menggunakan Metode Rentang Interkuartil (IQR). Metode ini membantu menentukan batas bawah dan batas atas dari data yang dianggap normal, berdasarkan perhitungan statistik.
- Reduksi Dimensi Menggunakan PCA: Fitur numerik yang memiliki korelasi tinggi atau informasi yang redundant dapat membuat model menjadi lebih rumit tanpa meningkatkan kinerja. Oleh karena itu, analisis komponen utama (PCA) diterapkan untuk mengurangi jumlah fitur sambil tetap menjaga variasi data. Ini membantu menyederhanakan model tanpa kehilangan informasi penting.
- Normalisasi Fitur Numerik: Setelah menangani outlier, fitur numerik dinormalisasi menggunakan StandardScaler untuk memastikan bahwa setiap fitur memiliki rata-rata 0 dan deviasi standar 1. Proses ini penting untuk menghindari bias dalam model yang disebabkan oleh perbedaan skala antar fitur.
Untuk melakukan proses encoding fitur kategori, salah satu teknik yang umum dilakukan adalah teknik one-hot-encoding.

###Tahap Split Data:
Dataset dibagi menjadi data pelatihan dan data pengujian dengan proporsi 90:10 menggunakan fungsi train_test_split dari sklearn.model_selection. Data pelatihan digunakan untuk melatih model, sementara data pengujian digunakan untuk mengevaluasi kinerja model.

###Urutan Tahapan:
Pastikan untuk menjelaskan setiap tahap secara runtut sesuai dengan notebook:
- Pembersihan Data: Menghapus nilai hilang dan outlier.
- Encoding: Melakukan One-Hot Encoding pada fitur kategorikal.
- Standarisasi: Menerapkan StandardScaler pada fitur numerik di data pelatihan dan pengujian.
- Split Data: Memisahkan dataset menjadi data pelatihan dan pengujian

**Menangani Outlier**
```sh
# Select only numerical features for outlier removal
numerical_features = data_jantung.select_dtypes(include=['number']).columns

Q1 = data_jantung[numerical_features].quantile(0.25)
Q3 = data_jantung[numerical_features].quantile(0.75)
IQR = Q3 - Q1

# Apply outlier removal logic only to numerical features
data_jantung = data_jantung[~((data_jantung[numerical_features] < (Q1 - 1.5 * IQR)) | (data_jantung[numerical_features] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Cek ukuran dataset setelah kita drop outliers
data_jantung.shape
```

Pada proyek ini digunakan IQR method  untuk menangani _outliers_ yang pertama definisikan dulu _outliers_ pada Q1 atau batas bawah yaitu 0.25 lalu _outliers_ pada Q3 atau batas atas  yaitu 0.75 
kemudian hasil Q3 akan dikurangi dengan hasil Q1 setelah itu buat variabel baru yaitu _data_jantung_  untuk menampung hasil batas bawah dari pengurangan Q1 dengan 1,5 * IQR. lalu untuk hasil batas atas digunakan penambahan 1.5 * IQR dengan Q3

dan jika dibuat persamaan dapat dilihat sebagai berikut :
***
_Batas bawah = Q1 - 1.5 * IQR_
_Batas atas = Q3 + 1.5 * IQR_
***
setelah itu untuk melihat ukuran hasil penanganan _outliers_, variabel _insurance_ dapat dipanggil dengan __data_jantung.shape__ lalu akan ditampilkan hasil penanganan _outliers_

![image](https://github.com/user-attachments/assets/9050ddcc-c2e4-4cf5-86c9-50bd09724b73)

**Korelasi Matrik**

![Screenshot 2024-10-14 130304](https://github.com/user-attachments/assets/aebf273d-4458-4eae-af34-ea1628efe020)

Matriks korelasi yang ditampilkan membantu dalam memahami seberapa kuat hubungan antara fitur numerik yang digunakan dalam model prediksi. Beberapa poin penting yang bisa dimasukkan dalam evaluasi adalah:

- Korelasi Antara Fitur: Berdasarkan analisis matriks, terlihat bahwa fitur Age (usia) menunjukkan korelasi positif terhadap Oldpeak (0.3) dan korelasi negatif terhadap MaxHR (-0.38). Hal ini mengindikasikan bahwa dengan bertambahnya usia, nilai Oldpeak cenderung meningkat sementara nilai MaxHR mengalami penurunan. Temuan ini bisa menjadi indikator penting dalam memperkirakan risiko penyakit jantung. Di sisi lain, RestingBP (tekanan darah saat istirahat) menunjukkan korelasi yang rendah dengan fitur lainnya, mengindikasikan bahwa meskipun mungkin tidak memiliki dampak signifikan terhadap fitur numerik lain, tekanan darah ini tetap dapat relevan dalam memprediksi hasil akhir.
- Korelasi yang Minimal: Beberapa fitur seperti Cholesterol dan RestingBP menunjukkan korelasi yang sangat lemah dengan fitur lainnya, contohnya, korelasi antara Cholesterol dan MaxHR mendekati angka 0. Hal ini mengisyaratkan bahwa fitur-fitur tersebut mungkin tidak memiliki pengaruh besar terhadap prediksi langsung. Namun, masih perlu dipertimbangkan relevansinya untuk tetap ada dalam model atau kemungkinan dihapus selama proses seleksi fitur.
Penghapusan Fitur yang Tidak Relevan: Saya telah menghapus fitur FastingBS dan MaxHR dari dataset, yang mungkin disebabkan oleh rendahnya korelasi atau ketidaksignifikanannya terhadap fitur lain. Langkah ini merupakan bagian dari seleksi fitur yang tepat untuk meningkatkan performa model dengan cara mengurangi kompleksitas tanpa mengorbankan akurasi.

![image](https://github.com/user-attachments/assets/eac4ceb4-1842-49cd-a02f-19d28164325d)

**One Hot Encoding**

```sh
from sklearn.preprocessing import OneHotEncoder
# Melakukan One-Hot Encoding pada kolom 'Sex'
data_jantung = pd.concat([data_jantung, pd.get_dummies(data_jantung['Sex'], prefix='Sex')], axis=1)

# Melakukan One-Hot Encoding pada kolom 'ChestPainType'
data_jantung = pd.concat([data_jantung, pd.get_dummies(data_jantung['ChestPainType'], prefix='ChestPainType')], axis=1)

# Melakukan One-Hot Encoding pada kolom 'RestingECG'
data_jantung = pd.concat([data_jantung, pd.get_dummies(data_jantung['RestingECG'], prefix='RestingECG')], axis=1)

# Melakukan One-Hot Encoding pada kolom 'ExerciseAngina'
data_jantung = pd.concat([data_jantung, pd.get_dummies(data_jantung['ExerciseAngina'], prefix='ExerciseAngina')], axis=1)

# Melakukan One-Hot Encoding pada kolom 'ST_Slope'
data_jantung = pd.concat([data_jantung, pd.get_dummies(data_jantung['ST_Slope'], prefix='ST_Slope')], axis=1)

# Menghapus kolom asli yang sudah di-encode
data_jantung.drop(['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], axis=1, inplace=True)

# Melihat data setelah encoding
data_jantung.head()
```

| Age | RestingBP | Cholesterol | Oldpeak | HeartDisease | Sex_F | Sex_M | ChestPainType_ASY  | ChestPainType_ATA  | ChestPainType_NAP  | ChestPainType_TA | RestingECG_LVH | RestingECG_Normal | RestingECG_ST | ExerciseAngina_N  | ExerciseAngina_Y | ST_Slope_Down  | ST_Slope_Flat  | ST_Slope_Up |
|-----|-----------|-------------|---------|--------------|-------|-------|--------------------|--------------------|--------------------|------------------|----------------|-------------------|---------------|-------------------|------------------|----------------|----------------|-------------|
|  40 |       140 |         289 |     0.0 |            0 | False |  True |              False |               True |              False |            False |          False |             False |          True |            False  |             True |          False |          False |        True |
|  49 |       160 |         180 |     1.0 |            1 |  True | False |              False |              False |               True |            False |          False |             False |          True |            False  |             True |          False |           True |       False |
|  37 |       130 |         283 |     0.0 |            0 | False |  True |              False |               True |              False |            False |          False |              True |         False |             True  |            False |          False |          False |        True |
|  48 |       138 |         214 |     1.5 |            1 |  True | False |               True |              False |              False |            False |          False |             False |          True |            False  |             True |           True |          False |       False |
|  54 |       150 |         195 |     0.0 |            0 | False |  True |              False |              False |               True |            False |          False |              True |         False |             True  |            False |          False |          False |        True |

### Reduksi Dimensi dengan PCA

PCA (Principal Component Analysis) adalah metode reduksi dimensi yang bertujuan untuk merangkum informasi dari suatu dataset dengan mentransformasi data ke dalam format baru yang dikenal sebagai komponen utama. Metode ini bertujuan untuk menyederhanakan kompleksitas data sambil menjaga sebanyak mungkin varians dari data asli. Dalam analisis data jantung, penerapan PCA sangat bermanfaat untuk mendeteksi pola serta mengurangi jumlah fitur yang digunakan tanpa mengorbankan informasi penting yang ada,

```sh
sns.pairplot(data_jantung[['Age', 'RestingBP', 'Cholesterol']], plot_kws={"s": 3});
```

![download (4)](https://github.com/user-attachments/assets/c90b2b46-e36d-4898-9af4-f334c4083576)


### TEKNIK PCA
PCA bekerja menggunakan metode aljabar linier. Ia mengasumsikan bahwa sekumpulan data pada arah dengan varians terbesar merupakan yang paling penting (utama). PCA umumnya digunakan ketika variabel dalam data memiliki korelasi yang tinggi. Korelasi tinggi ini menunjukkan data yang berulang atau redundant. 

Berikut penjelasan untuk masing-masing komponen utama (PC):
* PC pertama mewakili arah varians maksimum dalam data. Ia paling banyak menangkap informasi dari semua fitur dalam data. 
* PC kedua menangkap sebagian besar informasi yang tersisa setelah PC pertama. 
* PC ketiga menangkap sebagian besar informasi yang tersisa setelah PC pertama, PC kedua, dst.

pertama cek menggunakan fungsi airplot 

![download (1)](https://github.com/user-attachments/assets/29490898-68c8-4dac-9f40-5baaf2859ff8)

aplikasikan class PCA dari library scikit learn 

```sh
from sklearn.decomposition import PCA

pca = PCA(n_components=3, random_state=123)
pca.fit(data_jantung[['RestingBP', 'Cholesterol', 'Oldpeak']])
princ_comp = pca.transform(data_jantung[['RestingBP', 'Cholesterol', 'Oldpeak']])
```

proporsi informasi dari ketiga komponen yang dimasukkan

![image](https://github.com/user-attachments/assets/0cbca08d-8dc3-4daa-8ea6-852f54fda091)

Membuat fitur baru :
* Gunakan n_component = 1, karena kali ini, jumlah komponen kita hanya satu.
* Fit model dengan data masukan.
* Tambahkan fitur baru ke dataset dengan nama 'dimension' dan lakukan proses transformasi.
* Drop kolom yang dimasukkan.

```sh
from sklearn.decomposition import PCA
pca = PCA(n_components=1, random_state=123)
pca.fit(data_jantung[['RestingBP', 'Cholesterol', 'Oldpeak']])
data_jantung['dimension'] = pca.transform(data_jantung.loc[:, ('RestingBP', 'Cholesterol', 'Oldpeak')]).flatten()
data_jantung.drop(['RestingBP', 'Cholesterol', 'Oldpeak'], axis=1, inplace=True)
data_jantung
```

| Age | HeartDisease | Sex_F | Sex_M | ChestPainType_ASY  | ChestPainType_ATA  | ChestPainType_NAP  | ChestPainType_TA  | RestingECG_LVH  | RestingECG_Normal | RestingECG_ST  | ExerciseAngina_N  | ExerciseAngina_Y  | ST_Slope_Down  | ST_Slope_Flat  | ST_Slope_Up | Dimension   |
|-----|--------------|-------|-------|--------------------|--------------------|--------------------|-------------------|-----------------|-------------------|----------------|-------------------|-------------------|----------------|----------------|-------------|-------------|
|  40 |            0 | False |  True |              False |               True |              False |             False |           False |              True |          False |              True |             False |          False |          False |        True |  50.203699  |
|  49 |            1 |  True | False |              False |              False |               True |             False |           False |              True |          False |              True |             False |          False |           True |       False | -58.137443  |
|  37 |            0 | False |  True |              False |               True |              False |             False |           False |             False |           True |              True |             False |          False |          False |        True |  43.902772  |
|  48 |            1 |  True | False |               True |              False |              False |             False |           False |              True |          False |             False |              True |          False |           True |       False | -24.820727  |
|  54 |            0 | False |  True |              False |              False |               True |             False |           False |              True |          False |              True |             False |          False |          False |        True | -43.449184  |

### Split Dataset

Pada proyek ini digunakan teknik split dataset, split dataset sangat penting dilakukan sebelum tahap modelling. dan untuk melakukan, kita perlu mengimport library split data yaitu  _train_test_split_, kemudian buat 2 variabel yaitu  X yang berfungsi untuk menghapus kolom _age_ dan y untuk menampilkan kolom age lalu bagi dataset menjadi 4 variabel baru  yaitu X_train, X_test, y_train, y_test dengan library _train_test_split_ dengan parameter yang digunakan yaitu :
* X berfungsi untuk menghapus kolom age
* y berfungsi menampilkan kolom age
* test_size adalah ukuran pembagian dataset yaitu sekitar 80 % untuk training dan 20 % untuk testing, data testing ini bertujuan untuk  mengukur kinerja model pada data baru.
* random_state: digunakan untuk mengontrol random number generator yang digunakan, di proyek ini menggunakan __random_state = 123__

```sh
from sklearn.model_selection import train_test_split

X = data_jantung.drop(["Age"],axis =1) 
y = data_jantung["Age"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

# Menampilkan hasil
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')
```

![image](https://github.com/user-attachments/assets/b1ec8dfe-8e78-4fa3-957a-9ca3ac972638)

### Standarisai

Algoritma machine learning cenderung menunjukkan kinerja yang lebih baik dan mencapai konvergensi yang lebih cepat ketika data dimodelkan dengan skala yang relatif sama atau mendekati distribusi normal. Oleh karena itu, proses scaling dan standardisasi sangat penting untuk memudahkan algoritma dalam memproses fitur data. Standardisasi merupakan metode transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, tidak akan diterapkan transformasi seperti one-hot-encoding yang digunakan untuk fitur kategorikal. Sebagai gantinya, kita akan memanfaatkan teknik StandardScaler dari library Scikit-learn untuk menormalkan data,

```sh
from sklearn.preprocessing import StandardScaler

numerical_features = ['HeartDisease', 'dimension']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()
```

|       | HeartDisease | Dimension  |
|-------|--------------|------------|
|  488  |    -0.873689 |   0.264232 |
|  11   |     1.144571 |  -1.473456 |
|  814  |     1.144571 |   1.280659 |
|  426  |    -0.873689 |  -1.440052 |
|  761  |     1.144571 |  -0.186384 |

Proses standarisasi mengubah nilai rata-rata (mean) menjadi 0 dan nilai standar deviasi menjadi 1. Untuk mengecek nilai mean dan standar deviasi pada setelah proses standarisasi

```sh
X_train[numerical_features].describe().round(4)
```

|               | HeartDisease | dimension |
|---------------|--------------|-----------|
| count         | 529.0000     | 529.0000  |
| mean          | -0.0000      | -0.0000   |
| std           | 1.0009       | 1.0009    |
| min           | -0.8737      | -3.0409   |
| 25%           | -0.8737      | -0.6609   |
| 50%           | -0.8737      | -0.1164   |
| 75%           | 1.1446       | 0.6389    |
| max           | 1.1446       | 3.2617    |

Tabel ini menampilkan statistik deskriptif untuk dua kolom: **HeartDisease** dan **dimension**.

# Modeling
Penulis menerapkan empat algoritma machine learning yang berbeda dalam proyek ini, yaitu:

* KNN (K-Nearest Neighbors)
* Random Forest
* AdaBoost
* SVM (Support Vector Machine)
Semua model dilatih menggunakan parameter default yang tersedia di library scikit-learn.

### Random Forest

Random Forest adalah algoritma prediksi yang memanfaatkan teknik ensemble bagging, di mana beberapa model independen berkolaborasi untuk menyelesaikan suatu masalah. Setiap model melakukan prediksi secara terpisah, dan hasil prediksi tersebut kemudian digabungkan untuk menghasilkan prediksi akhir. Pendekatan ini menjadikan Random Forest lebih akurat dibandingkan dengan model tunggal.

Secara dasar, Random Forest merupakan versi bagging dari algoritma Decision Tree. Setiap Decision Tree dilatih menggunakan subset data yang dipilih secara acak, baik dari segi fitur maupun sampel. Inilah sebabnya model ini disebut "Random Forest," karena terdiri dari banyak Decision Tree yang beroperasi secara acak.

**Cara Kerja Umum Random Forest :**
Mengambil k sampel dataset secara acak dengan pengembalian.
Membangun Decision Tree ke-i dari dataset tersebut.
Ulangi langkah di atas untuk k Decision Tree.
Proyek ini menggunakan Random Forest Regressor dari scikit-learn untuk menyelesaikan masalah regresi dengan parameter sebagai berikut:
    - n_estimators=50: jumlah Decision Tree dalam model.
    - max_depth=16: kedalaman maksimal setiap Decision Tree.
    - random_state=55: untuk memastikan hasil yang konsisten dengan nomor acak yang sama.
    - n_jobs=-1: memanfaatkan semua core CPU yang tersedia untuk menjalankan model secara paralel.
    
**Kelebihan dan Kekurangan Random Forest :**
- Kelebihan: Tangguh terhadap noise, mampu menangani data besar, serta mengatasi missing value.
- Kekurangan: Interpretasi yang sulit dan memerlukan tuning model yang tepat.

### KNN (K-Nearest Neighbors)

KNN (K-Nearest Neighbors) beroperasi dengan membandingkan jarak antara sampel uji dan sampel latih, kemudian memilih k tetangga terdekat untuk membuat prediksi. Dalam proyek ini, nilai `n_neighbors` diatur menjadi 10, yang berarti bahwa algoritma akan mempertimbangkan 10 tetangga terdekat. Untuk mengukur kedekatan antar-titik data, digunakan metrik jarak Euclidean, yang menghitung jarak linier antara dua titik dalam ruang fitur.

**Cara Kerja Umum KNN**
- Menentukan Nilai k: Langkah pertama adalah memilih nilai k, yaitu jumlah tetangga terdekat yang akan dipertimbangkan untuk membuat prediksi. Nilai k yang tepat dapat mempengaruhi performa model secara signifikan.
- Menghitung Jarak: Selanjutnya, algoritma menghitung jarak antara sampel uji dan semua sampel latih. Ini biasanya dilakukan menggunakan metrik jarak seperti jarak Euclidean, yang mengukur kedekatan antara dua titik dalam ruang fitur.
- Mengurutkan Data: Setelah jarak dihitung, algoritma mengurutkan semua sampel latih berdasarkan jarak terkecil dari sampel uji. Ini membantu dalam mengidentifikasi k tetangga terdekat.
- Menentukan Prediksi: Terakhir, KNN mengambil label dari k tetangga terdekat yang telah diurutkan dan menentukan prediksi berdasarkan mayoritas label di antara tetangga tersebut. Jika terjadi seri, biasanya salah satu dari tetangga terdekat dipilih secara acak.

**Kelebihan dan Kekurangan KNN**
- Kelebihan: Algoritma ini tangguh terhadap data latih yang noisy dan efektif untuk data yang besar.
- Kekurangan: Memerlukan penentuan nilai k yang tepat dan mahal dari sisi komputasi karena harus menghitung jarak setiap sampel uji dengan semua sampel latih.

### AdaBoost
AdaBoost, atau Adaptive Boosting, adalah algoritma boosting yang dirancang untuk meningkatkan akurasi prediksi dengan menggabungkan beberapa model sederhana (weak learners) menjadi satu model yang lebih kuat. Dalam prosesnya, AdaBoost memberikan bobot lebih pada kesalahan prediksi dari model sebelumnya, sehingga model selanjutnya fokus untuk memperbaiki kesalahan tersebut. Dalam proyek ini, AdaBoost Regressor dari library scikit-learn digunakan untuk meningkatkan performa model dengan cara ini. Algoritma ini efektif dalam meningkatkan ketahanan model terhadap overfitting dan seringkali menghasilkan prediksi yang lebih akurat dibandingkan model tunggal.

**Cara Kerja Umum AdaBoost**
Setiap observasi data latih diberi bobot yang sama pada awalnya.
Model pertama dibangun, dan bobot yang lebih besar diberikan pada sampel yang salah diklasifikasikan.
Model kedua dibangun untuk memperbaiki kesalahan model pertama.
Proses ini diulangi sampai akurasi yang diinginkan tercapai.

**Parameter yang digunakan:**
    - learning_rate=0.05: bobot yang diterapkan pada setiap regressor.
    - random_state=55: untuk menjaga konsistensi hasil.
    - Kelebihan dan Kekurangan AdaBoost
    - Kelebihan: Mudah diimplementasikan dan cepat dalam pengujian, cocok untuk real-time implementation.
    - Kekurangan: Memerlukan hypertuning yang tepat untuk performa optimal.

### SVM (Support Vector Machine)
SVM adalah algoritma yang bertujuan mencari hyperplane terbaik untuk memisahkan data dalam ruang berdimensi tinggi. Dalam proyek ini, digunakan Support Vector Regressor (SVR), yang merupakan versi regresi dari SVM.

**Cara Kerja Umum SVR**
SVR berusaha mencari "jalan" yang dapat menampung sebanyak mungkin sampel dalam batas toleransi tertentu. Support vector adalah sampel yang berada pada batas atau pembatas dari "jalan" ini.

**Kelebihan dan Kekurangan SVM**
- Kelebihan: Efektif untuk data berdimensi tinggi dan menggunakan subset dari sampel pelatihan sehingga memori lebih efisien.
- Kekurangan: Sulit diterapkan pada problem berskala besar karena biaya komputasi yang tinggi.


# Evaluation

Metrik yang digunakan untuk mengevaluasi model prediksi dalam proyek ini adalah Mean Squared Error (MSE). MSE menghitung rata-rata kuadrat dari selisih antara nilai aktual (ground truth) dan nilai yang diprediksi, memberikan perhatian lebih pada kesalahan yang lebih besar karena mengkuadratkan selisih tersebut. Hal ini membuat MSE menjadi indikator yang sensitif terhadap outlier, sehingga memberikan gambaran yang jelas tentang seberapa baik model dalam memprediksi nilai yang mendekati kenyataan. Dengan menggunakan MSE, kita dapat menilai kinerja model secara efektif dan membandingkannya dengan model lainnya.

**Rumus MSE:**
![image](https://github.com/user-attachments/assets/ff4c8572-5c96-407a-97e9-9304120b5cd1)

Keterangan:
N = jumlah dataset
yi = nilai sebenarnya
y_pred = nilai prediksi

Berikut adalah tabel yang menunjukkan nilai presisi dan recall untuk setiap model:

| Model               | Presisi (%) | Recall (%) |
|---------------------|-------------|------------|
| KNN                 | 0.067668    | 0.082806   |
| Random Forest (RF)  | 0.014918    | 0.10448    |
| Boosting            | 0.07228     | 0.088933   |
| SVM                 | 0.073573    | 0.081076   |

**Evaluasi**

Berikut adalah hasil evaluasi menggunakan Mean Squared Error (MSE) untuk empat algoritma yang digunakan:

* KNN:
    - Train MSE: 0.067668
    - Test MSE: 0.082806
    - Interpretasi: Model KNN menunjukkan hasil yang cukup konsisten antara data latih dan data uji. Meskipun nilai MSE untuk data uji sedikit lebih tinggi, hal ini mengindikasikan adanya noise atau outlier dalam data, namun secara umum model KNN mampu melakukan generalisasi yang baik.

* Random Forest:
    - Train MSE: 0.014918
    - Test MSE: 0.10448
    - Interpretasi: Meskipun Random Forest memiliki nilai MSE yang sangat rendah pada data latih, test MSE yang lebih tinggi menandakan model ini mengalami overfitting. Model terlalu fokus pada data latih dan kurang mampu menangani data baru.
      
* AdaBoost:
    - Train MSE: 0.07228
    - Test MSE: 0.088933
    - Interpretasi: AdaBoost menunjukkan hasil yang lebih stabil antara data latih dan data uji. Performa yang konsisten ini menunjukkan bahwa model mampu menghasilkan prediksi yang baik tanpa mengalami overfitting.

* SVM (SVR):
    - Train MSE: 0.073573
    - Test MSE: 0.081076
    - Interpretasi: SVM menampilkan performa yang seimbang antara data latih dan uji, menunjukkan kemampuannya dalam generalisasi yang baik pada data baru, dan model ini relatif lebih stabil dalam menangani berbagai data.
Menampilkan plot metrik dengan bar chart

![download (2)](https://github.com/user-attachments/assets/1b08181b-5646-4dba-9f8c-0a2490c96e2b)

Interpretasi Hasil:

* KNN: Performanya cukup konsisten namun sedikit lebih rentan terhadap noise dalam data. Dengan perbedaan kecil antara train dan test MSE, model ini dapat menangani variasi data baru dengan baik.
* Random Forest: Mengalami overfitting karena perbedaan yang signifikan antara train dan test MSE, sehingga kurang cocok untuk data baru.
* AdaBoost: Menunjukkan hasil yang seimbang dan tidak terlalu rentan terhadap overfitting. Performanya stabil baik di data latih maupun uji, menjadikannya kandidat yang kuat.
* SVM: Keseimbangan yang baik antara data latih dan uji menunjukkan bahwa SVM adalah model yang mampu menangani variasi data dengan baik, walaupun kadang model ini bisa kesulitan pada dataset yang sangat besar.

**Hasil Prediksi berdasarkan penghitungan  Mean Squared Error masing-masing algoritma**

![Screenshot 2024-10-15 195050](https://github.com/user-attachments/assets/c5ff751f-eb52-4113-a608-3662d74fcd1f)

Terlihat bahwa prediksi dari algoritma Boosting memberikan nilai yang paling mendekati nilai sebenarnya (y_true = 56) dengan prediksi 52.1, diikuti oleh SVM dengan prediksi 54.6, KNN dengan prediksi 51.1, dan Random Forest yang memiliki prediksi terjauh yaitu 44.8.

**Evaluasi Terhadap Business Understanding**

- Apakah sudah menjawab problem statement?
    - Ya, model yang digunakan telah membantu menjawab problem statement, terutama dalam prediksi biaya medis berdasarkan fitur-fitur seperti usia, BMI, jumlah anak, jenis kelamin, status perokok, dan region. Evaluasi MSE membantu mengukur tingkat kesalahan dari prediksi yang dihasilkan.

- Apakah berhasil mencapai goals yang diharapkan?
    - Goals untuk mendapatkan model yang mampu memprediksi biaya medis dengan konsistensi tinggi tercapai dengan model AdaBoost dan SVM, yang menunjukkan stabilitas antara data latih dan uji. Ini memberikan kepercayaan bahwa model dapat diandalkan untuk digunakan pada data baru.

- Apakah solusi statement yang rencanakan berdampak?
    - Ya, solusi dengan menggunakan berbagai algoritma dan mengevaluasinya berdasarkan MSE telah memberikan insight penting mengenai kekuatan dan kelemahan tiap model. Dampaknya adalah dapat mengidentifikasi model yang paling optimal untuk digunakan dalam memprediksi biaya medis dengan akurasi yang lebih tinggi, sehingga dapat mendukung pengambilan keputusan yang lebih baik.

# Kesimpulan
Berdasarkan evaluasi yang dilakukan, model AdaBoost dan SVM menunjukkan performa yang lebih konsisten dan stabil dibandingkan model lainnya, terutama dalam menjaga keseimbangan antara data latih dan data uji. Sementara itu, model Random Forest, meskipun memiliki Mean Squared Error (MSE) yang rendah pada data latih, mengalami masalah overfitting, yang mengakibatkan penurunan performa pada data uji. Di sisi lain, KNN tampak sedikit lebih rentan terhadap gangguan atau noise dalam data, yang dapat mempengaruhi akurasi prediksinya. Hal ini menunjukkan pentingnya pemilihan model yang tepat berdasarkan karakteristik data dan kebutuhan spesifik dari proyek ini.

###Kriteria Saran
**Referensi:**
[1] J. Smith, “Data Analysis Techniques for Machine Learning,” Journal of Data Science, vol. 12, no. 3, pp. 45-67, 2022.
[2] A. Brown, “Introduction to Machine Learning,” 3rd ed. New York: Springer, 2021.

**Solution Statement:**
- Dalam bagian Business Understanding, tambahkan "Solution Statement" yang jelas, yang menjelaskan pendekatan untuk mencapai tujuan analisis. Contoh:
    - "Analisis ini bertujuan untuk mengembangkan model prediktif yang dapat mengidentifikasi risiko penyakit jantung pada pasien berdasarkan data medis yang tersedia. Dengan menggunakan teknik machine learning, diharapkan model ini dapat membantu dalam pengambilan keputusan klinis yang lebih baik."

**Exploratory Data Analysis (EDA):**
- Tahapan yang dilakukan untuk memahami data termasuk visualisasi data dan analisis eksplorasi. Gambar dapat dilampirkan untuk menunjukkan distribusi data dan hubungan antar fitur.

**Teknik Data Preparation:**
- Teknik yang digunakan dalam persiapan data meliputi penghapusan outlier untuk memastikan analisis tidak terpengaruh oleh nilai ekstrem, encoding untuk fitur kategorikal agar model dapat memahami data, dan standarisasi untuk memastikan semua fitur berada dalam skala yang sama.

**Pendekatan Model:**
- Pendekatan yang digunakan meliputi KNN, Random Forest, Boosting, dan SVM. Setiap pendekatan memiliki kelebihan dan kekurangan, misalnya:
    - KNN: Mudah dipahami dan diimplementasikan, tetapi dapat lambat pada dataset besar.
    - Random Forest: Tahan terhadap overfitting dan dapat menangani data yang hilang, tetapi kompleksitasnya lebih tinggi.
    - Boosting: Meningkatkan akurasi model dengan menggabungkan beberapa model lemah, tetapi dapat rentan terhadap noise.
    - SVM: Efektif dalam ruang dimensi tinggi, tetapi memerlukan tuning parameter yang baik.

**Benchmarking:**
- Melakukan benchmarking untuk mengetahui model terbaik berdasarkan metrik evaluasi yang digunakan.
