# Laporan Proyek Machine Learning - Gladys Lady Nathasha

## Project Overview

Industri musik telah beralih dari distribusi fisik ke layanan streaming berbasis internet seperti Spotify, Apple Music, dan Joox di era ini, memberikan jutaan lagu yang dapat diakses kapan saja dan di mana saja. Namun, ketersediaan lagu dalam jumlah besar ini menimbulkan tantangan baru, yaitu overload informasi, yang membuat pengguna sulit menemukan lagu yang sesuai dengan preferensi mereka tanpa menggunakan sistem.   

Sistem rekomendasi musik berbasis konten berhasil mengatasi masalah ini. Dengan melihat karakteristik intrinsik dari lagu, seperti tempo, danceability, energy, acousticness, dan valence, sistem ini menawarkan rekomendasi lagu yang memiliki karakteristik yang sebanding dengan yang disukai pengguna sebelumnya. Metode ini sangat efektif dalam situasi di mana data interaksi pengguna terbatas atau tidak tersedia [1].   

Proyek ini bertujuan untuk membangun sistem rekomendasi musik menggunakan dataset Spotify Tracks DB dengan pendekatan content-based filtering. Dataset ini memuat metadata dan fitur audio dari ribuan lagu yang tersedia di platform Spotify. Sistem yang dibangun diharapkan mampu menyajikan rekomendasi lagu yang relevan dan personal berdasarkan fitur konten dari lagu yang disukai pengguna. 

Kehadiran sistem rekomendasi yang kuat memiliki dampak bisnis yang signifikan bagi penyedia layanan streaming, karena tidak hanya membuat pengguna lebih mudah menemukan musik yang mereka butuhkan, tetapi juga membantu mereka menemukan musik yang mereka cari. Rekomendasi berbasis konten adalah strategi yang kuat untuk meningkatkan pengalaman pengguna secara pribadi, seperti yang ditunjukkan oleh penelitian sebelumnya [2], dan masih menjadi salah satu metode utama dalam sistem cerdas musik [3].   

Dengan membangun proyek ini diharapkan bisa menunjukkan bagaimana pembelajaran mesin dapat digunakan dalam kehidupan sehari-hari dengan memberikan saran musik yang personal dan berbasis konten.

Referensi
[1] M. D. Ekstrand, J. T. Riedl, and J. A. Konstan, “Collaborative Filtering Recommender Systems,” Foundations and Trends® in Human–Computer Interaction, vol. 4, no. 2, pp. 81–173, 2011. [Online]. Available: https://www.nowpublishers.com/article/Details/HCI-009   

[2] J. T. Anthony, G. E. Christian, V. Evanlim, H. Lucky, and D. Suhartono, “The Utilization of Content Based Filtering for Spotify Music Recommendation,” in 2022 International Conference on Information and Electronics Engineering (ICIEE), Yogyakarta, Indonesia, Oct. 2022. DOI: 10.1109/ICIEE55596.2022.10010097   

[3] Y. Deldjoo, M. Schedl, and P. Knees, “Content-driven Music Recommendation: Evolution, State of the Art, and Challenges,” arXiv preprint arXiv:2107.11803, 2021. [Online]. Available: https://arxiv.org/pdf/2107.11803   

## Business Understanding

Sistem rekomendasi telah menjadi bagian penting dari layanan digital, termasuk platform streaming musik seperti Spotify. Dengan jutaan lagu yang tersedia, pengguna sering kali kesulitan menemukan musik baru yang sesuai dengan preferensi mereka. Untuk membuat rekomendasi yang relevan, personal, dan menarik, diperlukan pendekatan yang cerdas yang memahami karakteristik konten musik dan preferensi pengguna. Tujuan proyek ini adalah untuk mengembangkan sistem rekomendasi musik berbasis filter konten yang berfokus pada kemiripan konten antar lagu dan faktor popularitas.   

### Problem Statements

1. Bagaimana cara membantu pengguna menemukan lagu baru yang sesuai dengan selera mereka dari jutaan lagu yang tersedia?   
2. Bagaimana sistem rekomendasi bisa memberikan hasil yang lebih personal daripada sekedar menampilkan lagu populer?   
3. Bagaimana menjaga keberagaman lagu dalam daftar rekomendasi agar pengalaman mendengarkan tetap menarik dan tidak monoton?   

### Goals

1. Membangun sistem rekomendasi lagu yang memberikan daftar lagu relevan berdasarkan input lagu tertentu.   
2. Menggabungkan metode content-based filtering dengan reranking berdasarkan popularitas untuk meningkatkan kualitas rekomendasi.   
3. Mengukur performa sistem menggunakan metrik seperti Intra-List Similarity (ILS), Novelty (rata-rata popularitas), dan Recommendation Coverage.   

### Solution Statements/Solution Approach
1. Menggunakan fitur konten lagu untuk menghitung kemiripan antar lagu
Ekstraksi fitur seperti genre, tempo, energy, danceability, dan bahkan popularitas (untuk Solusi 1) digunakan untuk merepresentasikan setiap lagu dalam bentuk vektor numerik. Kemiripan antar lagu dihitung menggunakan cosine similarity, yang mengukur kesamaan arah antar vektor fitur tersebut.
2. Membangun dua pendekatan rekomendasi:
* Solusi 1: Content-based filtering dengan popularitas sebagai fitur   
Dalam pendekatan ini, fitur popularitas lagu digabungkan langsung ke dalam representasi fitur konten lagu. Ini membuat lagu yang populer dan memiliki kemiripan konten lebih tinggi akan lebih cenderung direkomendasikan.
* Solusi 2: Solusi 2: Content-based filtering tanpa popularitas, kemudian di-rerank berdasarkan popularitas   
Pendekatan ini memisahkan proses identifikasi lagu serupa dan popularitas. Pertama, sistem mengambil kandidat lagu yang mirip berdasarkan konten murni, kemudian mengurutkan ulang hasilnya berdasarkan nilai popularity. Ini menjaga integritas kemiripan konten sambil memperhatikan preferensi umum pengguna. 
3. Mengevaluasi performa sistem dengan metrik yang relevan
* Intra-List Similarity (ILS): Mengukur tingkat kesamaan antar lagu yang direkomendasikan. Nilai terlalu tinggi menunjukkan kurangnya keberagaman.   
* Novelty (Average Popularity): Mengukur rata-rata popularitas lagu-lagu yang direkomendasikan. Nilai lebih rendah menunjukkan sistem menyarankan lagu yang lebih novel dan tidak terlalu umum.   
* Recommendation Coverage: Mengukur proporsi lagu dalam katalog yang muncul dalam rekomendasi untuk berbagai input, mencerminkan seberapa luas jangkauan sistem.   

## Data Understanding
Pada tahap ini, kita akan melakukan eksplorasi dan pemahaman mendalam terhadap dataset yang menjadi dasar pengembangan sistem rekomendasi musik. Proses ini mencakup identifikasi karakteristik data, kualitas data, serta penggalian insight awal melalui analisis statistik dan visualisasi.   
Dataset yang digunakan dalam proyek ini adalah "Spotify Tracks DB", sebuah koleksi data publik yang tersedia di platform Kaggle. Dataset ini menghimpun berbagai metadata dan fitur audio dari jutaan lagu yang terdapat di Spotify. Untuk detail lebih lanjut dan akses ke dataset, dapat merujuk ke tautan berikut : https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db   
Secara keseluruhan, dataset ini memiliki dimensi yang cukup besar, yaitu terdiri dari 232.725 entri lagu yang masing-masing dijelaskan oleh 18 fitur berbeda. Tipe data dalam dataset ini beragam, mencakup fitur numerik (seperti `popularity`, `acousticness`, `danceability`, dll), fitur teks atau kategorikal (seperti `artist_name`, `track_name`, dll), dan identifikasi unik (`track_id`).   

### Deskripsi Variabel (Fitur)
Dataset ini terdiri dari 18 variabel yang memberikan informasi detail mengenai setiap lagu. Berikut adalah deskripsi singkat untuk masing-masing variabel:   
* ï»genre (nama variabel sebelum diubah menjadi genre) (object): Genre musik utama dari lagu (misalnya, 'Movie', 'Pop', 'Rock', 'Jazz').
* artist_name (object): Nama artis atau grup musik yang membawakan lagu.
* track_name (object): Judul atau nama resmi dari lagu.
* track_id (object): ID unik yang diberikan oleh Spotify untuk setiap lagu, berfungsi sebagai pengenal utama.
* popularity (int64): Skor popularitas lagu (0-100) berdasarkan perhitungan internal Spotify, di mana 100 adalah yang paling populer.
* acousticness (float64): Ukuran (0.0-1.0) yang menunjukkan seberapa akustik sebuah lagu. Nilai 1.0 menandakan keyakinan tinggi bahwa lagu tersebut bersifat akustik.
* danceability (float64): Indikator (0.0-1.0) seberapa cocok sebuah lagu untuk menari, dihitung berdasarkan elemen musik seperti tempo, stabilitas ritme, dan kekuatan ketukan.
* duration_ms (int64): Durasi lagu dalam satuan milidetik.
energy (float64): Ukuran persepsi (0.0-1.0) yang merepresentasikan intensitas dan aktivitas dalam sebuah lagu. Lagu yang energik biasanya terasa cepat dan keras.
* instrumentalness (float64): Memprediksi probabilitas sebuah lagu tidak mengandung vokal. Nilai yang mendekati 1.0 menunjukkan kemungkinan besar lagu tersebut adalah instrumental.
* key (object): Kunci nada dasar dari lagu (contoh: 'C', 'C#', 'D').
* liveness (float64): Mendeteksi keberadaan audiens atau suasana rekaman langsung (live) dalam sebuah lagu.
* loudness (float64): Tingkat kenyaringan keseluruhan lagu dalam desibel (dB). Nilai ini umumnya negatif.
* mode (object): Menunjukkan modalitas tangga nada lagu, yaitu 'Major' (mayor) atau 'Minor' (minor).
* speechiness (float64): Mendeteksi keberadaan kata-kata yang diucapkan dalam lagu. Nilai tinggi menunjukkan konten yang dominan berupa ucapan (misalnya podcast, puisi).
* tempo (float64): Perkiraan tempo lagu dalam ketukan per menit (BPM).
* time_signature (object): Estimasi tanda birama lagu (misalnya, '4/4', '3/4').
* valence (float64): Ukuran (0.0-1.0) yang menggambarkan positivitas musik. Nilai tinggi mengindikasikan musik yang lebih ceria atau bahagia.   

### Exploratory Data Analysis (EDA)

Setelah data berhasil dimuat, langkah krusial berikutnya adalah melakukan analisis eksplorasi data (EDA). Tujuan dari EDA adalah untuk memahami lebih dalam karakteristik dataset, mengidentifikasi pola, menemukan anomali, serta menggali insight awal yang dapat menginformasikan tahapan selanjutnya seperti persiapan data dan pemilihan model. Proses EDA ini melibatkan pemeriksaan informasi dasar, statistik deskriptif, evaluasi kualitas data, dan yang terpenting, visualisasi data.   

Berikut adalah tahapan EDA yang dilakukan secara berurutan:   

1. Pemeriksaan Informasi Umum dan Tipe Data Dataset   
Langkah awal dalam EDA adalah menggunakan fungsi `df.info()` untuk mendapatkan ringkasan komprehensif mengenai dataset. Dari output ini, diketahui bahwa dataset terdiri dari 232.725 entri (baris/lagu) dan 18 kolom (fitur). Teridentifikasi juga tipe data untuk setiap kolom, dimana terdapat 9 fitur bertipe float64 (umumnya fitur audio numerik), 2 fitur bertipe int64 (popularity dan duration_ms), dan 7 fitur bertipe object (teks atau kategori seperti ï»genre, artist_name, track_name, track_id, key, mode, dan time_signature). Selain itu, output ini juga mengonfirmasi bahwa hanya kolom track_name yang memiliki satu nilai hilang, sementara semua kolom lainnya memiliki data yang lengkap.   

2. Analisis Statistik Deskriptif Fitur Numerik   
Untuk memahami sebaran dan tendensi sentral dari fitur-fitur numerik, `df.describe()` digunakan. Ringkasan statistik ini memberikan informasi penting seperti rata-rata, standar deviasi, nilai minimum, maksimum, serta nilai kuartil (25%, 50% atau median, dan 75%) untuk setiap fitur numerik. Beberapa poin penting dari statistik deskriptif ini adalah:
* `popularity`: Memiliki rentang 0-100 dengan rata-rata sekitar 41.13 dan median 43.0, mengindikasikan variasi yang cukup dalam tingkat popularitas lagu.
* Fitur Audio (Skala 0-1): Fitur seperti `acousticness` (rata-rata 0.37), `danceability` (rata-rata 0.55), `energy` (rata-rata 0.57), dan `valence` (rata-rata 0.45) menunjukkan nilai rata-rata yang moderat pada skala 0-1. Demikian pula, `liveness` memiliki rata-rata sekitar 0.215 dengan median 0.128, yang mengindikasikan bahwa sebagian besar lagu dalam dataset kemungkinan tidak direkam secara langsung (live) atau memiliki sedikit elemen rekaman live. Sementara itu, `speechiness` memiliki rata-rata sekitar 0.121 dengan median 0.0501, nilai yang relatif rendah ini menunjukkan bahwa mayoritas lagu lebih didominasi oleh musik daripada kata-kata yang diucapkan (seperti pada podcast atau puisi). Median `instrumentalness` yang sangat rendah (mendekati nol, tepatnya 4.43e-05) juga menyiratkan bahwa sebagian besar lagu dalam dataset ini memiliki vokal dan bukan murni instrumental.
* `duration_ms`: Durasi lagu sangat bervariasi, dengan rata-rata sekitar 235.121 ms (sekitar 3 menit 55 detik).
* `loudness`: Kenyaringan rata-rata adalah -9.57 dB.
* `tempo`: Tempo rata-rata lagu adalah sekitar 117.67 BPM. Statistik ini membantu memahami skala dan distribusi nilai awal dari masing-masing fitur numerik.   

3. Pemeriksaan Nilai Hilang (Missing Values)   
Hasil dari `df.isnull().sum()` secara eksplisit mengonfirmasi bahwa hanya terdapat satu nilai yang hilang dalam keseluruhan dataset, yaitu pada kolom track_name.   

4. Pemeriksaan Data Duplikat   
Dengan menjalankan `df.duplicated().sum()`, dipastikan bahwa tidak ada baris data yang terduplikasi secara keseluruhan dalam dataset ini.

5. Distribusi Popularitas Lagu   
Untuk memvisualisasikan sebaran skor popularitas lagu, sebuah histogram dibuat. Grafik ini dengan jelas menunjukkan bahwa distribusi skor popularity bersifat condong ke kanan (right-skewed). Artinya, mayoritas lagu dalam dataset memiliki skor popularitas yang lebih rendah, dan jumlah lagu menurun seiring dengan meningkatnya skor popularitas. Hanya sebagian kecil lagu yang memiliki popularitas sangat tinggi. Insight ini penting karena mengindikasikan bahwa sistem mungkin perlu menangani banyak lagu yang bersifat niche atau kurang dikenal.   
Berikut adalah hasil visualisasi popularitas lagu:   
![output1](https://github.com/user-attachments/assets/cdc962c6-ec96-4713-bf28-ece0cbb76b13)


6. Komposisi Genre Terbanyak   
Sebuah plot batang horizontal digunakan untuk menampilkan 10 genre dengan frekuensi kemunculan tertinggi dalam dataset (berdasarkan kolom ï»¿genre sebelum nama variabel diubah). Visualisasi ini secara efektif mengidentifikasi genre-genre musik yang paling dominan atau paling banyak terwakili dalam data, genre genre tersebut adalah comedy, soundtrack, indie, jazz, pop, electronic, children's music, folk, hip-hop, rock. Ini memberikan pemahaman tentang genre mana yang paling banyak terwakili dalam data.   
Insight ini penting karena menunjukkan bahwa dataset memiliki representasi yang kuat untuk genre-genre tersebut. Hal ini bisa menjadi pertimbangan dalam mengevaluasi seberapa baik sistem rekomendasi dapat melayani preferensi pada genre-genre populer ini, atau sebaliknya, bagaimana sistem dapat memberikan rekomendasi untuk genre yang kurang dominan. Pemahaman komposisi genre juga berguna jika ada keinginan untuk membuat sistem rekomendasi yang lebih spesifik atau seimbang antar genre.   
Berikut adalah hasil visualisasi komposisi genre teratas :   
![output2](https://github.com/user-attachments/assets/79b9fadd-0ca2-4a13-abfc-85d00553affa)


7. Matriks Korelasi antar Fitur Numerik Audio   
Untuk memahami hubungan linear antar berbagai fitur audio numerik (termasuk popularity), sebuah heatmap korelasi dibuat. Heatmap ini menyoroti beberapa pola korelasi yang signifikan:
* Fitur energy menunjukkan korelasi positif yang sangat kuat dengan loudness (nilai korelasi sekitar 0.82). Ini sangat intuitif, karena lagu dengan tingkat energi tinggi umumnya dipersepsikan memiliki volume yang lebih keras. Selain itu, energy juga berkorelasi positif moderat dengan valence (sekitar 0.44).
* Fitur acousticness memiliki korelasi negatif yang kuat dengan energy (sekitar -0.73) dan juga dengan loudness (sekitar -0.69). Hal ini mengindikasikan bahwa lagu-lagu yang lebih akustik cenderung memiliki tingkat energi dan kenyaringan yang lebih rendah.
* danceability menunjukkan korelasi positif yang cukup baik dengan valence (sekitar 0.55), yang menyiratkan bahwa lagu yang lebih mudah untuk menari cenderung memiliki nuansa musik yang lebih positif. Danceability juga memiliki korelasi positif moderat dengan loudness (sekitar 0.44).
* loudness sendiri, selain hubungannya dengan energy dan acousticness, juga berkorelasi positif moderat dengan valence (sekitar 0.40).
* Fitur instrumentalness menunjukkan korelasi negatif yang cukup signifikan dengan loudness (sekitar -0.51), yang mungkin berarti lagu-lagu instrumental cenderung lebih tenang.
* Terdapat korelasi positif yang moderat antara liveness dan speechiness (sekitar 0.51). Ini bisa jadi karena rekaman langsung (live) seringkali menangkap lebih banyak elemen "ucapan" atau interaksi verbal, atau kedua fitur ini sensitif terhadap jenis artefak audio tertentu.
* Fitur popularity tidak menunjukkan korelasi yang sangat kuat dengan fitur audio lainnya, dengan korelasi positif tertinggi teramati pada loudness (sekitar 0.36). Korelasi negatif terkuat untuk popularity adalah sekitar -0.38 pada acousticness. Memang secara umum fitur popularity, memiliki korelasi negatif yang moderat/lemah dengan beberapa fitur.
* Fitur duration_ms (durasi lagu) secara umum memiliki korelasi yang lemah dengan sebagian besar fitur audio lainnya, yang menunjukkan durasi tidak secara langsung linear terkait dengan karakteristik audio tersebut.
* Fitur tempo juga tidak menunjukkan korelasi yang sangat dominan, korelasi terkuatnya adalah positif dengan energy dan loudness (keduanya sekitar 0.23) dan negatif dengan acousticness (sekitar -0.24).   
Berikut adalah hasil visualisasi matriks korelasi antar numerik audio :   
![output3](https://github.com/user-attachments/assets/8ae7f98e-0b86-4ffb-8eda-af12b66d7c97)


8. Distribusi Durasi Lagu   
Distribusi durasi lagu, yang ditampilkan dalam satuan menit untuk kemudahan interpretasi, juga menunjukkan pola yang condong ke kanan (right-skewed). Ini berarti sebagian besar lagu memiliki durasi standar yang umum di industri musik (misalnya, 2 hingga 5 menit), namun terdapat juga sejumlah lagu dengan durasi yang jauh lebih pendek atau lebih panjang dari rata-rata.   
Berikut adalah hasil visualisasi distribusi durasi lagu :   
![output4](https://github.com/user-attachments/assets/a1b6e7f9-48d8-422e-be0e-4c6b8c8ee370)
 

9. Artis dengan Kontribusi Lagu Terbanyak   
Plot batang ini menampilkan 10 artis yang memiliki jumlah lagu paling banyak dalam dataset. Visualisasi ini memberikan gambaran tentang artis-artis mana yang paling produktif atau paling banyak terwakili dalam data yang digunakan untuk proyek ini. Grafik ini secara visual menunjukkan artis mana yang mendominasi dataset dari segi jumlah lagu. Giuseppe Verdi, misalnya, terlihat memiliki kontribusi lagu tertinggi secara signifikan dibandingkan artis lainnya dalam 10 besar.   
Berikut adalah hasil visualisasi kontribusi lagu terbanyak : 
![output6](https://github.com/user-attachments/assets/4468ac40-4011-4d2c-91f2-cfa239dc17c6)


## Data Preparation
Pada tahap persiapan data (data preparation), dilakukan serangkaian teknik untuk membersihkan, memilih fitur, dan mentransformasi dataset agar optimal untuk tahap pemodelan dan analisis selanjutnya. Proses ini krusial untuk memastikan kualitas data yang baik, yang pada gilirannya akan meningkatkan akurasi dan reliabilitas hasil. Berikut adalah tahapan persiapan data yang dilakukan secara berurutan:   

1. Pembersihan Data (Data Cleaning)   
Pembersihan data adalah langkah fundamental untuk mengatasi isu-isu kualitas data seperti kesalahan format, nilai yang hilang, atau data yang tidak relevan.   

a.  Perbaikan Nama Kolom   
* Proses: Kolom pertama pada dataset diidentifikasi memiliki nama ï»¿genre yang mengandung karakter tidak standar (BOM atau Byte Order Mark). Nama kolom ini diganti menjadi genre menggunakan fungsi `df.rename(columns={'ï»¿genre': 'genre'}, inplace=True)`.   
* Alasan: Nama kolom yang bersih dan standar memudahkan pemanggilan dan pemrosesan kolom tersebut di langkah-langkah berikutnya. Karakter aneh dapat menyebabkan error atau kesulitan dalam mengakses data kolom.   

b.  Pengecekan dan Penanganan Nilai Hilang (Null Values)   
* Proses: Dilakukan pengecekan nilai hilang pada seluruh dataset menggunakan `df.isnull().sum()`. Ditemukan adanya satu nilai hilang pada kolom track_name. Mengingat jumlahnya yang sangat kecil (hanya satu baris) dibandingkan dengan total keseluruhan data (232.725 baris sebelum penghapusan), diputuskan untuk menghapus baris yang mengandung nilai hilang tersebut menggunakan `df.dropna(subset=['track_name'], inplace=True)`.   
* Alasan: Nilai hilang dapat mengganggu proses analisis dan pemodelan. Beberapa algoritma tidak dapat menangani nilai hilang secara langsung. Menghapus baris dengan nilai hilang adalah strategi yang valid jika jumlah data yang hilang sangat kecil dan tidak akan signifikan memengaruhi distribusi data secara keseluruhan. Ini memastikan integritas data untuk fitur krusial seperti track_name.   

c.  Pengaturan Ulang Indeks (Reset Index)   
* Proses: Setelah menghapus baris yang memiliki nilai hilang, indeks DataFrame menjadi tidak berurutan. Indeks diatur ulang menggunakan `df.reset_index(drop=True, inplace=True)` agar indeks kembali berurutan mulai dari 0 dan menghapus indeks lama.   
* Alasan: Indeks yang berurutan dan bersih memudahkan operasi berbasis indeks dan mencegah potensi kebingungan atau kesalahan dalam pemrosesan data di tahap selanjutnya.   

2. Pemilihan Fitur (Feature Selection) dan Definisi   
Tahap ini melibatkan pemilihan fitur-fitur yang relevan dari dataset yang akan digunakan dalam proses pemodelan atau analisis.   
Proses:   
* Didefinisikan daftar fitur numerik yang akan digunakan, yaitu `features = ['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']`.
* Dibuat juga variasi daftar fitur numerik tanpa menyertakan 'popularity', yang dinamakan features_no_pop. Ini disiapkan untuk skenario analisis kemiripan di mana 'popularity' mungkin tidak ingin dijadikan input langsung.
* Fitur kategorikal yang akan digunakan juga didefinisikan, yaitu `categorical_features = ['genre']`.   
Alasan: Pemilihan fitur yang tepat sangat penting karena memengaruhi performa model dan relevansi analisis. Dengan mendefinisikan daftar fitur secara eksplisit, dipastikan bahwa hanya data yang relevan yang akan diproses lebih lanjut, mengurangi noise dan potensi overfitting. Pemisahan fitur numerik dan kategorikal juga diperlukan karena keduanya memerlukan teknik pra-pemrosesan yang berbeda.   

3. Transformasi Fitur (Feature Transformation/Preprocessing)   
Transformasi fitur bertujuan untuk mengubah data mentah ke dalam format yang lebih sesuai untuk algoritma machine learning. Ini melibatkan penskalaan fitur numerik dan pengkodean fitur kategorikal.   

a. Definisi Preprocessor Utama (preprocessor)   
* Proses: ColumnTransformer pertama, bernama preprocessor, didefinisikan untuk menerapkan MinMaxScaler() pada fitur-fitur numerik yang ada dalam daftar features dan OneHotEncoder() pada categorical_features (kolom 'genre').
* MinMaxScaler(): Menskalakan fitur numerik ke rentang antara 0 dan 1.
* `OneHotEncoder(handle_unknown='ignore', sparse_output=True)`: Mengubah fitur 'genre' menjadi kolom-kolom biner (satu untuk tiap genre unik). Opsi `handle_unknown='ignore'` mengabaikan genre baru saat transformasi, dan `sparse_output=True` menghasilkan sparse matrix untuk efisiensi memori.
* `remainder='passthrough'`: Fitur lain yang tidak ditentukan akan dilewatkan tanpa perubahan.   
Alasan:   
* Penskalaan (MinMaxScaler): Diperlukan karena banyak algoritma machine learning sensitif terhadap skala fitur. Tanpa penskalaan, fitur dengan nilai besar dapat mendominasi perhitungan, menghasilkan model yang bias. Penskalaan memastikan semua fitur numerik berkontribusi secara seimbang.
* Pengkodean (OneHotEncoder): Algoritma machine learning memerlukan input numerik. OneHotEncoder mengubah kategori teks 'genre' menjadi angka tanpa mengasumsikan urutan antar genre, yang sesuai untuk data genre musik.   

b. Definisi Preprocessor Alternatif (preprocessor_no_pop)
* Proses: ColumnTransformer kedua, preprocessor_no_pop, dibuat serupa dengan yang pertama. Perbedaannya, MinMaxScaler() diterapkan pada daftar features_no_pop, yang tidak menyertakan fitur 'popularity'. Pengkodean untuk categorical_features menggunakan OneHotEncoder tetap sama.
* Alasan: Menyediakan fleksibilitas jika 'popularity' tidak ingin disertakan sebagai fitur input yang diskalakan dalam beberapa skenario analisis atau pemodelan.   

c. Persiapan dan Penerapan preprocessor Utama   
Proses:   
* `feature_df_for_processing = df[features + categorical_features]`: DataFrame feature_df_for_processing dibuat, hanya berisi kolom-kolom dari features (termasuk 'popularity') dan categorical_features.   
* `processed_features = preprocessor.fit_transform(feature_df_for_processing)`: preprocessor di-fit ke feature_df_for_processing untuk mempelajari parameter skala dan kategori unik, kemudian data ditransformasi.   
* Hasilnya adalah processed_features dengan dimensi (232724, 38).   
Alasan: Memastikan hanya fitur relevan yang diproses oleh preprocessor pertama. Outputnya adalah data yang siap digunakan, dengan fitur numerik diskalakan dan fitur kategorikal dikodekan. Penambahan jumlah kolom (menjadi 38) disebabkan oleh one-hot encoding pada fitur 'genre'.   

d. Persiapan dan Penerapan preprocessor_no_pop   
Proses:   
* `feature_df_for_processing_no_pop = df[features_no_pop + categorical_features]`: DataFrame feature_df_for_processing_no_pop dibuat, berisi kolom dari features_no_pop (tanpa 'popularity') dan categorical_features.
* `processed_features_no_pop = preprocessor_no_pop.fit_transform(feature_df_for_processing_no_pop)`: preprocessor_no_pop di-fit dan mentransformasi data ini.   
Hasilnya adalah processed_features_no_pop dengan dimensi (232724, 37).   
Alasan: Menghasilkan versi data terproses di mana 'popularity' tidak termasuk dalam fitur yang diskalakan. Jumlah kolom menjadi 37 karena satu fitur numerik ('popularity') tidak dimasukkan ke MinMaxScaler pada preprocessor ini.   

Penggunaan ColumnTransformer menyederhanakan penerapan berbagai transformasi pada subset kolom yang berbeda secara terorganisir dan konsisten.   

## Modeling
Tahapan modeling berfokus pada implementasi algoritma untuk menghasilkan rekomendasi musik yang relevan bagi pengguna. Berdasarkan fitur-fitur yang telah dipersiapkan, dikembangkan dua buah fungsi rekomendasi yang mengimplementasikan pendekatan berbeda untuk menghasilkan daftar Top-N lagu.   

Kedua pendekatan ini memanfaatkan kemiripan konten (Content-Based) yang diukur menggunakan metrik Cosine Similarity. Cosine Similarity menghitung kesamaan arah antara dua vektor fitur, yang sangat efektif untuk menentukan seberapa mirip karakteristik audio dua lagu, terlepas dari magnitudonya. Vektor fitur yang digunakan adalah hasil dari tahap Data Preparation (processed_features dan processed_features_no_pop).   

Untuk memudahkan pencarian lagu, dibuat sebuah mapping (song_indices) dari nama lagu ke indeksnya dalam DataFrame.   

1. Pendekatan Pertama: Content-Based Filtering Murni (get_recommendations_on_the_fly)   
Pendekatan ini mengimplementasikan Content-Based Filtering secara langsung. Rekomendasi dihasilkan murni berdasarkan skor kemiripan konten tertinggi.   

**Cara Kerja**:   
a. Input: Menerima nama lagu (track_name) dan matriks fitur (all_processed_features, dalam contoh menggunakan processed_features yang mencakup 'popularity').   
b. Pencarian: Mencari indeks lagu input.   
c. Perhitungan Kemiripan: Menghitung Cosine Similarity antara vektor fitur lagu input dengan semua lagu lain dalam dataset.   
d. Pengurutan: Mengurutkan semua lagu berdasarkan skor kemiripan, dari yang tertinggi hingga terendah.   
e. Seleksi Top-N: Mengambil N lagu teratas (setelah membuang lagu input itu sendiri) yang memiliki skor kemiripan tertinggi.   
f. Output: Menampilkan N lagu rekomendasi beserta artist_name, genre, popularity, dan similarity_score.      

**Hasil Output**:   
Berikut adalah rekomendasi lagu berdasarkan kemiripan konten untuk lagu **"C'est beau de faire un Show" oleh Henri Salvador**:

|                 track_name              |    artist_name    | genre | popularity | Skor Kemiripan |
|-----------------------------------------|-------------------|-------|------------|----------------|
|                 Together                |    Donny Osmond   | Movie |      6     |      0.995     |
|  Le miroir magique (par Dany Brillant)  | Martin & les fÃ©es| Movie |      1     |      0.993     |
|         Sukhkarta Dukharta Remix        |      Chorus       | Movie |      0     |      0.993     |
|            Aarti Kunj Bihari Ki         |      Chorus       | Movie |      0     |      0.993     |
| Toute ma vie j'ai chanté du rock'n'roll |     DorothÃ©e     | Movie |      1     |      0.993     |

**Kelebihan**:   
* Relevansi Konten Tinggi: Menghasilkan lagu yang secara objektif paling mirip berdasarkan fitur audio dan genre (dan popularitas, jika disertakan dalam fitur).
* Transparan: Mudah dipahami mengapa sebuah lagu direkomendasikan (karena skor kemiripannya tinggi).
* Cepat & Sederhana: Implementasi langsung dari Content-Based, relatif cepat untuk dijalankan (on-the-fly).   

**Kekurangan**:   
* Kurang Beragam: Cenderung menghasilkan lagu yang sangat mirip, berpotensi kurang memberikan serendipity atau penemuan baru yang mengejutkan.
* Sensitif terhadap Fitur: Sangat bergantung pada kualitas fitur yang digunakan. Jika 'popularity' disertakan dalam fitur, bisa jadi lagu populer lebih sering muncul meski kemiripan audionya tidak setinggi yang lain.
* Mengabaikan Popularitas Global: Tidak secara eksplisit mempertimbangkan apakah lagu yang mirip tersebut disukai banyak orang atau tidak (kecuali 'popularity' dimasukkan sebagai fitur dalam perhitungan similarity).

1. Pendekatan Kedua: Content-Based Filtering dengan Popularity Re-ranking (get_recommendations_reranked_by_popularity)   
Pendekatan ini adalah strategi hibrida dua tahap. Tahap pertama menggunakan Content-Based Filtering untuk menghasilkan daftar kandidat yang relevan, dan tahap kedua mengurutkan ulang (re-rank) kandidat tersebut berdasarkan popularitas.   

**Cara Kerja**:   
a. Input: Menerima nama lagu (track_name), matriks fitur konten (all_processed_features_content, dalam contoh menggunakan processed_features_no_pop agar kemiripan murni konten), DataFrame asli (original_df untuk mengambil nilai 'popularity'), jumlah rekomendasi akhir (N), dan jumlah kandidat awal (M).   
b. Pencarian & Perhitungan Kemiripan: Sama seperti pendekatan pertama, menghitung Cosine Similarity (kali ini menggunakan fitur tanpa 'popularity').   
c. Seleksi Kandidat (Tahap 1): Mengambil M lagu teratas (misalnya 50 lagu) berdasarkan skor kemiripan. Ini adalah daftar kandidat yang lebih besar dari N.   
d. Re-ranking (Tahap 2): Mengurutkan ulang M kandidat tersebut. Kriteria utama pengurutan adalah popularity (dari yang tertinggi ke terendah). Jika ada lagu dengan popularitas yang sama, similarity_score digunakan sebagai kriteria kedua (tie-breaker).   
e. Seleksi Top-N Akhir: Mengambil N lagu teratas dari daftar yang sudah diurutkan ulang.   
f. Output: Menampilkan N lagu rekomendasi yang merupakan lagu paling populer di antara lagu-lagu yang paling mirip.   

**Hasil Output**:   
Berikut adalah rekomendasi lagu yang diurutkan ulang berdasarkan popularitas untuk lagu  **"C'est beau de faire un Show" oleh Henri Salvador**:

|                 track_name              |     artist_name     | genre | popularity | Skor Kemiripan |
|-----------------------------------------|---------------------|-------|------------|----------------|
|                 Loup loup               |     Chantal Goya    | Movie |     18     |      0.992     |
|          Pour faire une chanson         |      DorothÃ©e      | Movie |     14     |      0.989     |
|        Les chevaliers du zodiaque       | Le Club des Juniors | Movie |     11     |      0.989     |
|     Where Did All The Good Times Go     |     Donny Osmond    | Movie |     10     |      0.989     |
|     Twist De L'enrhumÃ© - Remastered    |    Henri Salvador   | Movie |      7     |      0.992     |


**Kelebihan**:   
* Keseimbangan Relevansi & Popularitas: Mencoba menyeimbangkan antara lagu yang "mirip" dan lagu yang "disukai banyak orang", yang seringkali meningkatkan kepuasan pengguna.
* Mengurangi Risiko Niche Tidak Populer: Mengurangi kemungkinan merekomendasikan lagu yang sangat mirip tetapi sangat tidak populer atau obskur (kecuali tidak ada pilihan lain).
* Potensi Kepuasan Lebih Tinggi: Pengguna mungkin lebih cenderung menyukai rekomendasi yang sudah terbukti populer di kalangan orang lain, selama masih relevan.  

**Kekurangan**:   
* Popularitas Bias: Masih dapat cenderung mengarah ke lagu-lagu populer, meskipun sudah difilter berdasarkan konten. Lagu niche yang relevan bisa tergeser oleh lagu populer yang relevansinya sedikit lebih rendah.
* Membutuhkan Parameter Tambahan: Memerlukan penyesuaian parameter M (jumlah kandidat) yang bisa memengaruhi hasil akhir.
* Lebih Kompleks: Sedikit lebih kompleks dalam implementasi dan penalaran dibandingkan Content-Based murni.

## Evaluation
Tahap evaluasi ini bertujuan untuk mengukur dan membandingkan performa dari dua solusi sistem rekomendasi yang telah dibangun:   
* Solusi 1: Content-Based Filtering di mana popularitas dimasukkan sebagai salah satu fitur (get_recommendations_on_the_fly).
* Solusi 2: Content-Based Filtering murni (tanpa popularitas) yang hasilnya diurutkan ulang (re-ranked) berdasarkan popularitas (get_recommendations_reranked_by_popularity).   

Evaluasi dilakukan melalui dua cara: analisis kualitatif dengan melihat contoh rekomendasi secara langsung, dan analisis kuantitatif menggunakan metrik formal: Intra-List Similarity (ILS), Novelty, dan Recommendation Coverage.

1. Analisis Kualitatif: Studi Kasus Rekomendasi   
Untuk mendapatkan pemahaman intuitif tentang perilaku kedua model, mari kita lihat hasil rekomendasi untuk lagu input "C'est beau de faire un Show" oleh Henri Salvador (Genre: Movie, Popularitas: 0).   

Dari contoh ini, terlihat jelas perbedaan perilaku kedua model. Solusi 1 memberikan lagu dengan skor kemiripan tertinggi, meskipun popularitasnya sangat rendah. Sementara itu, Solusi 2 berhasil mempromosikan lagu "Loup loup" dengan Popularitas 18 ke puncak daftar, meskipun skor kemiripannya (0.992) sedikit lebih rendah dari "Together" (0.995) dari Solusi 1. Ini secara kualitatif menunjukkan bahwa mekanisme re-ranking pada Solusi 2 bekerja sesuai tujuan, yaitu menyeimbangkan relevansi konten dengan popularitas.   

2. Analisis Kuantitatif: Hasil Metrik Evaluasi
Untuk mengukur performa secara objektif, evaluasi kuantitatif dilakukan pada sampel beberapa lagu uji.   

a. Intra-List Similarity (ILS) - Mengukur Keberagaman   
ILS mengukur seberapa mirip lagu-lagu di dalam sebuah daftar rekomendasi. Skor yang lebih rendah menunjukkan keberagaman yang lebih baik.   
* Cara Kerja dan Formula:   
Metrik ini bekerja dengan menghitung rata-rata skor kemiripan (misalnya, cosine similarity) antara semua kemungkinan pasangan lagu di dalam satu daftar rekomendasi. Proses ini diulang untuk banyak permintaan rekomendasi, lalu hasilnya dirata-ratakan. Skor yang lebih rendah menunjukkan keberagaman yang lebih baik.   
Formula untuk satu daftar rekomendasi L dengan N item adalah:

$$
\text{ILS}(L) = \frac{2}{N(N-1)} \sum_{i=1}^{N} \sum_{j=i+1}^{N} \text{similarity}(item_i, item_j)
$$   

Di mana:   
L adalah daftar lagu yang direkomendasikan.   
N adalah jumlah lagu dalam daftar tersebut.   
textsimilarity(item_i,item_j) adalah skor kemiripan (cosine similarity) antara lagu ke-i dan lagu ke-j dalam daftar.   

* Hasil: Berdasarkan eksperimen, Solusi 1 memiliki hasil yaitu 0.8594, 0.9940, 0.8847 dan jika kita hitung ketiga hasil tersebut menjadi rata-rata, maka rata-rata ILS 0.9127, sedangkan Solusi 2 memiliki hasil yaitu 0.9894, 0.8588, 0.7866 dan skor rata-rata dari ketiga hasil tersebut lebih rendah yaitu 0.8782.
* Visualisasi:
![output7](https://github.com/user-attachments/assets/0830622d-4084-4e59-a28d-5d71b0fbbe2b)


* Analisis: Hasil ini mengkonfirmasi bahwa Solusi 2 lebih unggul dalam hal keberagaman. Proses re-ranking berdasarkan popularitas berhasil memasukkan lagu-lagu yang mungkin tidak "paling mirip" dari segi konten murni, sehingga memecah homogenitas dan menyajikan daftar yang lebih bervariasi kepada pengguna.   

b. Novelty (Rata-Rata Popularitas) - Mengukur Tingkat Penemuan   
Metrik ini mengukur kemampuan sistem untuk merekomendasikan lagu-lagu baru atau yang kurang dikenal (novel). Metrik ini diukur dari rata-rata skor popularitas lagu yang direkomendasikan. Skor yang lebih rendah menunjukkan tingkat novelty yang lebih tinggi.   
* Cara Kerja dan Formula:   
Metrik ini dihitung dengan mengambil rata-rata dari skor popularity semua lagu dalam sebuah daftar rekomendasi. Skor yang lebih rendah menunjukkan tingkat novelty yang lebih tinggi.    
Formula untuk satu daftar rekomendasi L dengan N item adalah:   

$$
\text{AveragePopularity}(L) = \frac{1}{N} \sum_{i=1}^{N} \text{popularity}(item_i)
$$   

Di mana:   
L adalah daftar lagu yang direkomendasikan.   
N adalah jumlah lagu dalam daftar tersebut.   
textpopularity(item_i) adalah skor popularitas dari lagu ke-i.   

Formula untuk satu daftar rekomendasi L dengan N item adalah:
* Hasil: Solusi 1 menunjukkan hasil yaitu 1.6, 3.0, 60.8 dan rata-rata popularitas 21.80, yang secara signifikan lebih rendah dibandingkan Solusi 2 dengan skor hasil yaitu 12.0, 20.0, 59.8 dan rata-rata 30.60.
* Visualisasi:
![output8](https://github.com/user-attachments/assets/e3a80eaa-a1f4-4035-a9b3-a601e1b95a6c)


* Analisis: Solusi 1 secara jelas lebih unggul dalam menyajikan lagu-lagu yang novel. Dengan tidak adanya bias eksplisit terhadap popularitas dalam penentuan peringkat akhir, model ini mampu merekomendasikan "permata tersembunyi" yang relevan secara konten. Sebaliknya, Solusi 2, sesuai dengan desainnya, cenderung merekomendasikan lagu-lagu yang sudah populer.   

c. Recommendation Coverage - Mengukur Jangkauan Katalog   
Coverage mengukur seberapa banyak item unik dari keseluruhan katalog yang mampu direkomendasikan oleh sistem. Skor yang lebih tinggi menunjukkan jangkauan yang lebih baik.   
* Cara Kerja dan Formula:   
Coverage dihitung sebagai persentase dari jumlah lagu unik yang muncul di semua daftar rekomendasi yang dihasilkan selama pengujian, dibagi dengan jumlah total lagu unik yang ada dalam katalog. Skor yang lebih tinggi menunjukkan jangkauan yang lebih baik.   
Formula:   
$$
\text{Coverage} = \frac{|\bigcup_{u \in U} L_u|}{|I|} \times 100\%
$$   

Di mana:   
U adalah himpunan semua permintaan rekomendasi (misalnya, dari lagu-lagu uji).   
L_u adalah daftar rekomendasi untuk satu permintaan u.   
∣bigcup_uinUL_u∣ adalah jumlah total lagu unik yang muncul di semua daftar rekomendasi.   
∣I∣ adalah jumlah total lagu unik dalam keseluruhan katalog.   

* Hasil: Dalam estimasi yang dilakukan pada 50 lagu query, kedua solusi menunjukkan hasil coverage yang hampir identik dan dapat dianggap setara.   
Solusi 1: 0.0017 (merekomendasikan 249 lagu unik)   
Solusi 2: 0.0017 (merekomendasikan 250 lagu unik)   
* Visualisasi:
![output9](https://github.com/user-attachments/assets/b42089ee-6710-4797-9a7e-c36f650a21c9)


* Analisis: Hasil ini menunjukkan bahwa, pada skala pengujian ini, mekanisme re-ranking pada Solusi 2 tidak secara signifikan membatasi jangkauan katalog dibandingkan Solusi 1. Ini bisa jadi karena jumlah kandidat awal (M=20) yang diambil sebelum re-ranking sudah cukup beragam, sehingga item unik yang muncul di top-N tetap bervariasi. Dapat disimpulkan bahwa untuk metrik ini, tidak ada perbedaan performa yang signifikan antara kedua solusi.   

**Ringkasan Evaluasi dan Kesimpulan**

|         Metrik Evaluasi       |  Solusi 1 (Pop as Feature)  |  Solusi 2 (Pop for Rerank)  |  Pemenang  |
|-------------------------------|-----------------------------|-----------------------------|------------|
|        Keberagaman (ILS)      |   0.9127 (Kurang beragam)   |    0.8782 (Lebih beragam)   |  Solusi 2  |
|   Novelty (Avg. Popularity)   |     21.80 (Lebih novel)     |     30.60 (Kurang novel)    |  Solusi 1  |
|      Jangkauan (Coverage)     |       0.0017 (Setara)       |        0.0017 (Setara)      |   Setara   |


## Kesimpulan
Berdasarkan analisis kuantitatif dan kualitatif yang telah dilakukan, dapat disimpulkan bahwa proyek ini berhasil mencapai tujuannya dengan memberikan jawaban yang didukung oleh data untuk setiap problem statement yang ada. Kedua solusi yang dikembangkan, yang berlandaskan pada pendekatan content-based filtering dan cosine similarity, telah terbukti mampu memberikan rekomendasi yang relevan dan personal, lebih dari sekadar menampilkan daftar lagu populer.   

Untuk menjawab tantangan dalam membantu pengguna menemukan lagu baru yang sesuai selera mereka, evaluasi metrik Novelty menunjukkan bahwa Solusi 1 (Pop as Feature) lebih unggul. Dengan rata-rata popularitas rekomendasi yang rendah (21.80), model ini membuktikan kemampuannya untuk menyajikan lagu-lagu yang lebih novel atau kurang dikenal, sehingga memberikan pengalaman penemuan musik yang otentik dan personal. Selanjutnya, untuk mengatasi masalah keberagaman agar pengalaman mendengarkan tidak monoton, metrik Intra-List Similarity (ILS) memberikan jawaban yang jelas. Solusi 2 (Pop for Rerank), dengan mekanisme re-ranking-nya, berhasil mencatatkan skor ILS yang lebih rendah (0.8782), yang menandakan kemampuannya dalam menyajikan daftar lagu yang lebih beragam dibandingkan Solusi 1. Sementara itu, metrik Coverage menunjukkan bahwa kedua solusi memiliki potensi yang seimbang dalam menjangkau katalog musik yang luas.   

Secara keseluruhan, proyek ini berhasil memenuhi semua goals yang ditetapkan: membangun sistem rekomendasi yang relevan, menggabungkan dua pendekatan berbeda, dan mengukurnya dengan metrik yang sesuai. Evaluasi ini menyoroti adanya trade-off yang fundamental antara kedua solusi. Solusi 1 adalah pilihan terbaik untuk mendorong penemuan musik baru dan personalisasi yang mendalam, sementara Solusi 2 menawarkan keseimbangan yang lebih baik antara keberagaman dan penerimaan umum oleh pengguna.
