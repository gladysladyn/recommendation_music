# %% [markdown]
# # Project Model Sistem Rekomendasi
# * Nama : Gladys Lady Nathasha
# * ID Dicoding : MC222D5X1379
# * Kelas : MC-22

# %% [markdown]
# Tujuan dari notebook ini adalah untuk menganalisis fitur audio dari dataset Spotify Tracks DB dan membangun sebuah sistem rekomendasi lagu berdasarkan kemiripan karakteristiknya.

# %% [markdown]
# ## Import Library

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import combinations

# %% [markdown]
# Penjelasan Proses: Sel kode ini berfungsi untuk mengimpor semua pustaka (library) yang diperlukan untuk analisis.
# * pandas digunakan untuk manipulasi dan analisis data (membuat dan mengelola DataFrame).
# * matplotlib.pyplot dan seaborn digunakan untuk visualisasi data.
# * sklearn (Scikit-learn) akan digunakan nanti untuk proses preprocessing data dan machine learning, seperti scaling dan encoding.
# * numpy digunakan untuk komputasi numerik.   
# 
# Penjelasan Hasil: Setelah menjalankan sel ini, semua fungsi dari pustaka tersebut siap digunakan di dalam notebook.

# %% [markdown]
# ## Load Data

# %%
# Ganti 'nama_file.csv' dengan nama file yang telah diupload
df = pd.read_csv('music_spotify_dataset/SpotifyFeatures.csv', encoding='latin1')

# Tampilkan 5 baris pertama
print(df.head())

# %% [markdown]
# * Penjelasan Proses: Kode ini memuat dataset dari file SpotifyFeatures.csv ke dalam sebuah DataFrame pandas yang diberi nama df. Parameter encoding='latin1' digunakan karena dataset mungkin berisi karakter khusus yang tidak dapat dibaca oleh encoding default (UTF-8). Setelah itu, df.head() dipanggil untuk menampilkan 5 baris pertama dari DataFrame. Ini adalah langkah verifikasi awal untuk memastikan data telah dimuat dengan benar dan untuk mendapatkan gambaran pertama tentang struktur data.   
# 
# * Penjelasan Hasil: Output df.head() menunjukkan bahwa data berhasil dimuat. Terlihat ada 18 kolom, termasuk fitur kategorikal seperti genre, artist_name, dan track_name, serta fitur numerik seperti popularity, acousticness, danceability, dan lainnya. Ini mengonfirmasi struktur data yang akan kita analisis.

# %% [markdown]
# ## Exploratory Data

# %% [markdown]
# Pada tahap ini, kita akan melakukan investigasi awal pada data untuk menemukan pola, anomali, dan mendapatkan wawasan utama.

# %% [markdown]
# ###  Informasi Dasar & Pengecekan Missing Values

# %%
print(df.info())

# %%
df.describe()

# %%
print(df.isnull().sum())

# %%
print(f"Jumlah duplikat: {df.duplicated().sum()}")

# %% [markdown]
# Penjelasan Proses:
# * df.info(): Digunakan untuk mendapatkan ringkasan singkat dari DataFrame, termasuk jumlah total baris, nama setiap kolom, jumlah nilai non-null, dan tipe datanya.
# * df.isnull().sum(): Menghitung jumlah nilai yang hilang (kosong/NaN) untuk setiap kolom.
# * df.duplicated().sum(): Menghitung jumlah baris yang merupakan duplikat sempurna dari baris lain. Langkah-langkah ini sangat penting dalam tahap pembersihan data (data cleaning).   
# 
# Penjelasan Hasil:
# * Dari df.info(), kita tahu dataset memiliki 232,725 baris.
# * Dari df.isnull().sum(), teridentifikasi hanya ada 1 nilai yang hilang pada kolom track_name. Jumlah ini sangat kecil dibandingkan total data, sehingga tidak akan menjadi masalah besar.
# * Terdapat 0 baris duplikat. Ini menunjukkan bahwa dataset cukup bersih dari sisi duplikasi data.

# %% [markdown]
# ### Visualisasi Distribusi Fitur

# %%
# Distribusi Popularitas Lagu
plt.figure(figsize=(10,5))
sns.histplot(df['popularity'], bins=30, kde=True)
plt.title('Distribusi Popularitas Lagu')
plt.xlabel('Popularitas')
plt.ylabel('Jumlah Lagu')
plt.show()

# %% [markdown]
# * Penjelasan Proses: Kode ini membuat histogram untuk memvisualisasikan distribusi dari fitur popularity. kde=True menambahkan garis estimasi kepadatan kernel (Kernel Density Estimate) untuk memperhalus bentuk distribusi. Ini membantu kita memahami sebaran tingkat popularitas lagu dalam dataset.
# 
# * Penjelasan Hasil/Insight: Grafik menunjukkan bahwa distribusi popularitas tidak normal. Terdapat puncak utama di sekitar skor popularitas 50-60, yang berarti sebagian besar lagu memiliki tingkat popularitas moderat. Ada juga puncak yang lebih kecil di dekat 0, menandakan banyak lagu yang tidak populer. Sebaran ini memiliki "ekor panjang" ke kanan (right-skewed), menunjukkan lagu dengan popularitas sangat tinggi (di atas 80) lebih sedikit jumlahnya.

# %%
# Grafik 10 Genre Teratas
df['ï»¿genre'].value_counts().head(10).plot(kind='barh')
plt.title('Top 10 Genre Terbanyak')
plt.xlabel('Jumlah Lagu')
plt.ylabel('Genre')
plt.show()

# %% [markdown]
# * Penjelasan Proses: Kode ini menghitung jumlah lagu untuk setiap genre menggunakan value_counts(), memilih 10 genre teratas dengan .head(10), dan menampilkannya dalam bentuk diagram batang horizontal (barh) untuk kemudahan perbandingan.
# 
# * Penjelasan Hasil/Insight: Visualisasi menunjukkan 10 genre dengan jumlah lagu terbanyak dalam dataset. Genre Rock, Hip-Hop, Folk, dan Pop adalah genre yang paling dominan. Menariknya, jumlah lagu di antara 10 genre teratas ini relatif seimbang, masing-masing memiliki sekitar 9.000 hingga 9.500 lagu.

# %%
# Pilih fitur numerik untuk korelasi
numerical_features = [
    'popularity', 'acousticness', 'danceability', 'duration_ms',
    'energy', 'instrumentalness', 'liveness', 'loudness',
    'speechiness', 'tempo', 'valence'
]

# Plot heatmap korelasi ANtar Fitur Numerik
plt.figure(figsize=(12, 10))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Korelasi antar Fitur Numerik')
plt.show()

# %% [markdown]
# * Penjelasan Proses: Kode ini menghitung matriks korelasi menggunakan metode .corr() pada fitur-fitur numerik yang telah dipilih. Hasilnya kemudian divisualisasikan menggunakan heatmap dari Seaborn. annot=True menampilkan nilai korelasi pada setiap sel, dan cmap='coolwarm' memberikan skema warna di mana biru menunjukkan korelasi negatif dan merah menunjukkan korelasi positif.   
# 
# Penjelasan Hasil/Insight:
# * Korelasi Positif Kuat: Terlihat korelasi positif yang sangat kuat antara energy dan loudness (0.82). Ini sangat masuk akal, karena lagu yang lebih keras cenderung terdengar lebih berenergi.
# * Korelasi Negatif Kuat: Terdapat korelasi negatif yang kuat antara energy dan acousticness (-0.73). Ini juga intuitif, karena lagu akustik biasanya lebih tenang dan tidak terlalu berenergi.
# * Korelasi Lain: Fitur danceability memiliki korelasi positif moderat dengan valence (kebahagiaan/positivitas) sebesar 0.55.
# * Popularitas: Fitur popularity tidak menunjukkan korelasi yang kuat dengan fitur audio lainnya, menandakan bahwa popularitas lagu adalah konsep yang kompleks dan tidak ditentukan hanya oleh satu atau dua atribut audio saja.
# * Tempo dan duration_ms (durasi lagu) secara umum memiliki korelasi yang lemah dengan sebagian besar fitur audio lainnya, yang menunjukkan tempo dan durasi tidak secara langsung linear terkait dengan karakteristik audio tersebut.

# %%
# Distribusi Durasi Lagu
sns.histplot(df['duration_ms'] / 60000, bins=50, kde=True)
plt.title("Distribusi Durasi Lagu (dalam Menit)")
plt.xlabel("Durasi (menit)")
plt.show()

# %% [markdown]
# * Penjelasan Proses: Pertama, kolom duration_ms (durasi dalam milidetik) diubah menjadi menit untuk interpretasi yang lebih mudah. Kemudian, histogram dibuat untuk menunjukkan distribusi durasi lagu. Sumbu x dibatasi hingga 20 menit karena sebagian besar lagu berada dalam rentang ini, sehingga kita bisa melihat distribusinya dengan lebih jelas.   
# 
# * Penjelasan Hasil/Insight: Grafik menunjukkan distribusi yang sangat miring ke kanan (heavily right-skewed). Sebagian besar lagu dalam dataset memiliki durasi antara 2 hingga 5 menit, yang merupakan durasi standar untuk lagu-lagu populer. Ada sangat sedikit lagu yang memiliki durasi sangat panjang (di atas 10 menit).

# %%
# Grafik 10 Artis Teratas
df['artist_name'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Artist dengan Lagu Terbanyak')
plt.xlabel('Artist')
plt.ylabel('Jumlah Lagu')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# * Penjelasan Proses: Sama seperti pada analisis genre, kode ini menggunakan value_counts() untuk menghitung jumlah lagu per artis, mengambil 10 teratas, dan memvisualisasikannya dalam bentuk diagram batang vertikal.
# 
# * Penjelasan Hasil/Insight: Giuseppe Verdi dan Giacomo Puccini adalah dua artis dengan jumlah lagu terbanyak di dataset ini. Banyak dari artis di 10 besar ini adalah komposer musik klasik. Hal ini kemungkinan disebabkan oleh cara karya klasik dikatalogkan di Spotify, di mana setiap bagian atau gerakan dari sebuah simfoni atau opera dihitung sebagai "track" terpisah.

# %% [markdown]
# ## Preprocessing Data

# %% [markdown]
# Tahap ini bertujuan untuk membersihkan dan mentransformasi data agar siap digunakan untuk pemodelan. Proses ini meliputi pembersihan data, pemilihan fitur, dan transformasi fitur.

# %% [markdown]
# ### Data Cleaning

# %% [markdown]
# a. Perbaikan Nama Kolom   

# %%
df.rename(columns={'ï»¿genre': 'genre'}, inplace=True)

# %% [markdown]
# * Penjelasan Proses: Pada langkah ini, kita mengganti nama kolom pertama dari ï»¿genre menjadi genre. Karakter ï»¿ di awal adalah Byte Order Mark (BOM) yang sering muncul saat file CSV disimpan dengan encoding UTF-8. Karakter ini harus dihilangkan agar kolom dapat diakses dengan mudah menggunakan nama 'genre'. Parameter inplace=True digunakan untuk menerapkan perubahan langsung pada DataFrame df tanpa perlu membuatnya kembali.   
# 
# * Penjelasan Hasil: Nama kolom pertama telah berhasil diubah menjadi 'genre', membuat struktur data lebih bersih dan konsisten.

# %% [markdown]
# b. Cek dan Tangani Null   

# %%
# Ada satu nilai null di kolom track_name. Kita bisa drop baris tersebut.
df.dropna(subset=['track_name'], inplace=True)

# %%
# Verifikasi ulang
print(df.isnull().sum())

# %%
# Setelah menghapus baris, index perlu di-reset agar berurutan kembali.
df.reset_index(drop=True, inplace=True)

# %% [markdown]
# * Penjelasan Proses: Dari tahap EDA, kita mengetahui ada satu baris dengan nilai kosong (null) pada kolom track_name. Karena jumlahnya hanya satu dari total >200.000 data, dampak dari penghapusan baris ini sangat kecil. Oleh karena itu, kita menggunakan df.dropna() untuk menghapus baris tersebut. Setelah itu, df.reset_index(drop=True) dijalankan untuk mengatur ulang indeks DataFrame agar kembali berurutan dari 0 setelah satu baris dihapus.   
# 
# * Penjelasan Hasil: Output dari df.isnull().sum() menunjukkan bahwa sekarang tidak ada lagi nilai yang hilang di seluruh kolom dataset. Data sudah bersih dari missing values.

# %% [markdown]
# ### Feature Selection

# %%
# Fitur numerik yang akan digunakan untuk kemiripan konten
features = [
    'popularity', 'acousticness', 'danceability', 'duration_ms',
    'energy', 'instrumentalness', 'liveness', 'loudness',
    'speechiness', 'tempo', 'valence'
]

# %%
# Fitur numerik TANPA 'popularity'
features_no_pop = [
    'acousticness', 'danceability', 'duration_ms',
    'energy', 'instrumentalness', 'liveness', 'loudness',
    'speechiness', 'tempo', 'valence'
]

# %%
# Fitur kategorikal yang akan digunakan
categorical_features = ['genre']

# %% [markdown]
# Penjelasan Proses: Pada langkah ini, kita secara eksplisit memilih fitur-fitur yang akan digunakan untuk membangun sistem rekomendasi berbasis konten.
# * features: Mencakup semua fitur audio numerik ditambah popularity.
# * features_no_pop: Daftar fitur yang sama, tetapi tanpa popularity. Daftar ini dibuat khusus untuk menghitung kemiripan (similarity) konten murni, agar popularity tidak membiaskan perhitungan. Rekomendasi seharusnya didasarkan pada karakteristik audio lagu, bukan popularitasnya.
# * categorical_features: Memilih genre sebagai fitur kategoris yang akan dimasukkan ke dalam model.   
# 
# Penjelasan Hasil: Tiga buah list Python (features, features_no_pop, categorical_features) telah dibuat. List ini akan digunakan pada tahap transformasi untuk memastikan hanya fitur-fitur yang relevan yang diproses.

# %% [markdown]
# ### Feature Transformation

# %%
# Normalisasi fitur Scaling dan Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features)
    ],
    remainder='passthrough'
)

# %%
# Buat ulang preprocessor dengan features_no_pop
preprocessor_no_pop = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), features_no_pop),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features)
    ],
    remainder='passthrough'
)

# %%
# Membuat Pipeline untuk Transformasi Fitur
feature_df_for_processing = df[features + categorical_features]

# Terapkan preprocessor
print("Menerapkan preprocessing pada fitur...")
processed_features = preprocessor.fit_transform(feature_df_for_processing)
print(f"Dimensi fitur setelah preprocessing: {processed_features.shape}")

# %%
# Buat ulang feature_df_for_processing dan processed_features
feature_df_for_processing_no_pop = df[features_no_pop + categorical_features] # Gunakan features_no_pop

print("Menerapkan preprocessing pada fitur (tanpa popularity untuk similarity)...")
processed_features_no_pop = preprocessor_no_pop.fit_transform(feature_df_for_processing_no_pop)
print(f"Dimensi fitur baru setelah preprocessing: {processed_features_no_pop.shape}")

# %% [markdown]
# Penjelasan Proses:
# * ColumnTransformer: Kita menggunakan ColumnTransformer untuk menerapkan transformasi yang berbeda pada kolom yang berbeda secara bersamaan.
# * MinMaxScaler: Fitur numerik (seperti duration_ms, loudness, dll.) memiliki skala yang sangat berbeda. MinMaxScaler menormalkan semua fitur ini ke dalam rentang [0, 1]. Ini sangat penting agar fitur dengan skala besar tidak mendominasi perhitungan kemiripan (similarity).
# * OneHotEncoder: Algoritma machine learning tidak bisa memproses data teks ('genre'). OneHotEncoder mengubah setiap nilai genre menjadi vektor biner. Misalnya, jika ada 27 genre, setiap lagu akan direpresentasikan oleh vektor dengan panjang 27, di mana hanya satu elemen yang bernilai 1 (menandakan genre lagu tersebut).
# * Dua preprocessor dibuat untuk dua set fitur yang berbeda (dengan dan tanpa popularity). Kemudian, metode fit_transform diterapkan untuk melakukan normalisasi dan encoding pada data.   
# 
# Penjelasan Hasil:
# * processed_features: Data asli telah diubah menjadi matriks numerik berdimensi (232724, 38). Ini berarti 232.724 lagu, dengan masing-masing 38 fitur hasil transformasi (11 fitur numerik + 27 fitur genre hasil OneHotEncoding).
# * processed_features_no_pop: Matriks kedua berdimensi (232724, 37). Perbedaan 1 kolom ini karena fitur popularity tidak diikutsertakan. Matriks inilah yang akan kita gunakan untuk menghitung cosine_similarity.

# %% [markdown]
# ## Modelling

# %% [markdown]
# Pada tahap ini, kita membangun fungsi untuk memberikan rekomendasi lagu berdasarkan kemiripan konten.

# %%
#  Membuat Fungsi Rekomendasi
song_indices = pd.Series(df.index, index=df['track_name']).drop_duplicates()

def get_recommendations_on_the_fly(track_name, all_processed_features, N=10):
    if track_name not in song_indices:
        print(f"Lagu '{track_name}' tidak ditemukan dalam dataset.")
        return pd.DataFrame()

    idx = song_indices[track_name]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    # Ambil vektor fitur untuk lagu input
    query_song_features = all_processed_features[idx]

    if hasattr(query_song_features, "reshape"):
        sim_scores_vector = cosine_similarity(query_song_features.reshape(1, -1), all_processed_features)
    else: # Asumsi sparse matrix
        sim_scores_vector = cosine_similarity(query_song_features, all_processed_features)


    # sim_scores_vector akan berbentuk (1, jumlah_lagu), ambil baris pertama
    sim_scores = list(enumerate(sim_scores_vector[0]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    recommended_song_indices = [i[0] for i in sim_scores]
    
    recommended_songs_df = df.iloc[recommended_song_indices][['track_name', 'artist_name', 'genre', 'popularity']]
    similarity_values = [round(i[1], 3) for i in sim_scores]
    recommended_songs_df['similarity_score'] = similarity_values
    
    return recommended_songs_df

# %%
def get_recommendations_reranked_by_popularity(track_name, all_processed_features_content, original_df, N=10, M=50):
    # M adalah jumlah kandidat awal berdasarkan konten sebelum di-rerank
    if track_name not in song_indices:
        print(f"Lagu '{track_name}' tidak ditemukan dalam dataset.")
        return pd.DataFrame()

    idx = song_indices[track_name]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    query_song_features = all_processed_features_content[idx]

    if hasattr(query_song_features, "reshape"):
        sim_scores_vector = cosine_similarity(query_song_features.reshape(1, -1), all_processed_features_content)
    else:
        sim_scores_vector = cosine_similarity(query_song_features, all_processed_features_content)

    sim_scores = list(enumerate(sim_scores_vector[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Ambil M kandidat teratas berdasarkan konten (lebih banyak dari N)
    content_based_candidates_indices = [i[0] for i in sim_scores[1:M+1]]
    
    # Ambil detail lagu kandidat, termasuk popularity dari DataFrame ASLI
    candidate_songs_df = original_df.iloc[content_based_candidates_indices].copy()

    # Ambil similarity scores yang sesuai untuk candidate_songs_df
    idx_to_score_map = {original_df.index[tpl[0]]: tpl[1] for tpl in sim_scores[1:M+1] if tpl[0] < len(original_df)}
    candidate_songs_df['similarity_score'] = candidate_songs_df.index.map(idx_to_score_map)
    
    # Hapus baris di candidate_songs_df dimana similarity_score mungkin NaN jika ada ketidakcocokan indeks (jarang terjadi jika M kecil)
    candidate_songs_df.dropna(subset=['similarity_score'], inplace=True)


    # Urutkan ulang kandidat berdasarkan 'popularity' (descending), lalu 'similarity_score' (descending) sebagai tie-breaker
    reranked_songs_df = candidate_songs_df.sort_values(by=['popularity', 'similarity_score'], ascending=[False, False])

    # Ambil N teratas setelah di-rerank
    final_recommendations = reranked_songs_df.head(N)[['track_name', 'artist_name', 'genre', 'popularity', 'similarity_score']]

    return final_recommendations

# %% [markdown]
# Penjelasan Proses: Dua fungsi rekomendasi yang berbeda telah didefinisikan:   
# 
# 1. get_recommendations_on_the_fly: Fungsi ini menerapkan pendekatan Content-Based Filtering murni.
# * Ia mencari lagu input, lalu mengambil vektor fiturnya yang sudah diproses.
# * Menggunakan cosine_similarity, ia menghitung kemiripan antara lagu input dengan semua lagu lain di dataset.
# * Fungsi mengurutkan lagu berdasarkan skor kemiripan tertinggi dan mengembalikan N lagu teratas.   
# 
# 2. get_recommendations_reranked_by_popularity: Fungsi ini menggunakan pendekatan hybrid dua tahap untuk meningkatkan relevansi.
# * Tahap 1 (Candidate Generation): Sama seperti fungsi pertama, ia mencari M lagu yang paling mirip berdasarkan konten (menggunakan fitur tanpa popularity). M adalah angka yang lebih besar dari N (misal, 50).
# * Tahap 2 (Re-ranking): M lagu kandidat ini kemudian diurutkan ulang, bukan berdasarkan kemiripan, tetapi berdasarkan popularity (dari tertinggi ke terendah). Skor kemiripan digunakan sebagai pemecah jika ada popularitas yang sama.
# * Fungsi kemudian mengembalikan N lagu teratas dari daftar yang sudah diurutkan ulang ini. Tujuannya adalah untuk merekomendasikan lagu yang tidak hanya mirip, tetapi juga cenderung disukai banyak orang.   
# 
# Penjelasan Hasil: Dua fungsi telah berhasil dibuat dan siap untuk digunakan. song_indices juga dibuat untuk memetakan judul lagu ke indeksnya, yang akan mempercepat proses pencarian lagu di dalam fungsi.

# %%
# Contoh Penggunaan Fungsi Rekomendasi
if not df.empty:
    try:
        test_song_name = df['track_name'].iloc[0]
        print(f"\n--- Rekomendasi (on-the-fly) untuk lagu: {test_song_name} ({df['artist_name'].iloc[0]}) ---")
        # Panggil fungsi baru dengan 'processed_features' sebagai argumen
        recommendations = get_recommendations_on_the_fly(test_song_name, processed_features, N=5)
        print(recommendations)

    except IndexError:
        print("DataFrame kosong atau lagu tidak ditemukan untuk pengujian.")
    except KeyError as e:
        print(f"Kolom tidak ditemukan: {e}. Pastikan DataFrame 'df' sudah benar.")
else:
    print("DataFrame 'df' kosong. Tidak dapat menjalankan contoh rekomendasi.")

# %%
# Contoh Penggunaan Fungsi Rekomendasi (bagian yang diubah)
if not df.empty:
    try:
        test_song_name = df['track_name'].iloc[0]
        print(f"\n--- Rekomendasi (reranked by popularity) untuk lagu: {test_song_name} ({df['artist_name'].iloc[0]}) ---")
        
        recommendations_reranked = get_recommendations_reranked_by_popularity(
            test_song_name,                 
            processed_features_no_pop,      
            df,                             
            N=5,                            
            M=20                            
        )
        print(recommendations_reranked)

    except IndexError:
        print("DataFrame kosong atau lagu tidak ditemukan untuk pengujian.")
    except KeyError as e:
        print(f"Kolom tidak ditemukan: {e}. Pastikan DataFrame 'df' sudah benar.")
else:
    print("DataFrame 'df' kosong. Tidak dapat menjalankan contoh rekomendasi.")

# %% [markdown]
# Penjelasan Proses: Kode ini bertujuan untuk menguji dan membandingkan output dari kedua fungsi rekomendasi yang telah dibuat. Lagu pertama dalam dataset, "C'est beau de faire un Show", digunakan sebagai input. Hasil dari kedua fungsi kemudian dicetak untuk dianalisis perbedaannya.   
# 
# Penjelasan Hasil/Insight:
# * Output Fungsi 1 (On-the-fly): Rekomendasi yang diberikan memiliki skor kemiripan (similarity_score) yang sangat tinggi (di atas 0.99). Namun, skor popularity dari lagu-lagu yang direkomendasikan sangat rendah (berkisar 0-6). Ini menunjukkan model berhasil menemukan lagu yang secara audio sangat mirip, tetapi lagu-lagu tersebut mungkin tidak dikenal atau tidak populer.
# * Output Fungsi 2 (Reranked by Popularity): Rekomendasi yang diberikan masih memiliki kemiripan yang tinggi (di atas 0.98), tetapi skor popularity-nya jauh lebih baik (berkisar 7-18).
# K* esimpulan: Perbandingan ini dengan jelas menunjukkan keunggulan pendekatan re-ranking. Dengan menyeimbangkan antara kemiripan konten dan popularitas, model kedua mampu memberikan rekomendasi yang tidak hanya relevan secara audio tetapi juga lebih mungkin disukai oleh pengguna pada umumnya.

# %% [markdown]
# ## Evaluation

# %% [markdown]
# Pada tahap ini, kita akan mengevaluasi dan membandingkan performa dari dua solusi sistem rekomendasi yang telah kita bangun:   
# 
# * Solusi 1: Model yang menggunakan popularity sebagai salah satu fitur dalam perhitungan kemiripan konten.
# * Solusi 2: Model yang tidak menggunakan popularity untuk kemiripan konten, tetapi menggunakannya pada tahap akhir untuk mengurutkan ulang (re-ranking) hasil rekomendasi.   
# 
# Evaluasi dilakukan secara kualitatif (melihat contoh output) dan kuantitatif (menggunakan metrik offline).

# %%
# 1. Evaluasi Kualitatif: Menampilkan Contoh Rekomendasi
if 'track_name' in df.columns and len(df) > 200:
    sample_test_songs = [
        df['track_name'].iloc[0],  
        df['track_name'].iloc[100], 
        df['track_name'].iloc[200],
    ]
else:
    print("DataFrame 'df' tidak memiliki cukup data atau kolom 'track_name' tidak ada untuk membuat sample_test_songs.")
    sample_test_songs = []

N_recommendations = 5
M_candidates_for_reranking = 20

# %%
# Loop untuk menampilkan perbandingan hasil untuk setiap lagu uji
for song_title in sample_test_songs:
    if song_title not in song_indices:
        print(f"\nLagu uji '{song_title}' tidak ditemukan di song_indices. Skipping.")
        continue

    print(f"\n\n--- Lagu Input: {song_title} ---")
    artist_input = df.loc[song_indices[song_title] if not isinstance(song_indices[song_title], pd.Series) else song_indices[song_title].iloc[0], 'artist_name']
    genre_input = df.loc[song_indices[song_title] if not isinstance(song_indices[song_title], pd.Series) else song_indices[song_title].iloc[0], 'genre']
    popularity_input = df.loc[song_indices[song_title] if not isinstance(song_indices[song_title], pd.Series) else song_indices[song_title].iloc[0], 'popularity']
    print(f"(Artis: {artist_input}, Genre: {genre_input}, Popularitas: {popularity_input})")

    # Solusi 1: Popularity sebagai fitur konten
    print("\n--- Solusi 1: Rekomendasi (Popularity sebagai fitur) ---")
    recommendations_s1 = get_recommendations_on_the_fly(
        song_title,
        processed_features,
        N=N_recommendations
    )
    print(recommendations_s1)

    # Solusi 2: Konten tanpa Popularity, Re-ranked by Popularity
    print("\n--- Solusi 2: Rekomendasi (Konten tanpa Pop, Re-ranked by Pop) ---")
    recommendations_s2 = get_recommendations_reranked_by_popularity(
        song_title,
        processed_features_no_pop, # Menggunakan processed_features_no_pop
        df, # DataFrame asli untuk info popularity
        N=N_recommendations,
        M=M_candidates_for_reranking
    )
    print(recommendations_s2)
    print("-"*40)

# %% [markdown]
# Penjelasan Proses: Evaluasi kualitatif dilakukan dengan cara mengamati secara langsung hasil rekomendasi dari kedua solusi untuk beberapa lagu sampel. Proses ini membantu kita mendapatkan "rasa" atau intuisi tentang bagaimana perilaku setiap model. Kita memilih tiga lagu dari dataset sebagai input, kemudian memanggil kedua fungsi rekomendasi (get_recommendations_on_the_fly untuk Solusi 1 dan get_recommendations_reranked_by_popularity untuk Solusi 2) dan mencetak hasilnya berdampingan.   
# 
# Penjelasan Hasil/Insight: Dari output yang ditampilkan (dipotong untuk keringkasan), kita bisa melihat perbedaan yang jelas.
# * Untuk lagu input dengan popularitas rendah (contoh: 'C'est beau de faire un Show' dengan popularitas 0), Solusi 1 cenderung merekomendasikan lagu-lagu lain yang juga memiliki popularitas sangat rendah.
# * Sebaliknya, Solusi 2 untuk input yang sama berhasil merekomendasikan lagu-lagu dengan popularitas yang lebih tinggi. Ini menunjukkan bahwa mekanisme re-ranking efektif dalam "mengangkat" lagu yang lebih dikenal ke daftar rekomendasi teratas, yang berpotensi lebih memuaskan bagi pengguna.

# %%
# 2. Evaluasi Kuantitatif: Mendefinisikan dan Menghitung Metrik Offline
# a. Definisi Fungsi Metrik

# Intra-List Similarity (ILS)
def calculate_intra_list_similarity(recommendations_df, all_song_features, song_indices_map, original_df):
    recommended_indices_in_features = []
    for track_name_rec in recommendations_df['track_name']:
        if track_name_rec in song_indices_map:
            idx_val = song_indices_map[track_name_rec]
            recommended_indices_in_features.append(idx_val.iloc[0] if isinstance(idx_val, pd.Series) else idx_val)
        # else: item yg direkomendasikan tidak ada di map? Seharusnya tidak terjadi jika data konsisten

    if len(recommended_indices_in_features) < 2:
        return 0.0

    total_similarity = 0.0
    num_pairs = 0

    try:
        recommended_vectors = all_song_features[recommended_indices_in_features, :]
    except TypeError: # Jika indices masih ada pd.Series (seharusnya tidak)
         actual_indices = [idx.iloc[0] if isinstance(idx, pd.Series) else idx for idx in recommended_indices_in_features]
         recommended_vectors = all_song_features[actual_indices, :]


    for i in range(recommended_vectors.shape[0]):
        for j in range(i + 1, recommended_vectors.shape[0]):
            vec1 = recommended_vectors[i]
            vec2 = recommended_vectors[j]
            
            # cosine_similarity expects 2D arrays
            if hasattr(vec1, "reshape"): 
                similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]
            else: # If sparse matrix row
                similarity = cosine_similarity(vec1, vec2)[0, 0]
            
            total_similarity += similarity
            num_pairs += 1
    
    return total_similarity / num_pairs if num_pairs > 0 else 0.0

# %%
# Novelty (Average Popularity of Recommended Items)
def calculate_average_popularity_novelty(recommendations_df):
    if recommendations_df.empty or 'popularity' not in recommendations_df.columns:
        return None
    # Rata-rata popularitas (nilai lebih rendah = lebih novel)
    avg_popularity = recommendations_df['popularity'].mean()
    return avg_popularity

# %%
# Recommendation Coverage (Estimasi)
def calculate_recommendation_coverage(recommendation_function, processed_features_for_function, 
                                      query_song_titles_sample, total_unique_songs_in_catalog, 
                                      N_rec, **kwargs):

    all_recommended_items_set = set()
    
    for song_title_query in query_song_titles_sample:
        if song_title_query not in song_indices: 
            continue
        
        if 'original_df_for_rerank' in kwargs: 
            recs_df = recommendation_function(
                song_title_query, 
                processed_features_for_function,
                kwargs['original_df_for_rerank'], 
                N=N_rec,
                M=kwargs.get('M_candidates', 20) 
            )
        else: 
            recs_df = recommendation_function(
                song_title_query, 
                processed_features_for_function, 
                N=N_rec
            )
            
        if not recs_df.empty and 'track_name' in recs_df.columns:
            all_recommended_items_set.update(recs_df['track_name'].tolist())
            
    coverage_score = len(all_recommended_items_set) / total_unique_songs_in_catalog if total_unique_songs_in_catalog > 0 else 0
    return coverage_score, len(all_recommended_items_set)

# %% [markdown]
# Penjelasan Proses: Pada sel ini, kita mendefinisikan tiga fungsi untuk menghitung metrik evaluasi:   
# 
# 1. calculate_intra_list_similarity (ILS): Fungsi ini mengukur seberapa mirip item-item di dalam satu daftar rekomendasi. Ia menghitung rata-rata cosine similarity dari semua pasangan item yang direkomendasikan. Skor ILS yang tinggi menandakan rekomendasi yang sangat homogen (kurang beragam), sedangkan skor yang lebih rendah menunjukkan keragaman yang lebih baik.
# 2. calculate_average_popularity_novelty (Novelty): Fungsi ini digunakan sebagai proksi untuk mengukur kebaruan (novelty) dari rekomendasi. Ia menghitung rata-rata skor popularity dari item yang direkomendasikan. Asumsinya adalah, semakin rendah rata-rata popularitasnya, semakin "novel" atau tak terduga rekomendasinya.
# 3. calculate_recommendation_coverage: Fungsi ini mengestimasi seberapa banyak dari katalog lagu yang dapat direkomendasikan oleh sistem. Ia menghitung rasio antara jumlah lagu unik yang direkomendasikan (dari sejumlah query sampel) terhadap jumlah total lagu unik di dataset. Skor yang lebih tinggi lebih baik, karena menunjukkan sistem tidak hanya merekomendasikan lagu-lagu yang itu-itu saja.   
# 
# Penjelasan Hasil: Tiga fungsi metrik telah berhasil dibuat dan siap digunakan untuk mengevaluasi kedua solusi sistem rekomendasi kita secara kuantitatif.

# %%
# b. Menjalankan Evaluasi dan Agregasi Hasil

# --- Menjalankan Evaluasi untuk Sampel Lagu Uji ---
all_ils_s1 = []
all_novelty_s1 = []
all_ils_s2 = []
all_novelty_s2 = []

for song_title in sample_test_songs:
    if song_title not in song_indices:
        continue

    print(f"\n--- Mengevaluasi untuk Lagu Input: {song_title} ---")
    
    # Evaluasi Solusi 1
    recs_s1 = get_recommendations_on_the_fly(song_title, processed_features, N=N_recommendations)
    if not recs_s1.empty:
        ils_s1 = calculate_intra_list_similarity(recs_s1, processed_features, song_indices, df)
        novelty_s1 = calculate_average_popularity_novelty(recs_s1)
        all_ils_s1.append(ils_s1)
        if novelty_s1 is not None: all_novelty_s1.append(novelty_s1)
        print(f"  Solusi 1 - ILS: {ils_s1:.4f}, Avg Pop (Novelty): {novelty_s1 if novelty_s1 is not None else 'N/A'}")

    # Evaluasi Solusi 2
    recs_s2 = get_recommendations_reranked_by_popularity(song_title, processed_features_no_pop, df, N=N_recommendations, M=M_candidates_for_reranking)
    if not recs_s2.empty:
        # Untuk ILS Solusi 2, kemiripan antar item yang direkomendasikan harus dihitung berdasarkan fitur konten (processed_features_no_pop)
        ils_s2 = calculate_intra_list_similarity(recs_s2, processed_features_no_pop, song_indices, df)
        novelty_s2 = calculate_average_popularity_novelty(recs_s2)
        all_ils_s2.append(ils_s2)
        if novelty_s2 is not None: all_novelty_s2.append(novelty_s2)
        print(f"  Solusi 2 - ILS: {ils_s2:.4f}, Avg Pop (Novelty): {novelty_s2 if novelty_s2 is not None else 'N/A'}")

# %%
# Rata-rata Metrik (dari sampel lagu uji)
print("\n--- Rata-rata Metrik dari Sampel Lagu Uji ---")
if all_ils_s1: print(f"Solusi 1 - Rata-rata ILS: {np.mean(all_ils_s1):.4f}")
if all_novelty_s1: print(f"Solusi 1 - Rata-rata Avg Pop (Novelty): {np.mean(all_novelty_s1):.2f}")
if all_ils_s2: print(f"Solusi 2 - Rata-rata ILS: {np.mean(all_ils_s2):.4f}")
if all_novelty_s2: print(f"Solusi 2 - Rata-rata Avg Pop (Novelty): {np.mean(all_novelty_s2):.2f}")

# %%
# Kalkulasi Estimasi Coverage (Gunakan sampel yang lebih besar jika memungkinkan)
print("\n--- Estimasi Recommendation Coverage ---")
num_query_songs_for_coverage = 50
if len(df) >= num_query_songs_for_coverage:
    sample_coverage_query_songs = df['track_name'].sample(n=num_query_songs_for_coverage, random_state=42).tolist()
    total_catalog_songs = df['track_name'].nunique()

    # Coverage Solusi 1
    coverage_s1, unique_recs_s1 = calculate_recommendation_coverage(
        get_recommendations_on_the_fly,
        processed_features,
        sample_coverage_query_songs,
        total_catalog_songs,
        N_rec=N_recommendations
    )
    print(f"Solusi 1 - Coverage: {coverage_s1:.4f} ({unique_recs_s1} lagu unik direkomendasikan dari {num_query_songs_for_coverage} query)")

    # Coverage Solusi 2
    coverage_s2, unique_recs_s2 = calculate_recommendation_coverage(
        get_recommendations_reranked_by_popularity,
        processed_features_no_pop,
        sample_coverage_query_songs,
        total_catalog_songs,
        N_rec=N_recommendations,
        original_df_for_rerank=df, 
        M_candidates=M_candidates_for_reranking
    )
    print(f"Solusi 2 - Coverage: {coverage_s2:.4f} ({unique_recs_s2} lagu unik direkomendasikan dari {num_query_songs_for_coverage} query)")

else:
    print("Tidak cukup data untuk menghitung estimasi coverage dengan sampel yang diinginkan.")

# %% [markdown]
# Penjelasan Proses: Kode ini menjalankan proses evaluasi kuantitatif. Ia melakukan iterasi melalui sample_test_songs, menghasilkan rekomendasi untuk setiap solusi, lalu memanggil fungsi metrik (ILS dan Novelty) untuk setiap daftar rekomendasi. Skor-skor ini disimpan dalam list, dan pada akhirnya dihitung rata-ratanya untuk mendapatkan satu angka representatif per metrik untuk setiap solusi. Untuk coverage, evaluasi dilakukan pada sampel yang lebih besar (50 lagu) untuk mendapatkan estimasi yang lebih stabil.   
# 
# Penjelasan Hasil: Hasil perhitungan metrik menunjukkan trade-off yang jelas antara kedua solusi:
# * Rata-rata ILS: Solusi 1 (0.9127) > Solusi 2 (0.8782). Ini berarti rekomendasi dari Solusi 2 lebih beragam.
# * Rata-rata Popularitas (Novelty): Solusi 1 (21.80) &lt; Solusi 2 (30.60). Ini berarti Solusi 1 merekomendasikan lagu-lagu yang lebih "novel" (kurang populer), sedangkan Solusi 2 merekomendasikan lagu yang lebih populer.
# * Coverage: Kedua solusi memiliki skor coverage yang hampir identik (0.0017), menunjukkan keduanya memiliki kemampuan yang setara dalam menjangkau katalog lagu berdasarkan sampel uji ini.

# %%
# 3. Visualisasi Hasil Evaluasi

# Data untuk plotting
labels_solutions = ['Solusi 1 (Pop as Feature)', 'Solusi 2 (Pop for Rerank)']

# 1. Grafik Perbandingan Rata-rata Intra-List Similarity (ILS)
if all_ils_s1 and all_ils_s2: # Pastikan list tidak kosong
    mean_ils_values = [np.mean(all_ils_s1), np.mean(all_ils_s2)]
    
    plt.figure(figsize=(8, 6))
    bars_ils = plt.bar(labels_solutions, mean_ils_values, color=['skyblue', 'lightcoral'])
    plt.ylabel('Rata-rata ILS')
    plt.title('Perbandingan Rata-rata Intra-List Similarity (ILS) antar Solusi')
    plt.ylim(0, max(mean_ils_values) * 1.1) # Atur batas y agar bar terlihat baik

    # Tambahkan nilai di atas bar
    for bar in bars_ils:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

    plt.show()
else:
    print("Data ILS tidak cukup untuk membuat grafik perbandingan ILS.")

# %%
# 2. Grafik Perbandingan Rata-rata Novelty (berdasarkan Rata-rata Popularitas)
if all_novelty_s1 and all_novelty_s2: # Pastikan list tidak kosong
    # Ingat: Novelty lebih tinggi jika rata-rata popularitas lebih RENDAH
    mean_novelty_values = [np.mean(all_novelty_s1), np.mean(all_novelty_s2)]
    
    plt.figure(figsize=(8, 6))
    bars_novelty = plt.bar(labels_solutions, mean_novelty_values, color=['skyblue', 'lightcoral'])
    plt.ylabel('Rata-rata Popularitas (Lower is More Novel)')
    plt.title('Perbandingan Rata-rata Popularitas Rekomendasi (Novelty)')
    plt.ylim(0, max(mean_novelty_values) * 1.1)

    # Tambahkan nilai di atas bar
    for bar in bars_novelty:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom') # Disesuaikan untuk skala popularitas

    plt.show()
else:
    print("Data Novelty tidak cukup untuk membuat grafik perbandingan Novelty.")

# %%
# 3. Grafik Perbandingan Estimasi Recommendation Coverage

try:
    if 'coverage_s1' in locals() and 'coverage_s2' in locals() and \
       isinstance(coverage_s1, (int, float)) and isinstance(coverage_s2, (int, float)): # Pemeriksaan tambahan
        
        coverage_values = [coverage_s1, coverage_s2]
        labels_solutions = ['Solusi 1 (Pop as Feature)', 'Solusi 2 (Pop for Rerank)'] # Pastikan ini didefinisikan

        plt.figure(figsize=(8, 6))
        bars_coverage = plt.bar(labels_solutions, coverage_values, color=['skyblue', 'lightcoral'])
        plt.ylabel('Skor Recommendation Coverage')
        plt.title('Perbandingan Estimasi Recommendation Coverage')
        
        max_coverage_val = max(coverage_values) if coverage_values and max(coverage_values) > 0 else 0.002 # Handle jika coverage_values kosong atau semua 0
        plt.ylim(0, max_coverage_val * 1.5 if max_coverage_val > 0 else 0.003)

        for bar in bars_coverage:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max_coverage_val*0.05 if max_coverage_val > 0 else 0.0001), f'{yval:.4f}', ha='center', va='bottom')

        plt.show()
    else:
        print("Variabel coverage_s1 dan/atau coverage_s2 tidak ditemukan atau bukan numerik.")
        print("Pastikan sudah dihitung dan disimpan dengan benar dari fungsi calculate_recommendation_coverage.")
        print("Contoh: coverage_s1_score, _ = calculate_recommendation_coverage(...) lalu gunakan coverage_s1_score")

except Exception as e:
    print(f"Terjadi error saat membuat grafik coverage: {e}")

# %% [markdown]
# Penjelasan Proses: Tahap terakhir adalah memvisualisasikan hasil metrik yang telah dihitung menggunakan diagram batang. Visualisasi ini bertujuan untuk menyajikan perbandingan performa antara Solusi 1 dan Solusi 2 secara jelas dan mudah dipahami. Tiga grafik dibuat, masing-masing untuk metrik ILS, Novelty (rata-rata popularitas), dan Coverage.   
# 
# Penjelasan Hasil/Insight: Grafik-grafik ini mengonfirmasi temuan dari analisis kuantitatif dan memberikan kesimpulan akhir yang kuat:
# * Grafik ILS: Batang biru (Solusi 1) yang lebih tinggi menunjukkan bahwa memasukkan popularitas sebagai fitur konten menghasilkan rekomendasi yang kurang beragam. Sebaliknya, Solusi 2 (merah) lebih baik dalam memberikan keragaman.
# * Grafik Novelty: Batang biru (Solusi 1) yang lebih rendah menunjukkan bahwa Solusi 1 lebih unggul dalam menemukan item-item niche atau kurang populer. Sebaliknya, Solusi 2 (merah) cenderung merekomendasikan lagu-lagu yang lebih mainstream.
# * Grafik Coverage: Kedua batang memiliki tinggi yang sama, mengindikasikan tidak ada perbedaan signifikan dalam jangkauan katalog antara kedua pendekatan.


