import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from matplotlib.colors import ListedColormap

# Load data
kec = pd.read_csv("jumlah_keluarga_miskin_kecamatan_ambon_2014.csv")

# Ambil fitur yang akan digunakan untuk klasterisasi
X = kec[['Penduduk Total Jumlah Jiwa', 'Penduduk Miskin Jumlah Jiwa']].values

# Inisialisasi dan jalankan KMeans
kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Evaluasi klaster
inertia = kmeans.inertia_
sil_score = silhouette_score(X, labels)
db_score = davies_bouldin_score(X, labels)

# Cetak hasil ke terminal
print("\n=== Evaluasi Klaster KMeans ===")
print(f"Inertia: {inertia:.2f} (lebih kecil lebih baik)")
print(f"Silhouette Score: {sil_score:.2f} (semakin mendekati 1 lebih baik)")
print(f"Davies-Bouldin Index: {db_score:.2f} (lebih kecil lebih baik)")

# Visualisasi klaster dan boundary
x_min, x_max = X[:, 0].min() - 50, X[:, 0].max() + 50
y_min, y_max = X[:, 1].min() - 50, X[:, 1].max() + 50
xx, yy = np.meshgrid(np.arange(x_min, x_max, 2),
                     np.arange(y_min, y_max, 2))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap = ListedColormap(['#99ff99', '#ff9999', '#9999ff'])
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.4)

# Plot data
colors = ['green', 'red', 'blue']
for i in range(3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f"Klaster {i}", s=50, c=colors[i])

# Plot centroid
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=300, c='black', label='Centroid')

# Tambahkan label nama kecamatan
for i, txt in enumerate(kec['Kecamatan']):
    plt.annotate(txt, (X[i, 0], X[i, 1]), fontsize=9)

plt.xlabel('Jumlah Penduduk Total')
plt.ylabel('Jumlah Penduduk Miskin')
plt.title('Klasterisasi Kecamatan di Ambon Berdasarkan Kemiskinan (K-Means)')
plt.legend()

# Tambahkan info evaluasi dalam kotak
info = f"Inertia: {inertia:.2f} (lebih kecil lebih baik)\n"
info += f"Silhouette: {sil_score:.2f} (baik)\n"
info += f"Davies-Bouldin: {db_score:.2f} (lebih kecil lebih baik)"
plt.text(x_max - 300, y_min + 50, info, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()
