import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Membaca dataset
df_kecamatan = pd.read_csv('jumlah_keluarga_miskin_kecamatan_ambon_2014.csv')
df_kota = pd.read_csv('jumlah_keluarga_miskin_ambon_2010_2014.csv')

# 2. Visualisasi Data Kemiskinan per Kecamatan 2014
plt.figure(figsize=(10, 6))
bars = plt.bar(df_kecamatan['Kecamatan'], df_kecamatan['Penduduk Miskin Jumlah Jiwa'], color='skyblue')
plt.title('Jumlah Penduduk Miskin per Kecamatan (2014)')
plt.ylabel('Jumlah Jiwa')
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:,}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# 3. Pie Chart Persentase Kemiskinan
plt.figure(figsize=(8, 8))
df_kecamatan['Persentase Miskin'] = (df_kecamatan['Penduduk Miskin Jumlah Jiwa'] / 
                                     df_kecamatan['Penduduk Total Jumlah Jiwa']) * 100
plt.pie(df_kecamatan['Persentase Miskin'], 
        labels=df_kecamatan['Kecamatan'], 
        autopct='%1.1f%%', 
        startangle=140,
        colors=['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'violet'])
plt.title('Persentase Penduduk Miskin per Kecamatan (2014)')
plt.tight_layout()
plt.show()

# 4. Prediksi Kemiskinan Kota Ambon
plt.figure(figsize=(12, 6))
tahun = df_kota['Tahun'].values.reshape(-1, 1)
penduduk_miskin = df_kota['Penduduk Miskin Jumlah Jiwa'].values

model = LinearRegression()
model.fit(tahun, penduduk_miskin)

tahun_prediksi = np.array([2015, 2016, 2017, 2018, 2019]).reshape(-1, 1)
prediksi = model.predict(tahun_prediksi)

plt.plot(df_kota['Tahun'], df_kota['Penduduk Miskin Jumlah Jiwa'], 'bo-', label='Data Aktual (2010-2014)')
plt.plot(tahun_prediksi, prediksi, 'ro--', label='Prediksi (2015-2019)')
for t, p in zip(tahun_prediksi.flatten(), prediksi):
    plt.text(t, p, f'{p:.0f}', ha='center', va='bottom')

plt.title('Tren dan Prediksi Kemiskinan Kota Ambon (2010-2019)')
plt.xlabel('Tahun')
plt.ylabel('Jumlah Penduduk Miskin')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(2010, 2020, 1))
plt.tight_layout()
plt.show()

# 4b. Evaluasi Model Regresi
pred_train = model.predict(tahun)
r2 = r2_score(penduduk_miskin, pred_train)
mae = mean_absolute_error(penduduk_miskin, pred_train)
rmse = np.sqrt(mean_squared_error(penduduk_miskin, pred_train))

# Cetak ke TERMINAL
print("\n=== Evaluasi Model Regresi Linear ===")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Tampilkan juga dalam FIGURE
plt.figure(figsize=(6, 4))
plt.axis('off')
plt.title('Evaluasi Model Regresi Linear', fontsize=14, fontweight='bold')
textstr = f'R² Score: {r2:.4f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}'
plt.text(0.5, 0.5, textstr, fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightgrey', alpha=0.7))
plt.tight_layout()
plt.show()

# 5. Tabel Hasil Prediksi
tabel_prediksi = pd.DataFrame({
    'Tahun': tahun_prediksi.flatten(),
    'Prediksi Jumlah Penduduk Miskin': prediksi.round(),
    'Perubahan (%)': ((prediksi - penduduk_miskin[-1]) / penduduk_miskin[-1] * 100).round(1)
})
print("\nHasil Prediksi Jumlah Penduduk Miskin 2015-2019:")
print(tabel_prediksi.to_string(index=False))

# 6. Perbandingan Total vs Miskin
plt.figure(figsize=(12, 6))
width = 0.35
x = range(len(df_kecamatan))

bars1 = plt.bar(x, df_kecamatan['Penduduk Total Jumlah Jiwa'], width, label='Total Penduduk')
bars2 = plt.bar([i + width for i in x], df_kecamatan['Penduduk Miskin Jumlah Jiwa'], width, label='Penduduk Miskin')

plt.title('Perbandingan Total Penduduk vs Penduduk Miskin per Kecamatan (2014)')
plt.xlabel('Kecamatan')
plt.ylabel('Jumlah Jiwa')
plt.xticks([i + width/2 for i in x], df_kecamatan['Kecamatan'], rotation=45)
plt.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:,}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
