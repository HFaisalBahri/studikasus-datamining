# studikasus-datamining
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_excel("dataKasus-1.xlsx")
data.head()
data.info()
data.head()
data.describe()
print(data['USIA'].unique())
data.replace({
    'RIW HIPERTENSI': {'Tidak': 0, 'Ya': 1},
    'RIW DM': {'Tidak': 0, 'Ya': 1},
    'RIW HIPERTENSI/PE DALAM KELUARGA': {'Tidak': 0, 'Ada': 1},
    'PE/Non PE': {'Non PE': 0, 'PE': 1}
}, inplace=True)
# Memastikan kolom tidak ada spasi
data.columns = data.columns.str.strip()

# Memisahkan fitur (X) dan target (y)
X = data[['USIA', 'PARITAS', 'JARAK KELAHIRAN', 'RIW HIPERTENSI', 
           'RIW DM', 'RIW HIPERTENSI/PE DALAM KELUARGA', 'SOSEK RENDAH']]
y = data['PE/Non PE']

# Memeriksa kolom yang ada dalam X sebelum One-Hot Encoding
print("Kolom dalam X sebelum One-Hot Encoding:")
print(X.columns.tolist())

# Melakukan One-Hot Encoding untuk fitur kategorikal
X = pd.get_dummies(X, columns=['JARAK KELAHIRAN', 'SOSEK RENDAH'], drop_first=True)

# Menampilkan hasil setelah One-Hot Encoding
print("\nKolom dalam X setelah One-Hot Encoding:")
print(X.columns.tolist())
# Membagi data menjadi data latih dan data uji
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Mengimpor library yang diperlukan
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Menggunakan ColumnTransformer untuk preprocessing
# StandardScaler untuk kolom numerik dan OneHotEncoder untuk kolom kategorikal
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['USIA', 'PARITAS', 'RIW HIPERTENSI', 
                                    'RIW DM', 'RIW HIPERTENSI/PE DALAM KELUARGA']),
        ('cat', OneHotEncoder(drop='first'), ['JARAK KELAHIRAN', 'SOSEK RENDAH'])
    ])

# Setelah mengimpor, Anda dapat melanjutkan preprocessing data

