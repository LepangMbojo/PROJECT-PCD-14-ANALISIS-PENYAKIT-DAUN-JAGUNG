# PROJECT PCD - Klasifikasi CItra Daun Jagung Untuk Deteksi Penyakit Hawar, Karat, dan Daun Sehat

## Nama Anggota:
### - M. Khalid Al Rejeki (F1D02310122)  
### - Roman Ibrahima Archisappe Chareni (F1D02310024)  
### - Izzat Nazhiefa (F1D02310114)  
### - Alifah Rizki Saputri (F1D02310103)

---

# Project Overview

Pada project PCD ini, kami melakukan eksperimen klasifikasi citra menggunakan dataset yang telah kami siapkan. Tujuan utama dari project ini adalah:

- Mengimplementasikan teknik pengolahan citra digital dalam proses klasifikasi.
- Memilih tahapan preprocessing yang sesuai dengan karakteristik data.

Kami melakukan tiga kali percobaan dengan kombinasi preprocessing yang berbeda. Percobaan dilakukan dengan penambahan jumlah preprocessing secara bertahap. Selain itu, kami juga melakukan ekstraksi fitur menggunakan metode **GLCM (Gray-Level Co-occurrence Matrix)** serta membandingkan performa model klasifikasi: **Random Forest, SVM, dan KNN** dan **Preprosesing**.

---

# Import Library

```python
import os, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import entropy
