import cv2
import numpy as np
import os

# Pastikan folder output ada
os.makedirs('output', exist_ok=True)

# ==============================
# 1. Membuat karakter babi lucu 
# ==============================
canvas = np.full((500, 500, 3), 255, dtype=np.uint8)

# Wajah babi (lingkaran pink)
cv2.circle(canvas, (250, 250), 120, (180, 105, 255), -1)

# Hidung (oval)
cv2.ellipse(canvas, (250, 270), (50, 30), 0, 0, 360, (150, 70, 200), -1)

# Lubang hidung
cv2.circle(canvas, (230, 270), 8, (90, 40, 120), -1)
cv2.circle(canvas, (270, 270), 8, (90, 40, 120), -1)

# Mata
cv2.circle(canvas, (210, 210), 15, (0, 0, 0), -1)
cv2.circle(canvas, (290, 210), 15, (0, 0, 0), -1)

# Telinga kiri
pts_left = np.array([[170, 120], [190, 180], [130, 160]], np.int32)
cv2.fillPoly(canvas, [pts_left], (180, 105, 255))

# Telinga kanan
pts_right = np.array([[330, 120], [310, 180], [370, 160]], np.int32)
cv2.fillPoly(canvas, [pts_right], (180, 105, 255))

# Mulut (garis senyum)
cv2.ellipse(canvas, (250, 310), (40, 20), 0, 0, 180, (0, 0, 0), 2)

# Teks label
cv2.putText(canvas, "si lucu", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 70, 200), 3)

# Simpan karakter asli
cv2.imwrite('output/karakter.png', canvas)

# ==============================
# 2. Transformasi Gambar
# ==============================
rows, cols = canvas.shape[:2]

# a. Translasi (geser posisi)
M_translate = np.float32([[1, 0, 50], [0, 1, 70]])  # geser kanan & bawah
translated = cv2.warpAffine(canvas, M_translate, (cols, rows))
cv2.imwrite('output/translate.png', translated)

# b. Rotasi (putar 30 derajat)
M_rotate = cv2.getRotationMatrix2D((cols // 2, rows // 2), 30, 1)
rotated = cv2.warpAffine(canvas, M_rotate, (cols, rows))
cv2.imwrite('output/rotate.png', rotated)

# c. Resize (ubah ukuran jadi kecil)
resized = cv2.resize(canvas, (250, 250))
cv2.imwrite('output/resize.png', resized)

# d. Crop (ambil sebagian wajah)
cropped = canvas[150:350, 150:350]
cv2.imwrite('output/crop.png', cropped)

# ==============================
# 3. Operasi Aritmatika & Bitwise
# ==============================
# a. Tambahkan background hijau muda
bg = np.full((500, 500, 3), (130, 220, 160), dtype=np.uint8)
combined = cv2.addWeighted(canvas, 0.8, bg, 0.2, 0)
cv2.imwrite('output/add.png', combined)

# b. Bitwise efek
bit_not = cv2.bitwise_not(canvas)
cv2.imwrite('output/bitwise_not.png', bit_not)

bit_and = cv2.bitwise_and(canvas, bg)
cv2.imwrite('output/bitwise.png', bit_and)

# ==============================
# 4. Gabungkan hasil akhir
# ==============================
final = cv2.addWeighted(rotated, 0.6, bit_and, 0.4, 0)
cv2.imwrite('output/final.png', final)

# ==============================
# 5. Tampilkan hasil (opsional)
# ==============================
cv2.imshow('Karakter Babi', canvas)
cv2.imshow('Translate', translated)
cv2.imshow('Rotate', rotated)
cv2.imshow('Resize', resized)
cv2.imshow('Crop', cropped)
cv2.imshow('Bitwise', bit_and)
cv2.imshow('Final', final)

cv2.waitKey(0)
cv2.destroyAllWindows()
