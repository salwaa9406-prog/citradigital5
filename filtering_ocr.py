import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import os
import glob
import re


# ============================================================
# KONFIGURASI
# ============================================================

FOLDER_CITRA = "citra"
FOLDER_HASIL = "hasil_filtering"

os.makedirs(FOLDER_HASIL, exist_ok=True)


# ============================================================
# GROUND TRUTH
# ============================================================
# Nomor ijazah yang menjadi jawaban sebenarnya.
# Sesuaikan jika nomor pada citra kamu berbeda.

GROUND_TRUTH = "571012022000056"


# ============================================================
# CROP AREA NOMOR IJAZAH
# ============================================================
# Gambar diputar 90 derajat searah jarum jam terlebih dahulu.

X1 = 0.04
Y1 = 0.87
X2 = 0.35
Y2 = 0.98


# ============================================================
# FUNGSI CROP NOMOR IJAZAH
# ============================================================

def crop_nomor_ijazah(img):

    tinggi, lebar = img.shape[:2]

    x1 = int(X1 * lebar)
    y1 = int(Y1 * tinggi)

    x2 = int(X2 * lebar)
    y2 = int(Y2 * tinggi)

    crop = img[y1:y2, x1:x2]

    return crop


# ============================================================
# 1. MEAN FILTER
# ============================================================

def mean_filter(gray):

    hasil = cv2.blur(gray, (3, 3))

    return hasil


# ============================================================
# 2. MEDIAN FILTER
# ============================================================

def median_filter(gray):

    hasil = cv2.medianBlur(gray, 3)

    return hasil


# ============================================================
# 3. GAUSSIAN FILTER
# ============================================================

def gaussian_filter(gray):

    hasil = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    return hasil


# ============================================================
# 4. SHARPENING
# ============================================================

def sharpening(gray):

    # Kernel sharpening
    kernel = np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ])

    hasil = cv2.filter2D(
        gray,
        -1,
        kernel
    )

    return hasil


# ============================================================
# NORMALISASI HASIL OCR
# ============================================================

def bersihkan_ocr(teks):

    # Ambil hanya angka
    teks = re.sub(r'[^0-9]', '', teks)

    return teks


# ============================================================
# OCR
# ============================================================

def lakukan_ocr(image):

    # Perbesar gambar agar karakter lebih jelas
    image = cv2.resize(
        image,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    teks = pytesseract.image_to_string(
        image,
        lang="ind",
        config="--psm 7 -c tessedit_char_whitelist=0123456789"
    )

    teks = bersihkan_ocr(teks)

    return teks


# ============================================================
# HITUNG KARAKTER BENAR
# ============================================================

def hitung_karakter_benar(hasil_ocr, ground_truth):

    jumlah_benar = 0

    panjang_min = min(
        len(hasil_ocr),
        len(ground_truth)
    )

    for i in range(panjang_min):

        if hasil_ocr[i] == ground_truth[i]:

            jumlah_benar += 1

    return jumlah_benar


# ============================================================
# HITUNG AKURASI
# ============================================================

def hitung_akurasi(hasil_ocr, ground_truth):

    karakter_benar = hitung_karakter_benar(
        hasil_ocr,
        ground_truth
    )

    akurasi = (
        karakter_benar /
        len(ground_truth)
    ) * 100

    return karakter_benar, akurasi


# ============================================================
# PROSES SEMUA GAMBAR
# ============================================================

files = []

files += glob.glob(
    os.path.join(FOLDER_CITRA, "*.jpg")
)

files += glob.glob(
    os.path.join(FOLDER_CITRA, "*.jpeg")
)

files += glob.glob(
    os.path.join(FOLDER_CITRA, "*.png")
)

files += glob.glob(
    os.path.join(FOLDER_CITRA, "*.JPG")
)

files += glob.glob(
    os.path.join(FOLDER_CITRA, "*.PNG")
)

files.sort()


print("=" * 70)
print("PROGRAM FILTERING, NOISE REDUCTION, SHARPENING DAN OCR")
print("=" * 70)

print("Jumlah gambar:", len(files))
print("Ground Truth :", GROUND_TRUTH)


if len(files) == 0:

    print("Tidak ada gambar di folder citra!")

    exit()


# ============================================================
# REKAP HASIL
# ============================================================

rekap = []


# ============================================================
# LOOP SEMUA GAMBAR
# ============================================================

for nomor, file in enumerate(files, start=1):

    print("\n")
    print("=" * 70)
    print(f"GAMBAR {nomor}: {file}")
    print("=" * 70)


    # --------------------------------------------------------
    # BACA GAMBAR
    # --------------------------------------------------------

    img = cv2.imread(file)

    if img is None:

        print("Gagal membaca gambar!")

        continue


    # --------------------------------------------------------
    # ROTASI
    # --------------------------------------------------------

    img_rotated = cv2.rotate(
        img,
        cv2.ROTATE_90_CLOCKWISE
    )


    # --------------------------------------------------------
    # CROP NOMOR IJAZAH
    # --------------------------------------------------------

    roi = crop_nomor_ijazah(
        img_rotated
    )


    if roi.size == 0:

        print("Crop kosong!")

        continue


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # FILTERING
    # ========================================================

    mean = mean_filter(gray)

    median = median_filter(gray)

    gaussian = gaussian_filter(gray)

    sharp = sharpening(gray)


    # ========================================================
    # OCR
    # ========================================================

    ocr_original = lakukan_ocr(gray)

    ocr_mean = lakukan_ocr(mean)

    ocr_median = lakukan_ocr(median)

    ocr_gaussian = lakukan_ocr(gaussian)

    ocr_sharp = lakukan_ocr(sharp)


    # ========================================================
    # AKURASI OCR
    # ========================================================

    benar_original, akurasi_original = hitung_akurasi(
        ocr_original,
        GROUND_TRUTH
    )

    benar_mean, akurasi_mean = hitung_akurasi(
        ocr_mean,
        GROUND_TRUTH
    )

    benar_median, akurasi_median = hitung_akurasi(
        ocr_median,
        GROUND_TRUTH
    )

    benar_gaussian, akurasi_gaussian = hitung_akurasi(
        ocr_gaussian,
        GROUND_TRUTH
    )

    benar_sharp, akurasi_sharp = hitung_akurasi(
        ocr_sharp,
        GROUND_TRUTH
    )


    # ========================================================
    # NAMA FILE
    # ========================================================

    nama_file = os.path.splitext(
        os.path.basename(file)
    )[0]


    folder_gambar = os.path.join(
        FOLDER_HASIL,
        nama_file
    )

    os.makedirs(
        folder_gambar,
        exist_ok=True
    )


    # ========================================================
    # SIMPAN HASIL CITRA
    # ========================================================

    cv2.imwrite(
        os.path.join(
            folder_gambar,
            "01_original.png"
        ),
        gray
    )

    cv2.imwrite(
        os.path.join(
            folder_gambar,
            "02_mean_filter.png"
        ),
        mean
    )

    cv2.imwrite(
        os.path.join(
            folder_gambar,
            "03_median_filter.png"
        ),
        median
    )

    cv2.imwrite(
        os.path.join(
            folder_gambar,
            "04_gaussian_filter.png"
        ),
        gaussian
    )

    cv2.imwrite(
        os.path.join(
            folder_gambar,
            "05_sharpening.png"
        ),
        sharp
    )


    # ========================================================
    # TAMPILKAN HASIL OCR
    # ========================================================

    print("\nHASIL OCR:")

    print(
        "Original       :",
        ocr_original,
        "|",
        akurasi_original,
        "%"
    )

    print(
        "Mean Filter    :",
        ocr_mean,
        "|",
        akurasi_mean,
        "%"
    )

    print(
        "Median Filter  :",
        ocr_median,
        "|",
        akurasi_median,
        "%"
    )

    print(
        "Gaussian Filter:",
        ocr_gaussian,
        "|",
        akurasi_gaussian,
        "%"
    )

    print(
        "Sharpening     :",
        ocr_sharp,
        "|",
        akurasi_sharp,
        "%"
    )


    # ========================================================
    # SIMPAN HASIL OCR KE TXT
    # ========================================================

    with open(
        os.path.join(
            folder_gambar,
            "hasil_ocr.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "GROUND TRUTH: "
            + GROUND_TRUTH
            + "\n\n"
        )

        f.write(
            "Original:\n"
            + ocr_original
            + "\n"
        )

        f.write(
            "Karakter benar: "
            + str(benar_original)
            + "\n"
        )

        f.write(
            "Akurasi: "
            + f"{akurasi_original:.2f}%"
            + "\n\n"
        )


        f.write(
            "Mean Filter:\n"
            + ocr_mean
            + "\n"
        )

        f.write(
            "Karakter benar: "
            + str(benar_mean)
            + "\n"
        )

        f.write(
            "Akurasi: "
            + f"{akurasi_mean:.2f}%"
            + "\n\n"
        )


        f.write(
            "Median Filter:\n"
            + ocr_median
            + "\n"
        )

        f.write(
            "Karakter benar: "
            + str(benar_median)
            + "\n"
        )

        f.write(
            "Akurasi: "
            + f"{akurasi_median:.2f}%"
            + "\n\n"
        )


        f.write(
            "Gaussian Filter:\n"
            + ocr_gaussian
            + "\n"
        )

        f.write(
            "Karakter benar: "
            + str(benar_gaussian)
            + "\n"
        )

        f.write(
            "Akurasi: "
            + f"{akurasi_gaussian:.2f}%"
            + "\n\n"
        )


        f.write(
            "Sharpening:\n"
            + ocr_sharp
            + "\n"
        )

        f.write(
            "Karakter benar: "
            + str(benar_sharp)
            + "\n"
        )

        f.write(
            "Akurasi: "
            + f"{akurasi_sharp:.2f}%"
            + "\n"
        )


    # ========================================================
    # TABEL HASIL UNTUK GAMBAR INI
    # ========================================================

    hasil_tabel = [
        [
            "Original",
            ocr_original,
            benar_original,
            akurasi_original
        ],

        [
            "Mean Filter",
            ocr_mean,
            benar_mean,
            akurasi_mean
        ],

        [
            "Median Filter",
            ocr_median,
            benar_median,
            akurasi_median
        ],

        [
            "Gaussian Filter",
            ocr_gaussian,
            benar_gaussian,
            akurasi_gaussian
        ],

        [
            "Sharpening",
            ocr_sharp,
            benar_sharp,
            akurasi_sharp
        ]
    ]


    # ========================================================
    # TAMPILKAN PERBANDINGAN CITRA
    # ========================================================

    plt.figure(figsize=(15, 8))


    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Original")
    plt.axis("off")


    plt.subplot(2, 3, 2)
    plt.imshow(mean, cmap="gray")
    plt.title("Mean Filter")
    plt.axis("off")


    plt.subplot(2, 3, 3)
    plt.imshow(median, cmap="gray")
    plt.title("Median Filter")
    plt.axis("off")


    plt.subplot(2, 3, 4)
    plt.imshow(gaussian, cmap="gray")
    plt.title("Gaussian Filter")
    plt.axis("off")


    plt.subplot(2, 3, 5)
    plt.imshow(sharp, cmap="gray")
    plt.title("Sharpening")
    plt.axis("off")


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            folder_gambar,
            "perbandingan_filter.png"
        ),
        dpi=300
    )

    plt.close()


    # ========================================================
    # MASUKKAN KE REKAP
    # ========================================================

    rekap.append([
        nama_file,
        ocr_original,
        benar_original,
        akurasi_original,
        ocr_mean,
        benar_mean,
        akurasi_mean,
        ocr_median,
        benar_median,
        akurasi_median,
        ocr_gaussian,
        benar_gaussian,
        akurasi_gaussian,
        ocr_sharp,
        benar_sharp,
        akurasi_sharp
    ])


# ============================================================
# REKAP SEMUA GAMBAR
# ============================================================

with open(
    os.path.join(
        FOLDER_HASIL,
        "REKAP_OCR.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "GROUND TRUTH: "
        + GROUND_TRUTH
        + "\n"
    )

    f.write("=" * 100 + "\n")


    for data in rekap:

        nama = data[0]

        f.write(
            "\nGAMBAR: "
            + nama
            + "\n"
        )

        f.write("=" * 100 + "\n")


        f.write(
            f"Original       : {data[1]} | "
            f"Benar: {data[2]} | "
            f"Akurasi: {data[3]:.2f}%\n"
        )

        f.write(
            f"Mean Filter    : {data[4]} | "
            f"Benar: {data[5]} | "
            f"Akurasi: {data[6]:.2f}%\n"
        )

        f.write(
            f"Median Filter  : {data[7]} | "
            f"Benar: {data[8]} | "
            f"Akurasi: {data[9]:.2f}%\n"
        )

        f.write(
            f"Gaussian Filter: {data[10]} | "
            f"Benar: {data[11]} | "
            f"Akurasi: {data[12]:.2f}%\n"
        )

        f.write(
            f"Sharpening     : {data[13]} | "
            f"Benar: {data[14]} | "
            f"Akurasi: {data[15]:.2f}%\n"
        )


print("\n")
print("=" * 70)
print("SELESAI!")
print("=" * 70)

print(
    "Semua hasil tersimpan di folder:",
    FOLDER_HASIL
)