import cv2
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import math

def apply_dct(block):
    block_float = np.float32(block)
    shifted_block = block_float - 128
    return dct(dct(shifted_block, axis=0, norm='ortho'), axis=1, norm='ortho')

def quantize(dct_block, quantization_table):
    return np.round(dct_block / quantization_table)

def dequantize(quantized_block, quantization_table):
    return quantized_block * quantization_table

def apply_idct(dequantized_block):
    idct_block = idct(idct(dequantized_block, axis=0, norm='ortho'), axis=1, norm='ortho')
    return np.uint8(np.clip(idct_block + 128, 0, 255))

def calculate_psnr(img1, img2):
    """Menghitung PSNR antara dua gambar."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def main():
    # --- PENGATURAN ---
    image_path = 'selfie.jpg' 
    block_size = 8
    
    quality = 25

    # --- MEMBUAT TABEL KUANTISASI BERDASARKAN KUALITAS ---
    base_quant_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)

    if quality < 50:
        scale_factor = 5000 / quality
    else:
        scale_factor = 200 - 2 * quality
    
    quantization_table = (base_quant_table * scale_factor + 50) / 100
    quantization_table = np.clip(quantization_table, 1, 255) 

    # --- PROSES GAMBAR ---
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Gagal memuat gambar di '{image_path}'")
        return

    gray_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    h, w = gray_image.shape
    new_h = (h // block_size) * block_size
    new_w = (w // block_size) * block_size
    cropped_original_image = gray_image[:new_h, :new_w]

    decoded_image = np.zeros_like(cropped_original_image, dtype=np.uint8)

    print(f"Memproses gambar dengan ukuran {new_w}x{new_h}...")
    print(f"Faktor Kualitas: {quality}")

    for y in range(0, new_h, block_size):
        for x in range(0, new_w, block_size):
            original_block = cropped_original_image[y:y+block_size, x:x+block_size]
            
            # --- Proses Encoding & Decoding per blok ---
            dct_block = apply_dct(original_block)
            quantized_block = quantize(dct_block, quantization_table)
            dequantized_block = dequantize(quantized_block, quantization_table)
            decoded_block = apply_idct(dequantized_block)
            
            decoded_image[y:y+block_size, x:x+block_size] = decoded_block

    # --- ANALISIS HASIL ---
    psnr_value = calculate_psnr(cropped_original_image, decoded_image)
    print(f"\nProses Selesai.")
    print(f"PSNR (Peak Signal-to-Noise Ratio): {psnr_value:.2f} dB")
    
    # --- VISUALISASI ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(cropped_original_image, cmap='gray')
    axes[0].set_title('Gambar Asli (Grayscale)', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(decoded_image, cmap='gray')
    axes[1].set_title(f'Gambar Hasil Decoding (Kualitas: {quality})\nPSNR: {psnr_value:.2f} dB', fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()