import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# --- PENGATURAN AWAL ---
SAMPLE_RATE = 48000  # Sample rate dalam Hz (sesuai soal)
DURATION = 5         # Durasi sinyal dalam detik
FREQUENCY = 440      # Frekuensi sinyal sinusoida dalam Hz (nada A4)
HOST_AMP = 0.5       # Amplitudo sinyal asli (0.0 - 1.0)

# --- (!!) MASUKKAN INFORMASI ANDA DI SINI (!!) ---
BIRTH_MONTH = 6  
BIRTH_DAY = 26   

# Variasi bobot (weight/alpha) untuk watermarking (sesuai soal)
ALPHA_1 = 0.02   # Bobot yang lebih kuat
ALPHA_2 = 0.005  # Bobot yang lebih lemah

def generate_sine_wave(freq, duration, sample_rate, amp):
    """Menghasilkan sinyal sinusoida murni."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amp * np.sin(2 * np.pi * freq * t)
    return signal

def generate_watermark(seed, length):
    """Menghasilkan watermark berbasis spread spectrum (noise acak)."""
    # Atur seed agar watermark selalu sama jika kuncinya sama
    np.random.seed(seed)
    # Hasilkan noise acak dari distribusi normal
    watermark = np.random.randn(length)
    # Normalisasi watermark
    watermark /= np.max(np.abs(watermark))
    return watermark

def embed_watermark(host_signal, watermark, alpha):
    """Menyisipkan watermark ke dalam sinyal asli."""
    # Proses embedding: S_w = S + alpha * W
    watermarked_signal = host_signal + (alpha * watermark)
    # Normalisasi kembali untuk mencegah clipping
    watermarked_signal /= np.max(np.abs(watermarked_signal))
    return watermarked_signal

def detect_watermark(signal, watermark_key, length):
    """Mendeteksi kehadiran watermark dengan menghitung korelasi."""
    # Hasilkan kembali watermark yang sama persis menggunakan kunci yang sama
    original_watermark = generate_watermark(watermark_key, length)
    # Hitung korelasi (dot product) antara sinyal dan watermark
    correlation = np.dot(signal, original_watermark)
    return correlation

def play_audio(signal, sample_rate, text):
    """Memainkan sinyal audio."""
    print(f"\n(b) Memainkan suara: {text}...")
    # Normalisasi ke float32 agar bisa dimainkan sounddevice
    audio_to_play = signal.astype(np.float32)
    sd.play(audio_to_play, sample_rate)
    sd.wait() # Tunggu hingga selesai
    print("Selesai memainkan.")
    
def save_wav(filename, signal, sample_rate):
    """Menyimpan sinyal sebagai file WAV 16-bit."""
    # Konversi ke integer 16-bit
    scaled_signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    write(filename, sample_rate, scaled_signal)

# --- MAIN SCRIPT ---

# 1. Buat Kunci/Seed Watermark
watermark_seed = BIRTH_MONTH * 100 + BIRTH_DAY
print(f"Menggunakan seed untuk watermark: {watermark_seed}")

# 2. Hasilkan Sinyal Asli (Host)
host_signal = generate_sine_wave(FREQUENCY, DURATION, SAMPLE_RATE, HOST_AMP)
signal_length = len(host_signal)

# 3. Hasilkan Sinyal Watermark
watermark_signal = generate_watermark(watermark_seed, signal_length)

# 4. Proses Embedding dengan 2 bobot berbeda
watermarked_signal_1 = embed_watermark(host_signal, watermark_signal, ALPHA_1)
watermarked_signal_2 = embed_watermark(host_signal, watermark_signal, ALPHA_2)

# 5. Simpan file audio untuk dimainkan
save_wav("original_audio.wav", host_signal, SAMPLE_RATE)
save_wav(f"watermarked_audio_alpha_{ALPHA_1}.wav", watermarked_signal_1, SAMPLE_RATE)
save_wav(f"watermarked_audio_alpha_{ALPHA_2}.wav", watermarked_signal_2, SAMPLE_RATE)

# --- (a) Tampilkan Grafik ---
print("\n(a) Menampilkan grafik...")
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Visualisasi Sinyal Audio dan Watermark', fontsize=16)

# Zoom ke sebagian kecil sinyal agar gelombangnya terlihat
zoom_range = slice(0, int(SAMPLE_RATE * 0.01)) # Tampilkan 0.01 detik pertama

axs[0].plot(host_signal[zoom_range])
axs[0].set_title('Sinyal Asli (Sine 440 Hz)')
axs[0].set_ylabel('Amplitudo')
axs[0].grid(True)

axs[1].plot(watermark_signal[zoom_range])
axs[1].set_title(f'Sinyal Watermark (dihasilkan dari seed {watermark_seed})')
axs[1].set_ylabel('Amplitudo')
axs[1].grid(True)

axs[2].plot(watermarked_signal_1[zoom_range])
axs[2].set_title(f'Sinyal Ter-watermark (Bobot/Alpha = {ALPHA_1})')
axs[2].set_ylabel('Amplitudo')
axs[2].set_xlabel('Sampel')
axs[2].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- (b) Mainkan Suara ---
play_audio(host_signal, SAMPLE_RATE, "Sinyal Asli (Sebelum Watermarking)")
play_audio(watermarked_signal_1, SAMPLE_RATE, f"Sinyal Ter-watermark (Bobot = {ALPHA_1})")
# Anda bisa uncomment baris di bawah untuk mendengar efek bobot yang lebih lemah
# play_audio(watermarked_signal_2, SAMPLE_RATE, f"Sinyal Ter-watermark (Bobot = {ALPHA_2})")


# --- (c) Proses Deteksi dan Jelaskan Efeknya ---
print("\n(c) Melakukan deteksi watermark...")

# Tes 1: Deteksi pada sinyal ASLI (seharusnya hasilnya rendah)
correlation_original = detect_watermark(host_signal, watermark_seed, signal_length)

# Tes 2: Deteksi pada sinyal ter-watermark dengan bobot kuat
correlation_wm_1 = detect_watermark(watermarked_signal_1, watermark_seed, signal_length)

# Tes 3: Deteksi pada sinyal ter-watermark dengan bobot lemah
correlation_wm_2 = detect_watermark(watermarked_signal_2, watermark_seed, signal_length)

print("\n--- Hasil Deteksi (Skor Korelasi) ---")
print(f"Skor deteksi pada sinyal ASLI                    : {correlation_original:.4f}")
print(f"Skor deteksi pada sinyal ter-watermark (Bobot={ALPHA_1}): {correlation_wm_1:.4f}")
print(f"Skor deteksi pada sinyal ter-watermark (Bobot={ALPHA_2}): {correlation_wm_2:.4f}")