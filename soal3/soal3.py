import os
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def analyze_audio(file_path):
    """Membaca data audio dan sample rate dari file."""
    # Menggunakan pydub untuk fleksibilitas membaca berbagai format
    # kemudian mengekstrak data mentahnya
    audio = AudioSegment.from_file(file_path)
    
    # Mengambil sample rate (misal: 44100 Hz)
    sample_rate = audio.frame_rate
    
    # Mengambil data audio sebagai array numpy
    # Normalisasi ke rentang -1 hingga 1 untuk analisis
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(audio.sample_width * 8).max
    
    # Jika stereo, ambil channel kiri saja untuk simplifikasi analisis
    if audio.channels > 1:
        samples = samples[::audio.channels]
        
    return sample_rate, samples

def plot_waveform(ax, samples, sample_rate, title):
    """Membuat plot visualisasi gelombang (waveform)."""
    duration = len(samples) / sample_rate
    time = np.linspace(0., duration, len(samples))
    ax.plot(time, samples)
    ax.set_title(title)
    ax.set_xlabel("Waktu [detik]")
    ax.set_ylabel("Amplitudo")
    ax.grid(True)

def plot_spectrum(ax, samples, sample_rate, title):
    """Membuat plot spektrum frekuensi menggunakan FFT."""
    N = len(samples)
    yf = fft(samples)
    xf = fftfreq(N, 1 / sample_rate)
    
    # Kita hanya perlu separuh dari spektrum (karena simetris)
    N_half = N // 2
    yf_abs = np.abs(yf[0:N_half])
    
    ax.plot(xf[0:N_half], yf_abs)
    ax.set_title(title)
    ax.set_xlabel("Frekuensi [Hz]")
    ax.set_ylabel("Magnitudo")
    ax.set_yscale('log') # Skala logaritmik lebih mudah dibaca
    ax.grid(True)


# --- 1. KOMPRESI AUDIO ---
wav_file = "D:/Berkas_Rizki/Semester_6/Sistem Multimedia/UAS/soal2/THX Sound Effect - PMcComb.wav"
mp3_file = "output_audio.mp3"
bitrate = "32k"  # "16K", "32K", "64k", "192k", "320k"

print("Mengompresi file audio...")
# Muat file WAV
audio = AudioSegment.from_wav(wav_file)
# Ekspor ke MP3 dengan bitrate yang ditentukan
audio.export(mp3_file, format="mp3", bitrate=bitrate)
print(f"File '{wav_file}' berhasil dikompresi ke '{mp3_file}' dengan bitrate {bitrate}.")


# --- 2. PERBANDINGAN UKURAN FILE ---
wav_size = os.path.getsize(wav_file)
mp3_size = os.path.getsize(mp3_file)
rasio_kompresi = wav_size / mp3_size

print("\n--- Analisis Ukuran File ---")
print(f"Ukuran WAV: {wav_size / 1024:.2f} KB")
print(f"Ukuran MP3: {mp3_size / 1024:.2f} KB")
print(f"Rasio Kompresi: {rasio_kompresi:.2f} : 1 (WAV {rasio_kompresi:.2f} kali lebih besar dari MP3)")


# --- 3. ANALISIS & VISUALISASI ---
print("\nMenganalisis dan membuat visualisasi...")

# Baca data dari kedua file
sr_wav, samples_wav = analyze_audio(wav_file)
sr_mp3, samples_mp3 = analyze_audio(mp3_file)


# Siapkan plot
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Perbandingan Audio WAV (Asli) vs. MP3 (Kompresi)', fontsize=16)

# Plot Waveform
plot_waveform(axes[0, 0], samples_wav, sr_wav, "Waveform - WAV (Asli)")
plot_waveform(axes[0, 1], samples_mp3, sr_mp3, f"Waveform - MP3 ({bitrate})")

# Plot Spektrum Frekuensi
plot_spectrum(axes[1, 0], samples_wav, sr_wav, "Spektrum Frekuensi - WAV (Asli)")
plot_spectrum(axes[1, 1], samples_mp3, sr_mp3, f"Spektrum Frekuensi - MP3 ({bitrate})")
axes[1, 1].set_xlim(axes[1, 0].get_xlim()) # Samakan batas X agar mudah dibandingkan
axes[1, 1].set_ylim(axes[1, 0].get_ylim()) # Samakan batas Y agar mudah dibandingkan


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
print("\nAnalisis selesai.")