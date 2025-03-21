import os
import librosa
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks
from pathlib import Path
from scipy.io import wavfile
import matplotlib  as plt
import pandas as pd

# 归一化信号函数
def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)

# 计算短时能量和自动 threshold
def calculate_energy_and_threshold(y, frame_length=2048, hop_length=512, threshold_factor=1.5):
    """
    计算短时能量并根据平均能量自动计算阈值
    """
    energy = np.array([
        sum(abs(y[i:i + frame_length] ** 2))
        for i in range(0, len(y), hop_length)
    ])
    threshold = np.mean(energy) * threshold_factor
    return energy, threshold

# 提取敲击片段
def extract_strokes(signal, sr, energy, threshold, before=2400, after=12000, max_length=10000, show=False):
    """
    根据能量和阈值提取敲击片段，并填充静音到固定长度
    """
    peaks, _ = find_peaks(energy, height=threshold, distance=int(len(energy)/25))  # 动态调整distance
    strokes = []
    
    for peak in peaks:
        start_sample = max(int(peak * sr / len(energy)) - before, 0)
        end_sample = min(int(peak * sr / len(energy)) + after, len(signal))
        stroke = signal[start_sample:end_sample]
        
        # 如果片段长度小于目标长度，进行填充
        if len(stroke) < max_length:
            padding = np.zeros(max_length - len(stroke))
            stroke = np.concatenate((stroke, padding))
        elif len(stroke) > max_length:
            stroke = stroke[:max_length]
        
        strokes.append(stroke)
        
        if show:
            plt.figure(figsize=(7, 2))
            plt.plot(stroke)
            plt.title(f"Keystroke at {peak/sr:.2f}s")
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return strokes

# 提取标签
def extract_label(filename):
    base_name = os.path.basename(filename).rstrip('.wav')
    if '_' in base_name:
        return base_name.split('_', 1)[0]
    else:
        return base_name

# 处理单个音频文件
def process_audio_file(file_path, sample_rate=44100, threshold_factor=1.5, max_length=10000, show=False):
    """
    处理单个音频文件：计算能量、自动计算阈值、提取敲击片段并填充到固定长度
    """
    signal, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    signal = normalize_signal(signal)  # 添加归一化处理
    energy, threshold = calculate_energy_and_threshold(signal, threshold_factor=threshold_factor)
    
    if show:
        plt.figure(figsize=(10, 4))
        plt.plot(energy)
        plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
        plt.title("Short-Time Energy")
        plt.legend()
        plt.show()
    
    strokes = extract_strokes(signal, sr, energy, threshold, max_length=max_length, show=show)
    return strokes

# 遍历音频文件夹
def get_audio_files(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    return audio_files

# 保存处理结果
def save_processed_data(audio_dir, output_npz="processed_data.npz", output_csv="processed_data.csv", max_length=10000):
    """
    遍历音频文件夹，自动计算每个文件的 threshold，并提取敲击片段
    """
    all_files = get_audio_files(audio_dir)
    all_data = []
    all_labels = []
    successful_files = []  # 用于记录成功处理的文件路径
    
    for file_path in tqdm(all_files, desc="Processing Audio Files"):
        try:
            label = extract_label(os.path.basename(file_path))
            strokes = process_audio_file(file_path, max_length=max_length, show=False)
            
            if len(strokes) == 0:
                raise ValueError(f"No valid strokes extracted for {file_path}")
            
            all_data.extend(strokes)
            all_labels.extend([label] * len(strokes))
            successful_files.extend([file_path] * len(strokes))
        
        except Exception as e:
            print(f"[SKIP] {file_path}: {str(e)}")
            continue
    
    assert len(all_data) == len(all_labels) == len(successful_files), "Data/labels mismatch"
    
    # 保存为NPZ
    np.savez(output_npz, data=all_data, labels=all_labels)
    print(f"\nSaved {len(all_data)} samples to {output_npz}")
    
    # 保存为CSV
    df = pd.DataFrame({
        "file_path": [os.path.basename(f) for f in successful_files],
        "label": all_labels,
        "length": [len(s) for s in all_data]
    })
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV summary to {output_csv}")

# 生成 Mel 频谱图
def generate_mel_spectrogram(audio_segment, sr=44100, n_mels=64, hop_length=225):
    mel_spec = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# 在 preprocess_audio_files_for_spectrograms 函数中
def preprocess_audio_files_for_spectrograms(audio_dir, target_shape=(64, 156)):
    all_data = []
    all_labels = []
    all_files = get_audio_files(audio_dir)
    
    for file_path in tqdm(all_files, desc="Generating Spectrograms"):
        try:
            label = extract_label(os.path.basename(file_path))
            strokes = process_audio_file(file_path, max_length=10000, show=False)
            
            for stroke in strokes:
                # 生成频谱图并调整形状
                spectrogram = generate_mel_spectrogram(stroke, sr=44100)
                spectrogram_resized = librosa.util.fix_length(
                    spectrogram, size=target_shape[1], axis=1
                )
                
                # 关键修改：强制转换为 [1, 64, 156] 形状
                spectrogram_resized = np.expand_dims(spectrogram_resized, axis=0)  # 添加通道维度
                all_data.append(spectrogram_resized)
                all_labels.append(label)
                
        except Exception as e:
            print(f"[SKIP] {file_path}: {str(e)}")
            continue
    
    return np.array(all_data), np.array(all_labels)

# 设置参数
audio_dir = "Keystroke-Datasets/MBPWavs"
target_shape = (64, 156)  # 目标频谱图的形状

if __name__ == "__main__":
    # 第一步：处理音频文件并保存敲击片段
    save_processed_data(audio_dir)
    
    # 第二步：生成频谱图数据集并保存为 .npz 文件
    spectrograms, labels = preprocess_audio_files_for_spectrograms(audio_dir, target_shape)
    np.savez('spectrogram_data.npz', data=spectrograms, labels=labels)
    print("\nProcessing complete! Output files have been saved.")