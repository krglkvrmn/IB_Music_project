import argparse
import shutil

import pandas as pd
import os
from multiprocessing import Process
import librosa as lb
import numpy as np
import warnings
import math


warnings.filterwarnings("ignore")


def amplitude_envelope(signal: np.ndarray, frame_size: int, hop_length: int) -> np.ndarray:
    """Calculate the amplitude envelope of a signal with a given frame size nad hop length.
       Taken from here: https://github.com/musikalkemist/AudioSignalProcessingForML/"""
    amplitude_envelopes = []
    # calculate amplitude envelope for each frame
    for i in range(0, len(signal), hop_length): 
        amplitude_envelope_current_frame = max(signal[i:i+frame_size]) 
        amplitude_envelopes.append(amplitude_envelope_current_frame)
    return np.array(amplitude_envelopes)


parser = argparse.ArgumentParser(description="Extract audio features from fma dataset using multiple processes")
parser.add_argument("dataset_metadata", help="CSV file containing 3 fields: 'track_id', 'genre_id' and 'filepath'\
                                                  which correspond to genre name and path to music file of each track")
parser.add_argument("--fma_path", type=str, required=True,
                    help="Path to cloned fma repository")
parser.add_argument("-f", "--frame_size", type=int, default=2048,
                    help="Frame size that is used to extract all features")
parser.add_argument("-l", "--hop_length", type=int, default=512,
                    help="Hop length that is used to extract all features")
parser.add_argument("--mel_spec_frame_size", type=int, default=4096,
                    help="Frame size which is used for STFT")
parser.add_argument("--mel_spec_hop_length", type=int, default=2048,
                    help="Hop length which is used for STFT")
parser.add_argument("--nmels", type=int, default=20,
                    help="Number of mel bands")
parser.add_argument("--nmfcc", type=int, default=13,
                    help="Number of mel-frequency cepstrum coefficients (MFCC). Output includes also its 1st and 2nd\
                         derivatives")
parser.add_argument("-d", "--duration", type=float, default=29.5,
                    help="Use d first seconds of each track for feature extraction")
parser.add_argument("-t", "--threads", type=int, default=1,
                    help="Number of threads.")
parser.add_argument("-o", "--output", type=str, help="Output directory")
args = parser.parse_args()

DATA_FILE = args.dataset_metadata
FMA_PATH = args.fma_path
FRAME_SIZE = args.frame_size
HOP_LENGTH = args.hop_length
STFT_FRAME_SIZE= args.mel_spec_frame_size
STFT_HOP_LENGTH = args.mel_spec_hop_length
NMELS = args.nmels
NMFCC = args.nmfcc
DURATION = args.duration
N_THREADS = args.threads
OUT_DIR = args.output
FEATURE_LIST = ["mel_spec", "spectral_bandwidth", "spectral_centroid", "spectral_flatness", "mfcc",
                "amplitude_envelope", "zero_crossing_rate", "root_mean_square"]


def set_up(output_directory: str):
    """Create output directory if not exists"""
    if output_directory in os.listdir() and os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    if output_directory in os.listdir() and not os.path.isdir(output_directory):
        raise OSError(f"{output_directory} already exists and it is not a directory")
    else:
        os.mkdir(output_directory)


def combine_data_and_clean_up(output_directory: str):
    """Concatenate arrays obtained from multiprocess feature extraction run and delete them"""
    out_files = sorted(os.listdir(output_directory), key=lambda x: int(x.split(".")[-2].split("_")[-1]))
    for feature in FEATURE_LIST:
        feature_files = list(filter(lambda x: feature in x, out_files))
        concat_features = np.concatenate([np.load(os.path.join(output_directory, filename))
                                          for filename in feature_files])
        new_filename = "_".join(feature_files[0].split("_")[:-1])
        np.save(os.path.join(output_directory, f"{new_filename}.npy"), concat_features)
        # Clean up
        for filename in feature_files:
            os.remove(os.path.join(output_directory, filename))


def worker(subset: pd.DataFrame, worker_id: int):
    """Worker function that extracts audio features of given subset of data."""
    spectrograms, bandwidths, centroids, flatnesses, mfccs, zcrs, aenvs, rmss = [], [], [], [], [], [], [], []
    for idx, filepath in enumerate(subset.filepath):
        if worker_id == 0:
            if idx % 50 == 0:
                print(f"Processing file {idx + 1}/{len(subset)}")
        signal, sr = lb.load(filepath, duration=DURATION)
        mel_spec = lb.feature.melspectrogram(signal, sr=sr, n_fft=STFT_FRAME_SIZE,
                                             hop_length=STFT_HOP_LENGTH, n_mels=NMELS)
        log_mel_spec = lb.power_to_db(mel_spec)
        spectrograms.append(log_mel_spec)
        
        band = lb.feature.spectral_bandwidth(signal, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        bandwidths.append(band)
        
        centroid = lb.feature.spectral_centroid(signal, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        centroids.append(centroid)
        
        flatness = lb.feature.spectral_flatness(signal, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        flatnesses.append(flatness)
        
        zcr = lb.feature.zero_crossing_rate(signal, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        zcrs.append(zcr)

        aenv = amplitude_envelope(signal, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH)
        aenvs.append(aenv)
        
        rms = lb.feature.rms(signal, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        rmss.append(rms)
    
        mfcc = lb.feature.mfcc(signal, n_mfcc=NMFCC)
        mfcc2 = lb.feature.delta(mfcc)
        mfcc3 = lb.feature.delta(mfcc, order=2)
        mfcc = np.concatenate((mfcc, mfcc2, mfcc3))
        mfccs.append(mfcc.T)
    # Every worker saves features into separate files.
    np.save(os.path.join(OUT_DIR, f"mel_spec_{STFT_FRAME_SIZE}_{STFT_HOP_LENGTH}_{NMELS}_{worker_id + 1}.npy"),
            np.array(spectrograms))
    np.save(os.path.join(OUT_DIR, f"spectral_bandwidth_{FRAME_SIZE}_{HOP_LENGTH}_{worker_id + 1}.npy"),
            np.array(bandwidths))
    np.save(os.path.join(OUT_DIR, f"spectral_centroid_{FRAME_SIZE}_{HOP_LENGTH}_{worker_id + 1}.npy"),
            np.array(centroids))
    np.save(os.path.join(OUT_DIR, f"spectral_flatness_{FRAME_SIZE}_{HOP_LENGTH}_{worker_id + 1}.npy"),
            np.array(flatnesses))
    np.save(os.path.join(OUT_DIR, f"mfcc_{NMFCC}_{worker_id + 1}.npy"), np.array(mfccs))
    np.save(os.path.join(OUT_DIR, f"zero_crossing_rate_{FRAME_SIZE}_{HOP_LENGTH}_{worker_id + 1}.npy"),
            np.array(zcrs))
    np.save(os.path.join(OUT_DIR, f"amplitude_envelope_{FRAME_SIZE}_{HOP_LENGTH}_{worker_id + 1}.npy"),
            np.array(aenvs))
    np.save(os.path.join(OUT_DIR, f"root_mean_square_{FRAME_SIZE}_{HOP_LENGTH}_{worker_id + 1}.npy"),
            np.array(rmss))


set_up(OUT_DIR)

data = pd.read_csv(DATA_FILE)
data.filepath = data.filepath.apply(lambda path: os.path.join(FMA_PATH, path))
# List of broken dataset files (duration << 30 seconds)
# Set up manually according to your own filepath field in metadata table.
blacklist = {os.path.join(FMA_PATH, "fma_small/098/098565.mp3"), os.path.join(FMA_PATH, "fma_small/098/098567.mp3"),
             os.path.join(FMA_PATH, "fma_small/098/098569.mp3"), os.path.join(FMA_PATH, "fma_small/099/099134.mp3"),
             os.path.join(FMA_PATH, "fma_small/108/108925.mp3"), os.path.join(FMA_PATH, "fma_small/133/133297.mp3")}
data = data.query("filepath not in @blacklist")

data_split = math.ceil(len(data) / int(N_THREADS))  # Split data into N_THREADS of size data_split
processes = []
for i in range(N_THREADS):
    subset = data.iloc[data_split*i:data_split*(i+1), :]  # Select subset of size data_split for every process
    processes.append(Process(target=worker, args=(subset, i)))

for idx, _ in enumerate(processes):
    processes[idx].start()
for idx, _ in enumerate(processes):
    processes[idx].join()

combine_data_and_clean_up(OUT_DIR)
