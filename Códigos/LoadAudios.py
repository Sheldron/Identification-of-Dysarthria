import os
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import tsfel
from python_speech_features import mfcc

def ConfigAudioFiles():
    torgoPath = r"C:\Users\User\00 Sheldron\UERJ\00 Projetos\Camila\torgo"

    def load_audio_files(torgoPath):
        """
        Load audio files from TORGO dataset directories

        Args:
            base_path (str): Root directory of the TORGO dataset

        Returns:
            list: Paths to audio files
            list: Corresponding labels
        """
        audio_files = []
        labels = []

        # Define dataset subdirectories
        subdirs = ['F_Con', 'F_Dys', 'M_Con', 'M_Dys']

        for subdir in tqdm(subdirs, desc="Processing Directories"):
            current_path = os.path.join(torgoPath, subdir)

            # Determine label based on directory
            label = 'Dysarthric' if 'Dys' in subdir else 'Control'

            # Walk through all audio files
            for root, dirs, files in os.walk(current_path):
                for file in tqdm(files, desc=f"Scanning {subdir}", leave=False):
                    if file.endswith('.wav'):
                        file_path = os.path.join(root, file)
                        audio_files.append(file_path)
                        labels.append(label)

        return audio_files, labels

    def CheckDys():
        cont =0
        for i in labels:
            if i == 'Dysarthric':
                cont = cont+1
        print(cont)

    print("Step 1: Loading Audio Files")
    audio_files, labels = load_audio_files(torgoPath)
    print(f"Total audio files found: {len(audio_files)}")
    CheckDys()
    print("\n========================================\n")
    
    def audio_read_feat_extractor(sr=44100, domain='temporal', audio1=audio_files,label=labels):

        X = pd.DataFrame([])
        y=[]
        indexErrorFiles = []

        if domain == 'temporal':

            cfg = tsfel.get_features_by_domain(domain)

    
            for file in audio1:
                if file != r'C:\Users\User\00 Sheldron\UERJ\00 Projetos\Camila\torgo\F_Dys\wav_headMic_F01\wav_headMic_F01_0067.wav':
                    if file != r'C:\Users\User\00 Sheldron\UERJ\00 Projetos\Camila\torgo\F_Dys\wav_headMic_F01\wav_headMic_F01_0068.wav':
                        if file != r'C:\Users\User\00 Sheldron\UERJ\00 Projetos\Camila\torgo\F_Con\wav_arrayMic_FC01S01\wav_arrayMic_FC01S01_0256.wav':
                            audio, fs = librosa.load(file, sr=sr)
                            print(file)

                            features = tsfel.time_series_features_extractor(cfg, audio, fs)

                            X = pd.concat([X, pd.DataFrame(features)], axis = 0)
                        else: indexErrorFiles.append(audio1.index(file))
                    else: indexErrorFiles.append(audio1.index(file))
                else: indexErrorFiles.append(audio1.index(file))
                        
            X.reset_index(inplace=True, drop=True)

            X.to_csv('./data/temporal_features_tsfel.csv', index=False, header=False)
            y = pd.DataFrame(label)
            y.columns = ['Classes']
            newY = y['Classes'].drop([indexErrorFiles[0], indexErrorFiles[1], indexErrorFiles[2]])
            pd.DataFrame(newY).to_csv('./data/label.csv', index=False, header=False)
    
    audio_read_feat_extractor(sr=44100, domain='temporal', audio1=audio_files, label=labels)

ConfigAudioFiles()
