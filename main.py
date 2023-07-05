from fastapi import FastAPI, File, UploadFile
import uvicorn
import pickle
import pandas as pd
import numpy as np
import librosa as lr
import noisereduce as nr

app = FastAPI()

# Load your machine learning model
model = pickle.load(open('modelsvm.pkl', 'rb'))
file_path=""
def read_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            return file.read()
    except IOError:
        print("Error: Unable to open the file.")
        return None


audio_file=read_file(file_path)
        

# Function to remove noise from audio
def remove_noise(audio_file):
    cry_data = lr.load(audio_file, sr = 22050,mono = True)
    cry_data = cry_data[0]
    reduced_noise = nr.reduce_noise(y = cry_data, sr=22050, thresh_n_mult_nonstationary=2,stationary=False)
    return reduced_noise


def calculate_amplitude_envelope(signal, FRAME_SIZE, HOP_LENGTH):
    """Calculate the amplitude envelope of a signal with a given frame size and hop length."""
    amplitude_envelope = []

    for i in range(0, len(signal), HOP_LENGTH):
        amplitude_envelope_current_frame = max(signal[i:i + FRAME_SIZE])
        amplitude_envelope.append(amplitude_envelope_current_frame)
        z = np.array(amplitude_envelope)
    return z

def extract_amplitude_envelope(audio_file, FRAME_SIZE, HOP_LENGTH):
    ae = calculate_amplitude_envelope(audio_file, FRAME_SIZE, HOP_LENGTH)
    ae_array=np.array(ae)
    ae_me=ae_array.mean()
    return ae_me

def extract_rms(audio_file, FRAME_SIZE, HOP_LENGTH):
    rms = lr.feature.rms(y=audio_file, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    rms_array=np.array(rms)
    rms_me=rms_array.mean()
    return rms_me

def extract_zcr(audio_file, FRAME_SIZE, HOP_LENGTH):
    zcr = lr.feature.zero_crossing_rate(y=audio_file, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    zcr_array=np.array(zcr)
    zcr_me=zcr_array.mean()
    return zcr_me

def extract_stft(audio_file, FRAME_SIZE, HOP_LENGTH):
    stft=lr.stft(audio_file, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
    stft_mag=np.abs(stft) ** 2
    stft_array=np.array(stft_mag)
    stft_me=stft_array.mean()
    return stft_me

def extract_sc(audio_file, FRAME_SIZE, HOP_LENGTH):
    sc = lr.feature.spectral_centroid(y=audio_file, sr=22050, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    sc_array = np.array(sc)
    sc_me=sc_array.mean()
    return sc_me

def extract_sban(audio_file, FRAME_SIZE, HOP_LENGTH):
    sban = lr.feature.spectral_bandwidth(y=audio_file, sr=22050, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    sban_array=np.array(sban)
    sban_me=sban_array.mean()
    return sban_me


def extract_scon(audio_file, FRAME_SIZE, HOP_LENGTH):
    S = np.abs(lr.stft(audio_file))
    scon = lr.feature.spectral_contrast(S=S, sr=22050)
    scon_array=np.array(scon)
    scon_me=scon_array.mean()
    return scon_me

def extract_mfccs13(audio_file):
    mfccs_array = lr.feature.mfcc(y=audio_file, n_mfcc=13, sr=22050)
    mfccs_me=mfccs_array.mean()
    return mfccs_me

def extract_delmfccs13(audio_file):
    mfccs_array = lr.feature.mfcc(y=audio_file, n_mfcc=13, sr=22050)
    delmfccs_array = lr.feature.delta(mfccs_array)
    delmfccs_me=delmfccs_array.mean()
    return delmfccs_me

def extract_del2mfccs13(audio_file):
    mfccs_array = lr.feature.mfcc(y=audio_file, n_mfcc=13, sr=22050)
    del2mfccs_array = lr.feature.delta(mfccs_array,order=2)
    del2mfccs_me=del2mfccs_array.mean()
    return del2mfccs_me


def extract_melspec(audio_file):
    mel_spectrogram = lr.feature.melspectrogram(y=audio_file, sr=22050, n_fft=2048, hop_length=512, n_mels=256)
    log_mel_spectrogram = lr.power_to_db(mel_spectrogram)
    spec_df=np.array(log_mel_spectrogram)
    spec_mean=np.mean(spec_df)
    return spec_mean

def extract_mfccs20(audio_file):
    mfccs20_array = lr.feature.mfcc(y=audio_file, n_mfcc=20, sr=22050)
    mfccs20_me=mfccs20_array.mean()
    return mfccs20_me

def extract_mfccs1_13(audio_file, FRAME_SIZE, HOP_LENGTH):
    file_clean = remove_noise(audio_file)
    mfccs13_array = lr.feature.mfcc(y=file_clean, n_mfcc=13, sr=22050)
    zz = mfccs13_array
    a0 = zz[0].mean()
    a1 = zz[1].mean()
    a2 = zz[2].mean()
    a3 = zz[3].mean()
    a4 = zz[4].mean()
    a5 = zz[5].mean()
    a6 = zz[6].mean()
    a7 = zz[7].mean()
    a8 = zz[8].mean()
    a9 = zz[9].mean()
    a10 = zz[10].mean()
    a11 = zz[11].mean()
    a12 = zz[12].mean()
    mfccs = [a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12]
    return mfccs


def feature_extractor(file_clean, FRAME_SIZE, HOP_LENGTH):
    a = extract_amplitude_envelope(file_clean, FRAME_SIZE, HOP_LENGTH)
    b = extract_rms(file_clean, FRAME_SIZE, HOP_LENGTH)
    c = extract_zcr(file_clean, FRAME_SIZE, HOP_LENGTH)
    d = extract_stft(file_clean, FRAME_SIZE, HOP_LENGTH)
    e = extract_sc(file_clean, FRAME_SIZE, HOP_LENGTH)
    f = extract_sban(file_clean, FRAME_SIZE, HOP_LENGTH)
    g = extract_scon(file_clean, FRAME_SIZE, HOP_LENGTH)
    h = extract_mfccs13(file_clean)
    i = extract_delmfccs13(file_clean)
    j = extract_del2mfccs13(file_clean)
    k = extract_melspec(file_clean)
    l = extract_mfccs20(file_clean)
    df_pred = pd.DataFrame([[a, b, c, d, e, f, g, h, i, j, k, l]],
                           columns=['Amplitude_Envelope_Mean', 'RMS_Mean', 'ZCR_Mean', 'STFT_Mean', 'SC_Mean',
                                    'SBAN_Mean', 'SCON_Mean', 'MFCCs13Mean', 'delMFCCs13', 'del2MFCCs13', 'MelSpec',
                                    'MFCCs20'])
    return df_pred

FRAME_SIZE=1024
HOP_LENGTH=512

@app.post("/predict")
async def predict_audio(audio_file_post: UploadFile = File(...)):
    # Save the uploaded audio file
    with open("uploaded_audio.wav", "wb") as audio:
        audio.write(await audio_file_post.read())
    
    # Remove noise from the audio file
    clean_audio = remove_noise("uploaded_audio.wav")
    
    # Extract features from the clean audio
    features = feature_extractor(clean_audio, FRAME_SIZE, HOP_LENGTH)
    
    # Make prediction using the loaded model
    prediction = model.predict(features).round()
    
    # Define the class labels
    classes = {
        0: "The baby has belly pain",
        1: "The baby is Burping",
        2: "The baby is experiencing Discomfort",
        3: "The baby is Hungry",
        4: "The baby is tired"
    }
    
    # Get the predicted class label
    predicted_label = classes[prediction[0]]
    
    # Return the prediction as JSON response
    return {"prediction": predicted_label}

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=10000)
