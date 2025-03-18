import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.responses import FileResponse
from pydantic import BaseModel
import logging

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

app = FastAPI(title="Zonos TTS API", version="1.1")

print("Model yükleniyor...")
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
print("Model yüklendi.")

logging.basicConfig(level=logging.INFO)

class TTSRequest(BaseModel):
    text: str
    language: str = "en-us"
    speaking_rate: float = 15.0  # Konuşma hızı (0-40, 30 çok hızlı, 10 yavaş)
    pitch_std: float = 20.0  # Perde standart sapması (20-45 normal, 60-150 ifade içeren konuşmalar)
    emotions: list[float] = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]  # Duygusal durum vektörü
    #                       Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral
@app.get("/")
def home():
    return {"message": "Zonos TTS API is running!"}

@app.post("/tts/")
async def generate_tts(request: TTSRequest):

    torch.manual_seed(421)

    logging.info(f"Gelen istek: {request.text} | Dil: {request.language} | Hız: {request.speaking_rate} | Perde: {request.pitch_std} | Duygu: {request.emotions}")

    speaker = None  

    try:
        cond_dict = make_cond_dict(
            text=request.text, 
            speaker=speaker, 
            language=request.language,
            speaking_rate=request.speaking_rate,
            pitch_std=request.pitch_std,
            emotion=request.emotions
        )
        conditioning = model.prepare_conditioning(cond_dict)
        codes = model.generate(conditioning)
        wavs = model.autoencoder.decode(codes).cpu()
        sample_rate = model.autoencoder.sampling_rate
        audio_data = wavs.squeeze(0)  

        logging.info(f"Ses verisi şekli: {audio_data.shape}")
        logging.info(f"Min-Max Değerleri: {np.min(audio_data.numpy())}, {np.max(audio_data.numpy())}")
        logging.info(f"NaN Var mı?: {np.isnan(audio_data.numpy()).any()} | Inf Var mı?: {np.isinf(audio_data.numpy()).any()}")

        if np.isnan(audio_data.numpy()).any() or np.isinf(audio_data.numpy()).any():
            raise ValueError("Ses verisi geçersiz (NaN veya Inf içeriyor).")

        output_path = "output.wav"
        torchaudio.save(output_path, audio_data, sample_rate, format="wav")
        logging.info(f"Ses başarıyla oluşturuldu: {output_path}")

        return FileResponse(output_path, media_type="audio/wav", filename="output.wav")

    except Exception as e:
        logging.error(f"HATA: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.post("/tts_with_speaker/")
async def generate_tts_with_speaker(
    text: str, 
    speaker_audio: UploadFile = File(...), 
    language: str = "en-us",
    speaking_rate: float = 15.0,
    pitch_std: float = 20.0,
    emotions: list[float] = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]
):
    
    logging.info(f"Gelen istek: {text} | Dil: {language} | Hız: {speaking_rate} | Perde: {pitch_std} | Duygu: {emotions}")

    try:
        wav, sr = torchaudio.load(speaker_audio.file)
        speaker = model.make_speaker_embedding(wav, sr)
        torch.manual_seed(421)

        cond_dict = make_cond_dict(
            text=text, 
            speaker=speaker, 
            language=language,
            speaking_rate=speaking_rate,
            pitch_std=pitch_std,
            emotion=emotions
        )
        conditioning = model.prepare_conditioning(cond_dict)
        codes = model.generate(conditioning)
        wavs = model.autoencoder.decode(codes).cpu()
        sample_rate = model.autoencoder.sampling_rate
        audio_data = wavs.squeeze(0)  

        logging.info(f"Ses verisi şekli: {audio_data.shape}")
        logging.info(f"Min-Max Değerleri: {np.min(audio_data.numpy())}, {np.max(audio_data.numpy())}")
        logging.info(f"NaN Var mı?: {np.isnan(audio_data.numpy()).any()} | Inf Var mı?: {np.isinf(audio_data.numpy()).any()}")

        if np.isnan(audio_data.numpy()).any() or np.isinf(audio_data.numpy()).any():
            raise ValueError("Ses verisi geçersiz (NaN veya Inf içeriyor).")

        output_path = "output.wav"
        torchaudio.save(output_path, audio_data, sample_rate, format="wav")
        logging.info(f"Ses başarıyla oluşturuldu: {output_path}")

        return FileResponse(output_path, media_type="audio/wav", filename="output.wav")

    except Exception as e:
        logging.error(f"HATA: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
