# reference_audio/

Chatterbox TTS ses klonlama için referans ses dosyaları bu dizine konulur.

## Kullanım

1. 10–60 saniye uzunluğunda temiz ses kaydı alın (gürültüsüz ortam)
2. Dosyayı bu dizine koyun: `reference_audio/<isim>.wav` (veya .mp3/.flac/.m4a)
3. Dashboard'da ses seçiciden `🔬 Chatterbox (Ses Klonlama)` seçin
4. `voice` parametresi otomatik olarak `chatterbox:clone:<isim>` formatına çevrilir

## Ortam Değişkenleri

- `CHATTERBOX_DEVICE=cpu` (varsayılan) veya `cuda` (GPU kullanımı için)
- `CHATTERBOX_CFG_WEIGHT=0.2` (düşük = daha yavaş, doğal konuşma)
- `CHATTERBOX_CHUNK_THRESHOLD=600` (bu karakter sayısını geçen metinler parçalanır)

## Kurulum

```bash
pip install chatterbox-tts
pip install git+https://github.com/m-bain/whisperX.git
```
