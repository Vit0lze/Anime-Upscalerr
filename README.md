# 🎌 Anime Upscalerr — Colab Backend

> Backend GPU para o [Anime Upscalerr](https://anime-upscalerr.vercel.app/). Rode no Google Colab gratuitamente.
> GPU Backend for [Anime Upscalerr](https://anime-upscalerr.vercel.app/). Run on Google Colab for free.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gwMMYblkUFLzUbzm-JfPKXgY9698utJQ?usp=sharing)

![Demo](https://i.imgur.com/CEP4Mgw.gif)

---

## 🇧🇷 Português

### 📝 Descrição
O **Anime Upscalerr Colab Backend** fornece o poder de processamento necessário para fazer o upscaling de vídeos de anime usando GPUs gratuitas do Google Colab. Ele serve como o motor de processamento para a interface web.

### ⚡ Engines & Performance (T4 GPU)
| Engine | FPS | Qualidade | Melhor para |
|---|---|---|---|
| **Anime4K** | ~20 fps | ⭐⭐⭐⭐ | Anime (recomendado) |
| **Real-ESRGAN** | ~1 fps | ⭐⭐⭐⭐⭐ | Qualidade máxima |
| **Waifu2x** | ~1 fps | ⭐⭐⭐⭐⭐ | Suavidade + detalhes |
| **Bicubic/Lanczos** | ~50 fps | ⭐⭐⭐ | Previews rápidos |

### ✨ Funcionalidades
- **Color Grading GPU**: Exposure, Contrast, Saturation, HSL.
- **Denoise CUDA**: Redução de ruído bilateral via hardware.
- **Encoder HEVC/H.264**: Utiliza NVENC para processamento ultra rápido.
- **Live Preview**: Acompanhe o processamento em tempo real.

### 🚀 Como usar
1. Abra o [Anime Upscalerr Web](https://anime-upscalerr.vercel.app/).
2. Execute o notebook no Colab e copie a URL pública gerada.
3. Cole a URL no site no campo "NODE TUNNEL ADDRESS" → ESTABLISH LINK.
4. Faça o upload do vídeo e comece o processamento.

---

## 🇺🇸 English

### 📝 Description
The **Anime Upscalerr Colab Backend** provides the computational power required to upscale anime videos using free GPUs from Google Colab. It acts as the processing engine for the web interface.

### ⚡ Engines & Performance (T4 GPU)
| Engine | FPS | Quality | Best for |
|---|---|---|---|
| **Anime4K** | ~20 fps | ⭐⭐⭐⭐ | Anime (recommended) |
| **Real-ESRGAN** | ~1 fps | ⭐⭐⭐⭐⭐ | Maximum quality |
| **Waifu2x** | ~1 fps | ⭐⭐⭐⭐⭐ | Smoothness + details |
| **Bicubic/Lanczos** | ~50 fps | ⭐⭐⭐ | Fast previews |

### ✨ Features
- **GPU Color Grading**: Exposure, Contrast, Saturation, HSL.
- **CUDA Denoise**: Bilateral hardware-accelerated noise reduction.
- **HEVC/H.264 Encoder**: Uses NVENC for ultra-fast processing.
- **Live Preview**: Monitor processing in real-time.

### 🚀 How to use
1. Open [Anime Upscalerr Web](https://anime-upscalerr.vercel.app/).
2. Run the Colab notebook and copy the generated public URL.
3. Paste the URL into the "NODE TUNNEL ADDRESS" → ESTABLISH LINK.
4. Upload your video and start processing.

---

## ❓ FAQ
**PT:** É grátis? Sim, usa a GPU T4 do Colab. Meus vídeos ficam salvos? Não, baixe antes de fechar.
**EN:** Is it free? Yes, uses Colab's T4 GPU. Are my videos saved? No, download before closing.

---
<div align="center">

**[Usar agora / Use now →](https://anime-upscalerr.vercel.app/)**

[Anime4K](https://github.com/bloc97/Anime4K) · [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) · [Waifu2x](https://github.com/nagadomi/nunif)

**Author:** Vitor · **License:** MIT
</div>
