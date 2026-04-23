import subprocess, sys, os
import torch, cv2
import numpy as np
from PIL import Image
from pathlib import Path
from backend.config import VRAM_GB, log

PYTORCH_ENGINE_PATHS = {'waifu2x': {'installed': False, 'nunif_dir': None}}
NUNIF_DIR = Path('/content/nunif')


def install_waifu2x():
    """Instala waifu2x via nunif (clone + download de modelos)."""
    global PYTORCH_ENGINE_PATHS

    # 1. Clone nunif se não existir
    if not NUNIF_DIR.exists():
        print("⬇️  Clonando nunif (waifu2x)...")
        try:
            subprocess.check_call([
                'git', 'clone', '--depth', '1',
                'https://github.com/nagadomi/nunif.git',
                str(NUNIF_DIR)
            ], timeout=120)
        except Exception as e:
            print(f"⚠️  Clone falhou: {e} — Waifu2x usará Lanczos fallback")
            return

    # 2. Adicionar ao sys.path
    nunif_str = str(NUNIF_DIR)
    if nunif_str not in sys.path:
        sys.path.insert(0, nunif_str)

    # 3. Baixar modelos pré-treinados
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'waifu2x.download_models'],
            cwd=nunif_str, timeout=300
        )
        print("  ✅ Modelos waifu2x baixados")
    except Exception as e:
        print(f"⚠️  Download de modelos waifu2x falhou: {e}")
        print("   torch.hub.load tentará baixar automaticamente em runtime")

    PYTORCH_ENGINE_PATHS['waifu2x']['installed'] = True
    PYTORCH_ENGINE_PATHS['waifu2x']['nunif_dir'] = nunif_str
    print("  ✅ Waifu2x (nunif) configurado via torch.hub")


class Waifu2xEnginePyTorch:
    """Engine Waifu2x real via torch.hub.load (nunif).
    
    Usa a API de alto nível Waifu2xImageModel com infer() que aceita PIL Images.
    Fallback para Lanczos 4-tap se o carregamento falhar.
    """

    # Mapeamento frontend → waifu2x method
    VALID_METHODS = {'scale', 'noise', 'noise_scale', 'scale4x', 'noise_scale4x'}

    def __init__(self, model_type='art', noise_level=2, method='noise_scale', tile=0, gpu_id=0):
        self.model_type = model_type
        self.noise_level = noise_level
        self.method = method if method in self.VALID_METHODS else 'noise_scale'
        self.native_scale = 1 if method == 'noise' else (4 if '4x' in method else 2)
        self._model = None
        self._fallback = False
        self._logged_fallback = False

        # Tile size automático baseado em VRAM
        if tile == 0:
            if VRAM_GB < 6: tile = 192
            elif VRAM_GB < 12: tile = 384
            else: tile = 512
        self.tile = tile

        # Garantir nunif no sys.path
        nunif_dir = str(PYTORCH_ENGINE_PATHS['waifu2x'].get('nunif_dir', NUNIF_DIR))
        if nunif_dir not in sys.path:
            sys.path.insert(0, nunif_dir)

        # Carregar modelo via torch.hub (API de alto nível com infer())
        try:
            self._model = torch.hub.load(
                nunif_dir,
                "waifu2x",
                model_type=model_type,
                method=self.method,
                noise_level=noise_level,
                source="local",
                trust_repo=True
            )
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                # FP16 para economizar VRAM na T4
                if VRAM_GB < 16:
                    self._model = self._model.half()
            log.info(f"Waifu2x loaded: type={model_type} method={self.method} noise={noise_level}")
        except Exception as e:
            log.warning(f"Waifu2x torch.hub.load failed: {e} — usando Lanczos fallback")
            self._fallback = True

    def enhance(self, frame_bgr):
        if self._fallback or self._model is None:
            return self._enhance_lanczos(frame_bgr)

        try:
            # Converter BGR (OpenCV) → RGB (PIL)
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            with torch.no_grad():
                result = self._model.infer(pil_img)

            # Converter PIL result → BGR numpy
            result_np = np.array(result)
            return cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                log.warning("Waifu2x OOM, tentando com half precision")
                try:
                    self._model = self._model.half()
                    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    with torch.no_grad():
                        result = self._model.infer(pil_img)
                    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                except Exception:
                    log.warning("Waifu2x OOM retry falhou, fallback lanczos")
                    return self._enhance_lanczos(frame_bgr)
            log.warning(f"Waifu2x infer failed: {e}, fallback lanczos")
            return self._enhance_lanczos(frame_bgr)

        except Exception as e:
            if not self._logged_fallback:
                log.warning(f"Waifu2x enhance failed: {e}, usando Lanczos")
                self._logged_fallback = True
            return self._enhance_lanczos(frame_bgr)

    def _enhance_lanczos(self, frame_bgr):
        if not self._logged_fallback:
            log.warning("Waifu2x: usando Lanczos fallback")
            self._logged_fallback = True
        h, w = frame_bgr.shape[:2]
        s = self.native_scale
        return cv2.resize(frame_bgr, (w * s, h * s), interpolation=cv2.INTER_LANCZOS4)

    def enhance_batch(self, frames_np, target_h, target_w):
        results = []
        for i in range(frames_np.shape[0]):
            out = self.enhance(frames_np[i])
            if out.shape[0] != target_h or out.shape[1] != target_w:
                out = cv2.resize(out, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            results.append(out)
        return np.stack(results)

    def release(self):
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
