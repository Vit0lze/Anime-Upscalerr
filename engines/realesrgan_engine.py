import subprocess, sys, os

# Shim: basicsr importa torchvision.transforms.functional_tensor
# que foi removido em torchvision >= 0.18. Criamos um alias.
import torchvision.transforms.functional as _tvF
import types as _types
if 'torchvision.transforms.functional_tensor' not in sys.modules:
    _shim = _types.ModuleType('torchvision.transforms.functional_tensor')
    for _attr in dir(_tvF):
        if not _attr.startswith('_'):
            setattr(_shim, _attr, getattr(_tvF, _attr))
    sys.modules['torchvision.transforms.functional_tensor'] = _shim

import torch, cv2
import numpy as np
from pathlib import Path
from backend.config import VRAM_GB, log

PYTORCH_ENGINE_PATHS = {'realesrgan': {'installed': False, 'models': {}}}

def install_realesrgan():
    global PYTORCH_ENGINE_PATHS
    try:
        import realesrgan
        PYTORCH_ENGINE_PATHS['realesrgan']['installed'] = True
        print(f"  Real-ESRGAN {getattr(realesrgan, '__version__', '?')} já disponível")
    except ImportError:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'realesrgan'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'basicsr==1.4.2'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'gfpgan'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'filterpy', 'facexlib', 'lmdb', 'tb-nightly', 'yapf'])
            import realesrgan
            PYTORCH_ENGINE_PATHS['realesrgan']['installed'] = True
            print(f"  Real-ESRGAN {getattr(realesrgan, '__version__', '?')} instalado")
        except Exception as e:
            print(f"⚠️  Real-ESRGAN install falhou: {e}")
            print(f"   Modelos foram baixados — engine tentará carregar em runtime")

    models_dir = Path('/content/realesrgan_models')
    models_dir.mkdir(parents=True, exist_ok=True)

    realesrgan_models = {
        'x4plus_anime': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
            'filename': 'RealESRGAN_x4plus_anime_6B.pth',
            'scale': 4, 'num_block': 6, 'num_feat': 64, 'num_grow_ch': 32,
            'arch': 'RRDBNet'
        },
        'x4plus': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'filename': 'RealESRGAN_x4plus.pth',
            'scale': 4, 'num_block': 23, 'num_feat': 64, 'num_grow_ch': 32,
            'arch': 'RRDBNet'
        },
        'x2plus': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            'filename': 'RealESRGAN_x2plus.pth',
            'scale': 2, 'num_block': 23, 'num_feat': 64, 'num_grow_ch': 32,
            'arch': 'RRDBNet'
        },
        'animevideov3': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
            'filename': 'realesr-animevideov3.pth',
            'scale': 4, 'num_block': None, 'num_feat': None, 'num_grow_ch': None,
            'arch': 'SRVGGNetCompact'
        },
    }
    realesrgan_models['_gfpgan'] = {
        'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        'filename': 'GFPGANv1.3.pth',
        'scale': None, 'arch': 'gfpgan'
    }

    for model_key, model_info in realesrgan_models.items():
        model_path = models_dir / model_info['filename']
        if not model_path.exists():
            print(f"⬇️  Baixando {model_info['filename']}...")
            try:
                subprocess.run(['wget', '-q', '-O', str(model_path), model_info['url']], check=True, timeout=120)
                print(f"   ✅ {model_info['filename']}")
            except Exception as e:
                print(f"   ❌ {model_info['filename']}: {e}")
                continue
        PYTORCH_ENGINE_PATHS['realesrgan']['models'][model_key] = str(model_path)

class RealESRGANEnginePyTorch:
    def __init__(self, model_name='x4plus_anime', tile=0, half=True, gpu_id=0, denoise_strength=0.5, face_enhance=False):
        self.model_name = model_name
        self.half = half
        self.gpu_id = gpu_id
        self.face_enhance = face_enhance
        self._upsampler = None
        self._face_enhancer = None

        model_info_map = {
            'x4plus_anime': {'scale': 4, 'num_block': 6, 'num_feat': 64, 'num_grow_ch': 32, 'arch': 'RRDBNet'},
            'x4plus': {'scale': 4, 'num_block': 23, 'num_feat': 64, 'num_grow_ch': 32, 'arch': 'RRDBNet'},
            'x2plus': {'scale': 2, 'num_block': 23, 'num_feat': 64, 'num_grow_ch': 32, 'arch': 'RRDBNet'},
            'animevideov3': {'scale': 4, 'arch': 'SRVGGNetCompact'},
        }
        info = model_info_map.get(model_name, model_info_map['x4plus_anime'])
        self.native_scale = info['scale']

        if tile == 0:
            if VRAM_GB < 6: tile = 192
            elif VRAM_GB < 12: tile = 384
            else: tile = 512
        self.tile = tile

        model_path = PYTORCH_ENGINE_PATHS['realesrgan']['models'].get(model_name)
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo Real-ESRGAN '{model_name}' não encontrado")

        if info['arch'] == 'SRVGGNetCompact':
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=info['num_feat'],
                            num_block=info['num_block'], num_grow_ch=info['num_grow_ch'], scale=info['scale'])

        from realesrgan import RealESRGANer
        self._upsampler = RealESRGANer(
            scale=info['scale'], model_path=model_path, model=model,
            tile=self.tile, tile_pad=10, pre_pad=0, half=half, gpu_id=gpu_id
        )

        if face_enhance:
            gfpgan_path = PYTORCH_ENGINE_PATHS['realesrgan']['models'].get('_gfpgan')
            if gfpgan_path and Path(gfpgan_path).exists():
                try:
                    from gfpgan import GFPGANer
                    self._face_enhancer = GFPGANer(
                        model_path=gfpgan_path, upscale=info['scale'],
                        arch='clean', channel_multiplier=2, bg_upsampler=self._upsampler
                    )
                except Exception as e:
                    log.warning(f"GFPGAN init failed: {e}")

    def enhance(self, frame_bgr):
        try:
            if self._face_enhancer:
                _, _, output = self._face_enhancer.enhance(frame_bgr, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = self._upsampler.enhance(frame_bgr, outscale=self.native_scale)
            return output
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                old_tile = self._upsampler.tile_size
                self._upsampler.tile_size = max(128, old_tile // 2)
                log.warning(f"OOM: tile {old_tile} → {self._upsampler.tile_size}")
                try:
                    output, _ = self._upsampler.enhance(frame_bgr, outscale=self.native_scale)
                    return output
                except Exception:
                    log.error("OOM retry failed, fallback to lanczos")
                    h, w = frame_bgr.shape[:2]
                    return cv2.resize(frame_bgr, (w * self.native_scale, h * self.native_scale), interpolation=cv2.INTER_LANCZOS4)
            raise

    def enhance_batch(self, frames_np, target_h, target_w):
        results = []
        for i in range(frames_np.shape[0]):
            out = self.enhance(frames_np[i])
            if out.shape[0] != target_h or out.shape[1] != target_w:
                out = cv2.resize(out, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            results.append(out)
        return np.stack(results)

    def release(self):
        del self._upsampler
        self._upsampler = None
        if self._face_enhancer:
            del self._face_enhancer
            self._face_enhancer = None
        torch.cuda.empty_cache()
