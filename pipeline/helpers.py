import cv2, torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from backend.config import log
from backend.engines import cache
from backend.engines import realesrgan_engine as resrgan_eng
from backend.engines import waifu2x_engine as w2x_eng
from backend.engines import gpu_kernels as gk

def calc_optimal_batch(H, W, Ho, Wo, vram_gb=15.6, safety=0.80):
    bpf = (H*W*3 + Ho*Wo*3 + max(H*W, Ho*Wo)*2) * 4
    return min(max(1, int(vram_gb * 1024**3 * safety / bpf)), 128)

def compute_output_dims(H, W, target_h):
    Ho = target_h; Wo = int(Ho * W / H)
    if Wo % 2 != 0: Wo += 1
    if Ho % 2 != 0: Ho += 1
    return Ho, Wo

def upscale_gpu_torch(frames_np, Ho, Wo, mode='bicubic'):
    if frames_np.shape[1] == Ho and frames_np.shape[2] == Wo: return frames_np
    try:
        t = torch.from_numpy(frames_np).to(device='cuda', dtype=torch.float32)
        t = t.permute(0,3,1,2) / 255.0
        out = torch.nn.functional.interpolate(t, size=(Ho,Wo), mode=mode, 
                                            align_corners=False if mode=='bicubic' else None, 
                                            antialias=True if mode=='bicubic' else False)
        return torch.clamp(out * 255.0, 0, 255).byte().permute(0,2,3,1).cpu().numpy()
    except Exception as e:
        log.error(f"upscale_gpu_torch fatal error: {e}")
        raise e

def upscale_cpu(frames_np, Ho, Wo, method="lanczos"):
    if frames_np.shape[1] == Ho and frames_np.shape[2] == Wo: return frames_np
    interp = cv2.INTER_LANCZOS4 if method == "lanczos" else cv2.INTER_CUBIC
    out = np.empty((frames_np.shape[0], Ho, Wo, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=6) as ex:
        def rz(i): out[i] = cv2.resize(frames_np[i], (Wo,Ho), interpolation=interp)
        list(ex.map(rz, range(frames_np.shape[0])))
    return out

def denoise_cpu(frames_np, strength=5):
    if strength <= 0: return frames_np
    out = np.empty_like(frames_np)
    with ThreadPoolExecutor(max_workers=4) as ex:
        def dn(i): out[i] = cv2.fastNlMeansDenoisingColored(frames_np[i], None, strength, strength, 5, 15)
        list(ex.map(dn, range(frames_np.shape[0])))
    return out

def _parse_realesrgan_params(params):
    u = params.get('upscale') or {}
    return {
        'model_name': u.get('realesrganModel', 'x4plus_anime'),
        'denoise_strength': float(u.get('realesrganDenoise', 0.5) or 0.5),
        'face_enhance': bool(u.get('realesrganFaceEnhance', False)),
        'tile': int(u.get('realesrganTile', 0) or 0),
        'half': bool(u.get('realesrganHalf', True)),
    }

def _parse_waifu2x_params(params):
    u = params.get('upscale') or {}
    return {
        'model_type': u.get('waifu2xModel', 'art'),
        'noise_level': int(u.get('waifu2xNoiseLevel', 2) or 2),
        'method': u.get('waifu2xMethod', 'noise_scale'),
        'tile': int(u.get('waifu2xTile', 0) or 0),
    }

def get_realesrgan_engine(params):
    p = _parse_realesrgan_params(params)
    key = ('realesrgan_pt', p['model_name'], p['tile'], p['half'], p['face_enhance'])
    return cache._engine_cache.get_or_create(key, lambda: resrgan_eng.RealESRGANEnginePyTorch(
        model_name=p['model_name'], tile=p['tile'], half=p['half'],
        denoise_strength=p['denoise_strength'], face_enhance=p['face_enhance']
    ))

def get_waifu2x_engine(params):
    p = _parse_waifu2x_params(params)
    key = ('waifu2x_pt', p['model_type'], p['noise_level'], p['method'], p['tile'])
    return cache._engine_cache.get_or_create(key, lambda: w2x_eng.Waifu2xEnginePyTorch(
        model_type=p['model_type'], noise_level=p['noise_level'], method=p['method'], tile=p['tile']
    ))

def do_upscale_batch(batch_np, engine, H, W, Ho, Wo, params, a4k_native, a4k_post):
    if engine == 'anime4k' and a4k_native:
        return a4k_native.process_upscale(batch_np, params)

    if engine == 'realesrgan':
        if gk.mempool: gk.mempool.free_all_blocks()
        try:
            eng = get_realesrgan_engine(params)
            return eng.enhance_batch(batch_np, Ho, Wo)
        except Exception as e:
            log.error(f"Real-ESRGAN batch failed: {e}, fallback to bicubic")
            return upscale_gpu_torch(batch_np, Ho, Wo, mode='bicubic')

    if engine == 'waifu2x':
        if gk.mempool: gk.mempool.free_all_blocks()
        try:
            eng = get_waifu2x_engine(params)
            return eng.enhance_batch(batch_np, Ho, Wo)
        except Exception as e:
            log.error(f"Waifu2x batch failed: {e}, fallback to bicubic")
            return upscale_gpu_torch(batch_np, Ho, Wo, mode='bicubic')

    gpu_mode = 'bicubic'
    if engine == 'lanczos': gpu_mode = 'area'

    if H != Ho or W != Wo:
        batch_np = upscale_gpu_torch(batch_np, Ho, Wo, mode=gpu_mode)

    if a4k_post:
        batch_np = a4k_post.apply_lines_only(batch_np, params)
    return batch_np
