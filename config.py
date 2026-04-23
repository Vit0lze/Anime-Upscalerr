# ============================================
# CONFIGURAÇÕES E HARDWARE DETECTION
# ============================================
import os, subprocess, sys, logging, torch
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('upscaler')

# ═══════════════════════════════════════════════════════════
# DIRETÓRIOS E PATHS
# ═══════════════════════════════════════════════════════════
_BASE_ENV = os.environ.get('UPSCALER_BASE_DIR', '/content')
TEMP_DIR = Path(os.environ.get('UPSCALER_TEMP_DIR', f'{_BASE_ENV}/temp_work'))
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

BASE = Path(os.environ.get('UPSCALER_DATA_DIR', f'{_BASE_ENV}/AnimeUpscaler'))
INPUT_DIR = BASE / 'inputs'
OUTPUT_DIR = BASE / 'outputs'
PREVIEW_DIR = TEMP_DIR / 'previews'
PRESETS_DIR = BASE / 'presets'
LOGS_DIR = BASE / 'logs'

for d in [INPUT_DIR, OUTPUT_DIR, PREVIEW_DIR, PRESETS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# CONSTANTES DE VALIDAÇÃO
# ═══════════════════════════════════════════════════════════
TIMEOUTS = {'decode_total_s': 7200, 'encode_s': 3600}
RETRIES = {'decode_open': 2, 'encode': 2}
BLK = 256
PRESET_SCHEMA_VERSION = 2

VALID_ENGINES = {'none', 'bicubic', 'lanczos', 'anime4k', 'anime4k_fast', 'realesrgan', 'waifu2x'}
VALID_CODECS = {'h264_nvenc', 'hevc_nvenc', 'libx265', 'libx264'}
VALID_QUALITY_PRESETS = {'performance', 'balanced', 'quality', 'ultra', 'near_lossless'}
VALID_PIX_FMTS = {'yuv420p', 'yuv420p10le', 'yuv444p10le'}
VALID_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.webm', '.mov', '.flv', '.wmv', '.m4v', '.ts'}
VALID_REALESRGAN_MODELS = {'x4plus_anime', 'x4plus', 'x2plus', 'animevideov3'}
VALID_WAIFU2X_MODELS = {'art', 'art_scan', 'photo'}
VALID_WAIFU2X_METHODS = {'scale', 'noise', 'noise_scale'}
VALID_TILE_SIZES = {0, 128, 192, 256, 384, 512}

# ═══════════════════════════════════════════════════════════
# HARDWARE DETECTION
# ═══════════════════════════════════════════════════════════
HAS_GPU = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if HAS_GPU else 'CPU'
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0

def detect_hardware_profile(vram_gb, has_gpu=True):
    if not has_gpu: return 'low'
    if vram_gb >= 14: return 'high'
    if vram_gb >= 8: return 'mid'
    return 'low'

HARDWARE_PROFILE = detect_hardware_profile(VRAM_GB, HAS_GPU)

PERFORMANCE_LIMITS_BY_PROFILE = {
    'low': {'batchSize':{'min':1,'max':8,'default':4},'ramBuffer':{'min':1,'max':6,'default':3},'vramUsage':{'min':35,'max':65,'default':50}},
    'mid': {'batchSize':{'min':2,'max':24,'default':10},'ramBuffer':{'min':2,'max':12,'default':6},'vramUsage':{'min':45,'max':80,'default':70}},
    'high': {'batchSize':{'min':4,'max':64,'default':20},'ramBuffer':{'min':4,'max':24,'default':10},'vramUsage':{'min':55,'max':95,'default':80}},
}

def _runtime_has_gpu_type():
    env_gpu = os.environ.get('COLAB_GPU')
    if env_gpu and str(env_gpu).strip() not in ('0', '', 'None', 'none'):
        return True, f'COLAB_GPU={env_gpu}'
    try:
        completed = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10)
        output = (completed.stdout or '') + (completed.stderr or '')
        if completed.returncode == 0 and 'GPU ' in output:
            return True, output.strip().splitlines()[0]
    except Exception: pass
    return False, 'No GPU detected.'

def run_startup_gpu_validation():
    checks, errors = {}, []
    runtime_ok, detail = _runtime_has_gpu_type()
    checks['runtime_gpu_type'] = {'ok': runtime_ok, 'detail': detail}
    if not runtime_ok: errors.append('Runtime sem GPU.')
    cuda_ok = bool(torch.cuda.is_available())
    checks['torch_cuda_available'] = {'ok': cuda_ok, 'detail': str(cuda_ok)}
    if not cuda_ok: errors.append('CUDA unavailable.')
    device_ok = bool(GPU_NAME and GPU_NAME != 'CPU')
    checks['cuda_device_name'] = {'ok': device_ok, 'detail': GPU_NAME}
    alloc_ok, alloc_detail = False, 'skipped'
    if cuda_ok:
        try:
            probe = torch.zeros((1,1), device='cuda', dtype=torch.float32)
            alloc_ok = bool(probe.is_cuda and float(probe.item()) == 0.0)
            del probe; torch.cuda.synchronize(); alloc_detail = 'ok'
        except Exception as e: alloc_detail = str(e)
    checks['cuda_allocation_test'] = {'ok': alloc_ok, 'detail': alloc_detail}
    if not alloc_ok: errors.append('CUDA alloc failed.')
    return {'ok': len(errors) == 0, 'checks': checks, 'errors': errors}

GPU_RUNTIME_VALIDATION = run_startup_gpu_validation()
HEAVY_PROCESSING_BLOCKED = not GPU_RUNTIME_VALIDATION['ok']

def auto_configure_vram():
    global HARDWARE_PROFILE
    if not HAS_GPU: return
    try:
        suggested = detect_hardware_profile(VRAM_GB, HAS_GPU)
        if suggested != HARDWARE_PROFILE:
            log.warning(f'VRAM: Auto-correcting {HARDWARE_PROFILE} -> {suggested}')
            HARDWARE_PROFILE = suggested
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            max_vram = PERFORMANCE_LIMITS_BY_PROFILE[HARDWARE_PROFILE]['vramUsage']['max']
            torch.cuda.set_per_process_memory_fraction(max_vram / 100.0)
        log.info(f'VRAM: Profile={HARDWARE_PROFILE}, VRAM={VRAM_GB:.1f}GB')
    except Exception as e: log.warning(f'VRAM auto-config failed: {e}')

def log_event(level, job_id, stage, message, error=None):
    suffix = f' | erro={error}' if error else ''
    log.log(level, f'job_id={job_id} etapa={stage} {message}{suffix}')
