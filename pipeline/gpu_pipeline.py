import cv2, time
import numpy as np
from pathlib import Path
from backend.engines import gpu_kernels as gk, color_engine as ce, anime4k_engine as a4k, cache
from .ffmpeg import FFmpegPipeWriter
from .helpers import compute_output_dims, calc_optimal_batch, do_upscale_batch, denoise_cpu

def get_color_engine(H, W, bs): 
    return cache._engine_cache.get_or_create(('color',H,W,bs), lambda: ce.GPUColorEngineCUDA(H,W,bs))

def get_anime4k_engine(Hi, Wi, Ho, Wo, bs): 
    return cache._engine_cache.get_or_create(('a4k',Hi,Wi,Ho,Wo,bs), lambda: a4k.Anime4KEngineCUDA(Hi,Wi,Ho,Wo,bs))

def process_gpu_pipeline(input_path, output_path, params, log_fn=None, progress_fn=None, cancel_check=None):
    if not gk.GPU_OK: return {'error':'GPU not init','code':'GPU_NOT_READY'}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened(): return {'error':'Failed to open video','code':'VIDEO_OPEN_FAILED'}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ucfg = params.get('upscale',{}); perf = params.get('performance',{}); enc = params.get('encoder',{})
    Ho, Wo = compute_output_dims(H, W, ucfg.get('targetHeight',1080))
    engine = str(ucfg.get('engine','anime4k')).lower()
    denoise_val = float(params.get('fx',{}).get('denoise', 0) or 0)
    
    gpu_mem_gb = 15.6
    try: gpu_mem_gb = gk.cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']/(1024**3)
    except: pass
    batch_size = max(1, min(perf.get('batchSize',8), calc_optimal_batch(H,W,Ho,Wo,gpu_mem_gb*(perf.get('vramUsage',80)/100.0))))
    if log_fn: log_fn(f"GPU Pipeline: {W}x{H}→{Wo}x{Ho} | engine={engine} | batch={batch_size}")
    
    has_color = any(params.get('color',{}).get(k,0)!=0 for k in ('exposure','contrast','saturation','hslGreen','hslBlue','hslSkin'))
    cg = get_color_engine(H,W,batch_size) if (has_color or denoise_val > 0) else None
    a4k_native, a4k_post = None, None
    if engine == 'anime4k': a4k_native = get_anime4k_engine(H,W,Ho,Wo,batch_size)
    elif engine in ('anime4k_fast',) or ucfg.get('anime4kLines',False): a4k_post = get_anime4k_engine(Ho,Wo,Ho,Wo,batch_size)
    writer = FFmpegPipeWriter(output_path, Wo, Ho, fps, input_path, enc, perf.get('ramBuffer',4))

    done = 0; t0 = time.time(); last_progress = 0; cancelled = False
    try:
        while True:
            if cancel_check and cancel_check():
                cancelled = True
                if log_fn: log_fn("Cancelado")
                break
            batch = []
            for _ in range(batch_size):
                ret, fr = cap.read()
                if not ret: break
                batch.append(fr)
            if not batch: break
            batch_np = np.stack(batch)
            if cg: batch_np = cg.process(batch_np, params)
            elif denoise_val > 0: batch_np = denoise_cpu(batch_np, int(denoise_val))
            batch_np = do_upscale_batch(batch_np, engine, H, W, Ho, Wo, params, a4k_native, a4k_post)
            writer.write(batch_np); done += batch_np.shape[0]
            if progress_fn and (done - last_progress) >= batch_size * 2:
                elapsed = max(0.001, time.time()-t0); cfps = done/elapsed
                progress_fn(done/max(total,1), cfps, (total-done)/max(cfps,0.1)); last_progress = done
    except Exception as e:
        if log_fn: log_fn(f"Pipeline error: {e}")
        return {'error':str(e),'code':'PIPELINE_ERROR'}
    finally:
        cap.release()
        writer.close(force=cancelled)
        if gk.mempool: gk.mempool.free_all_blocks()

    if cancelled:
        return {'status':'cancelled','frames':done}

    elapsed = time.time()-t0; avg_fps = done/max(elapsed,0.001)
    if log_fn: log_fn(f"✅ Done {elapsed/60:.1f}min | {avg_fps:.1f}fps | {done} frames")
    return {'status':'completed','fps':avg_fps,'elapsed':elapsed,'frames':done}

def preview_frame_gpu(input_path, percent, params):
    if not gk.GPU_OK: return None
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened(): return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target = max(0, min(total-1, int((total-1)*(percent/100.0))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target); ret, frame = cap.read(); cap.release()
    if not ret: return None
    ucfg = params.get('upscale',{}); Ho, Wo = compute_output_dims(H, W, ucfg.get('targetHeight',1080))
    engine = str(ucfg.get('engine','anime4k')).lower()
    batch = np.expand_dims(frame, axis=0)
    has_color = any(params.get('color',{}).get(k,0)!=0 for k in ('exposure','contrast','saturation','hslGreen','hslBlue','hslSkin'))
    denoise_val = float(params.get('fx', {}).get('denoise', 0) or 0)
    if (has_color or denoise_val > 0): batch = get_color_engine(H,W,1).process(batch, params)
    a4k_native, a4k_post = None, None
    if engine == 'anime4k': a4k_native = get_anime4k_engine(H,W,Ho,Wo,1)
    elif engine in ('anime4k_fast',) or ucfg.get('anime4kLines',False): a4k_post = get_anime4k_engine(Ho,Wo,Ho,Wo,1)
    batch = do_upscale_batch(batch, engine, H, W, Ho, Wo, params, a4k_native, a4k_post)
    if gk.mempool: gk.mempool.free_all_blocks()
    return batch[0]
