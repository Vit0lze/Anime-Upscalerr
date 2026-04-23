import json, uuid, shutil, base64, random, cv2
from pathlib import Path
import gradio as gr
from fastapi.responses import FileResponse
from backend.config import (
    BASE, HAS_GPU, GPU_NAME, HARDWARE_PROFILE, HEAVY_PROCESSING_BLOCKED,
    INPUT_DIR, LOGS_DIR, OUTPUT_DIR, PERFORMANCE_LIMITS_BY_PROFILE,
    PRESETS_DIR, PREVIEW_DIR, PRESET_SCHEMA_VERSION, TEMP_DIR, TIMEOUTS,
    VALID_CODECS, VALID_ENGINES, VALID_PIX_FMTS, VALID_QUALITY_PRESETS,
    VALID_REALESRGAN_MODELS, VALID_TILE_SIZES, VALID_VIDEO_EXTENSIONS,
    VALID_WAIFU2X_METHODS, VALID_WAIFU2X_MODELS, VRAM_GB, RETRIES,
    auto_configure_vram, log, log_event,
)
from backend.jobs.job import Job, PresetStore
from backend.jobs.queue import JobQueue
from backend.pipeline import gpu_pipeline as gp

job_queue = JobQueue()
preset_store = PresetStore()
SESSION_TOKEN = str(uuid.uuid4())[:16]

def _as_dict(payload):
    if isinstance(payload, dict): return payload
    if isinstance(payload, str):
        txt = payload.strip()
        if not txt: return {}
        try:
            decoded = json.loads(txt)
            return decoded if isinstance(decoded, dict) else {}
        except Exception: return {}
    return {}

def _auth_guard(payload):
    data = _as_dict(payload)
    if data.get('authToken') != SESSION_TOKEN:
        return None, {'error': 'invalid token', 'code': 'UNAUTHORIZED'}
    return data, None

def _extract_job_id(payload):
    data = _as_dict(payload)
    jid = data.get('job_id') or data.get('id') or (payload if isinstance(payload, str) and payload and payload[0] != '{' else '')
    return str(jid or '').strip(), data

def api_session_init(data):
    nvenc = HAS_GPU and any(t in GPU_NAME.lower() for t in ('nvidia', 'geforce', 'tesla', 'rtx', 'gtx', 'quadro', 'a100', 't4', 'v100'))
    
    engine_capabilities = {
        'realesrgan': {
            'sub_flags': {
                'model': VALID_REALESRGAN_MODELS,
                'tile': VALID_TILE_SIZES,
                'denoise': [0, 1]
            }
        },
        'waifu2x': {
            'sub_flags': {
                'model': VALID_WAIFU2X_MODELS,
                'method': VALID_WAIFU2X_METHODS,
                'tile': VALID_TILE_SIZES
            }
        },
        'anime4k': {
            'sub_flags': {
                'mode': ['A', 'B', 'C', 'A+A', 'B+B', 'C+C'],
                'strength': [0, 1]
            }
        },
        'color': {
            'sub_flags': {
                'method': ['basic', 'advanced'],
                'strength': [0, 1]
            }
        }
    }

    return {
        'token': SESSION_TOKEN,
        'gpu': GPU_NAME,
        'vram_gb': round(VRAM_GB, 1),
        'nvenc': nvenc,
        'hardware_profile': HARDWARE_PROFILE,
        'performance_limits': PERFORMANCE_LIMITS_BY_PROFILE[HARDWARE_PROFILE],
        'engine_capabilities': engine_capabilities,
        'jobs': job_queue.list_all(),
        'status': 'connected'
    }

def api_list_jobs(data=''):
    _, auth_err = _auth_guard(data)
    if auth_err: return auth_err
    return job_queue.list_all()

def api_create_job(file_obj, params_json='{}'):
    if not file_obj or not hasattr(file_obj, 'name'):
        return {'error': 'file required', 'code': 'MISSING_FILE'}
    ext = Path(file_obj.name).suffix.lower()
    if ext not in VALID_VIDEO_EXTENSIONS:
        return {'error': f'unsupported: {ext}', 'code': 'INVALID_FILE_TYPE'}
    
    data = _as_dict(params_json)
    if data.get('authToken') != SESSION_TOKEN:
        return {'error': 'invalid token', 'code': 'UNAUTHORIZED'}
    
    job_id = uuid.uuid4().hex[:8]
    original_name = Path(file_obj.name).name  # Store original filename (e.g., EP1.mkv)
    base_name = Path(original_name).stem
    dest = INPUT_DIR / f'{base_name}_{job_id}{ext}'
    shutil.copy(file_obj.name, dest)

    job = Job(id=job_id, name=original_name, input_path=dest, params=data, original_name=original_name)
    
    # Gerar thumbnail do vídeo uploadado
    try:
        cap = cv2.VideoCapture(str(dest))
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target = max(0, min(total - 1, total // 10))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                thumb_w = 320
                thumb_h = max(1, int(thumb_w * h / w))
                thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buf).decode('utf-8')
                job.thumbnail_url = f'data:image/jpeg;base64,{b64}'
        else:
            cap.release()
    except Exception as e:
        log.warning(f'Thumbnail generation failed for {job_id}: {e}')
    
    job_queue.register(job)

    response = job.to_dict()
    response.update({
        'queue_contract': 'ready_only',
        'queue_state': 'ready',
        'requires_render_all': True,
        'next_action': 'call_render_all_after_upload_batch',
        'ordering_hint': 'render_all enqueues jobs in ascending seq (FIFO)'
    })
    return response

def api_get_job(data=''):
    jid, parsed = _extract_job_id(data)
    _, auth_err = _auth_guard(parsed)
    if auth_err: return auth_err
    j = job_queue.get(jid)
    return j.to_dict() if j else {'error': 'not found', 'code': 'JOB_NOT_FOUND'}

def api_cancel_job(data=''):
    jid, parsed = _extract_job_id(data)
    _, auth_err = _auth_guard(parsed)
    if auth_err: return auth_err
    if job_queue.cancel(jid):
        return {'status': 'cancelled', 'job_id': jid}
    return {'error': 'could not cancel', 'code': 'CANCEL_FAILED'}

def api_list_presets(data=''):
    _, auth_err = _auth_guard(data)
    if auth_err: return auth_err
    return {'status': 'ok', 'items': preset_store.list()}

def api_get_output(payload, request: gr.Request):
    data, auth_err = _auth_guard(payload)
    if auth_err: return auth_err
    jid = data.get('job_id')
    j = job_queue.get(jid)
    if j and j.output_path:
        output_path = Path(j.output_path)
        # Validate file exists AND has content > 0 bytes
        if output_path.exists() and output_path.stat().st_size > 0:
            base_url = str(getattr(request, "base_url", "")).rstrip("/")
            download_url = f"{base_url}/download/{jid}?authToken={SESSION_TOKEN}" if base_url else f"/download/{jid}?authToken={SESSION_TOKEN}"
            return {
                'status': 'ready',
                'output_url': download_url,
                'download_url': download_url,
                'output_path': str(output_path),
                'file_size': output_path.stat().st_size,
            }
        # File missing or empty - return error with details
        if output_path.exists():
            file_size = output_path.stat().st_size
            return {
                'error': 'output file empty', 'code': 'OUTPUT_EMPTY',
                'status': 'failed', 'output_url': '', 'download_url': '',
                'output_path': str(output_path),
                'file_size': file_size,
                'job_status': j.status,
                'job_error': j.error
            }
        return {
            'error': 'output file not found', 'code': 'OUTPUT_NOT_FOUND',
            'status': j.status, 'output_url': '', 'download_url': '',
            'output_path': str(j.output_path) if j.output_path else '',
            'job_status': j.status,
            'job_error': j.error
        }
    return {
        'error': 'not ready', 'code': 'OUTPUT_NOT_READY',
        'status': 'pending', 'output_url': '', 'download_url': '', 'output_path': ''
    }

# ── Endpoints que faltavam ──────────────────────────────────

def api_get_system_stats(data=''):
    _, auth_err = _auth_guard(data)
    if auth_err: return auth_err
    import shutil as _shutil
    try:
        disk = _shutil.disk_usage(BASE)
        disk_total = round(disk.total / 1e9, 1)
        disk_used = round(disk.used / 1e9, 1)
        disk_free = round(disk.free / 1e9, 1)
    except Exception:
        disk_total = disk_used = disk_free = 0
    return {
        'disk_total_gb': disk_total,
        'disk_used_gb': disk_used,
        'disk_free_gb': disk_free,
        'vram_total_gb': round(VRAM_GB, 1),
        'gpu_name': GPU_NAME,
        'hardware_profile': HARDWARE_PROFILE,
        'performance_limits': PERFORMANCE_LIMITS_BY_PROFILE[HARDWARE_PROFILE],
    }

def api_delete_job(data=''):
    jid, parsed = _extract_job_id(data)
    _, auth_err = _auth_guard(parsed)
    if auth_err: return auth_err
    with job_queue._lock:
        j = job_queue.jobs.get(jid)
        if not j:
            return {'error': 'not found', 'code': 'JOB_NOT_FOUND'}
        if j.status in Job.TERMINAL_STATUSES:
            job_queue._purge_job_files(j)
            job_queue.jobs.pop(jid, None)
            return {'status': 'deleted', 'job_id': jid}

        j.cancel_requested = True
        j.delete_requested = True
        if j.status in ('ready', 'queued'):
            job_queue._set_status(j, 'cancelled')
            j.error = 'Cancelado'
            job_queue._purge_job_files(j)
            job_queue.jobs.pop(jid, None)
            return {'status': 'deleted', 'job_id': jid}

    return {'status': 'deleting', 'job_id': jid}

def api_save_preset(data=''):
    parsed, auth_err = _auth_guard(data)
    if auth_err: return auth_err
    preset_data = {k: v for k, v in parsed.items() if k != 'authToken'}
    return preset_store.save(preset_data)

def api_delete_preset(data=''):
    parsed, auth_err = _auth_guard(data)
    if auth_err: return auth_err
    name = parsed.get('name')
    if not name:
        return {'error': 'name required', 'code': 'MISSING_NAME'}
    if preset_store.delete(name):
        return {'status': 'deleted', 'name': name}
    return {'error': 'preset not found', 'code': 'PRESET_NOT_FOUND'}

def api_cancel_all(data=''):
    _, auth_err = _auth_guard(data)
    if auth_err: return auth_err
    marked_for_cancel = []
    cancelled_immediately = []
    already_terminal = []
    with job_queue._lock:
        for j in job_queue.jobs.values():
            if j.status in Job.TERMINAL_STATUSES:
                already_terminal.append(j.id)
                continue

            j.cancel_requested = True
            if j.status in ('ready', 'queued'):
                job_queue._set_status(j, 'cancelled')
                j.error = 'Cancelado (cancel_all)'
                cancelled_immediately.append(j.id)
            else:
                # processing/muxing (and any other non-terminal in-flight states)
                # are cancelled on the next worker checkpoint.
                marked_for_cancel.append(j.id)


    jobs_snapshot = job_queue.list_all()

    return {
        'status': 'ok',
        'marked_for_cancel': len(marked_for_cancel),
        'cancelled_immediately': len(cancelled_immediately),
        'already_terminal': len(already_terminal),
        'job_ids': {
            'marked_for_cancel': marked_for_cancel,
            'cancelled_immediately': cancelled_immediately,
            'already_terminal': already_terminal,
        },
        # Canonical backend snapshot so wrapper can replace local state
        # without merging stale local queue data.
        'jobs': jobs_snapshot,
    }

def api_extract_frame(data=''):
    parsed, auth_err = _auth_guard(data)
    if auth_err: return auth_err
    jid = parsed.get('job_id')
    percent = float(parsed.get('percent', 0))
    frame_params = parsed.get('params', parsed)
    j = job_queue.get(jid)
    if not j:
        return {'error': 'job not found', 'code': 'JOB_NOT_FOUND'}
    if not j.input_path or not Path(j.input_path).exists():
        return {'error': 'input file missing', 'code': 'FILE_MISSING'}
    cap = cv2.VideoCapture(str(j.input_path))
    if not cap.isOpened():
        return {'error': 'cannot open video', 'code': 'VIDEO_OPEN_FAILED'}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(0, min(total - 1, int((total - 1) * (percent / 100.0))))
    cap.release()

    # Tenta GPU pipeline preview
    frame = gp.preview_frame_gpu(j.input_path, percent, frame_params)
    if frame is None:
        # Fallback CPU: extrai frame cru sem processamento
        cap = cv2.VideoCapture(str(j.input_path))
        if not cap.isOpened():
            return {'error': 'cannot open video', 'code': 'VIDEO_OPEN_FAILED'}
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return {'error': 'frame extraction failed', 'code': 'FRAME_FAILED'}

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode('utf-8')
    return {'url': f'data:image/jpeg;base64,{b64}', 'frame_index': target}



def api_render_job(data=''):
    parsed, auth_err = _auth_guard(data)
    if auth_err: return auth_err
    if HEAVY_PROCESSING_BLOCKED:
        return {'error': 'GPU runtime unavailable', 'code': 'GPU_RUNTIME_UNAVAILABLE'}
    jid = parsed.get('job_id')
    j = job_queue.get(jid)
    if not j:
        return {'error': 'job not found', 'code': 'JOB_NOT_FOUND'}
    
    enqueued_before = []
    with job_queue._lock:
        force = bool(parsed.get('force', False))
        if j.status == 'completed' and not force:
            return {
                'error': 'job already completed',
                'code': 'JOB_ALREADY_COMPLETED',
                'job_id': jid
            }
        
        # Se já estiver na fila ou processando, ignorar
        if j.status in ('processing', 'muxing', 'queued'):
            return {'error': 'job already in progress', 'code': 'JOB_BUSY'}

        # Parâmetros atuais da interface enviados no request
        new_params = {k: v for k, v in parsed.items() if k not in ('authToken', 'job_id', 'force')}

        # Encontrar jobs "ready" anteriores para garantir ordem FIFO
        ready_before = sorted(
            [candidate for candidate in job_queue.jobs.values() 
             if candidate.status == 'ready' and candidate.seq < j.seq],
            key=lambda candidate: candidate.seq
        )
        
        for candidate in ready_before:
            if new_params:
                import copy
                candidate.params.update(copy.deepcopy(new_params))
            candidate.reset_for_requeue()
            job_queue._set_status(candidate, 'queued')
            job_queue.queue.put(candidate.id)
            enqueued_before.append(candidate.id)

        # Atualizar parâmetros do job alvo e enfileirar
        if new_params:
            j.params.update(new_params)
        
        j.reset_for_requeue()
        job_queue._set_status(j, 'queued')
        job_queue.queue.put(j.id)

    response = j.to_dict()
    if enqueued_before:
        response['enqueued_before'] = enqueued_before
    return response

def api_render_all(data=''):
    parsed, auth_err = _auth_guard(data)
    if auth_err:
        return auth_err
    if HEAVY_PROCESSING_BLOCKED:
        return {'error': 'GPU runtime unavailable', 'code': 'GPU_RUNTIME_UNAVAILABLE'}

    enqueued = []
    skipped = []

    with job_queue._lock:
        # Ordenar rigorosamente por sequência de criação (seq)
        jobs_in_order = sorted(job_queue.jobs.values(), key=lambda x: x.seq)
        for j in jobs_in_order:
            if j.status in ('ready', 'completed', 'failed', 'cancelled'):
                # Só enfileira se for explicitamente 'ready' ou se for um retry
                # Aqui vamos focar em quem está 'ready' (fluxo padrão)
                if j.status == 'ready':
                    j.reset_for_requeue()
                    job_queue._set_status(j, 'queued')
                    job_queue.queue.put(j.id)
                    enqueued.append(j.id)
                else:
                    skipped.append({'id': j.id, 'status': j.status})
            else:
                # Já está queued ou processing
                skipped.append({'id': j.id, 'status': j.status})

    return {'status': 'ok', 'enqueued': enqueued, 'skipped': skipped}
