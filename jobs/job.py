import threading, queue, json, uuid, shutil, time, logging, re, atexit, copy
from datetime import datetime
from pathlib import Path
from backend.config import log, log_event, LOGS_DIR, PRESETS_DIR, OUTPUT_DIR, TEMP_DIR, PRESET_SCHEMA_VERSION

# ═══════════════════════════════════════════════════════════
# JSONL LOGGER
# ═══════════════════════════════════════════════════════════
class JSONLLogger:
    def __init__(self, log_dir=None):
        self.log_dir=Path(log_dir) if log_dir else LOGS_DIR
        self.log_dir.mkdir(parents=True,exist_ok=True)
        self._lock=threading.Lock(); self._current_date=None; self._file=None
    def _get_file(self):
        today=datetime.now().strftime('%Y-%m-%d')
        if self._current_date!=today or self._file is None:
            if self._file:
                try: self._file.close()
                except: pass
            self._current_date=today; self._file=open(self.log_dir/f'jobs_{today}.jsonl','a',encoding='utf-8')
        return self._file
    def log(self, event_type, job_id=None, stage=None, data=None, level='info'):
        entry={'timestamp':datetime.now().isoformat(),'level':level,'event':event_type}
        if job_id: entry['job_id']=job_id
        if stage: entry['stage']=stage
        if data: entry['data']=data
        with self._lock:
            try: f=self._get_file(); f.write(json.dumps(entry,ensure_ascii=False)+'\n'); f.flush()
            except Exception as e: log.warning(f'JSONL write failed: {e}')
    def log_job_status(self, job_id, old, new): self.log('job_status_change',job_id=job_id,data={'from':old,'to':new})
    def log_job_progress(self, job_id, progress, fps=None, eta=None, metrics=None):
        self.log('job_progress',job_id=job_id,data={'progress':progress,'fps':fps,'eta':eta,**(metrics or {})})
    def log_job_error(self, job_id, error, stage=None):
        self.log('job_error',job_id=job_id,stage=stage,data={'error':str(error)},level='error')
    def close(self):
        with self._lock:
            if self._file:
                try: self._file.close()
                except: pass
                self._file=None

jsonl_logger = JSONLLogger()
atexit.register(jsonl_logger.close)

# ═══════════════════════════════════════════════════════════
# JOB MODEL
# ═══════════════════════════════════════════════════════════
class Job:
    TERMINAL_STATUSES = {'completed', 'failed', 'cancelled'}
    _seq_counter = 0
    _seq_lock = threading.Lock()

    def __init__(self, id, name, input_path, params=None, original_name=None):
        self.id = id
        self.name = name
        self.input_path = Path(input_path)
        self.original_name = original_name or name
        self.params = params or {}
        self.status = 'ready'
        self.progress = 0
        self.fps = None
        self.eta = None
        self.error = None
        self.output_path = None
        self.preview_frames = []
        self.thumbnail_url = None
        self.cancel_requested = False
        self.delete_requested = False
        self.execution_backend = 'cpu'
        self.backend_reason = 'not_processed_yet'
        self.backend_warning = None
        self.created = datetime.now().isoformat()
        with Job._seq_lock:
            Job._seq_counter += 1
            self.seq = Job._seq_counter
        self.metrics = self._default_metrics()

    @staticmethod
    def _default_metrics():
        return {
            'avg_fps': None,
            'stage_fps': None,
            'encode_fps': None,
            'final_fps': None,
            'bottleneck': 'idle',
            'eta_seconds': None,
            'throughput_samples': 0,
            'stage_elapsed_s': {
                'decode': 0.0,
                'process': 0.0,
                'upscale': 0.0,
                'encode': 0.0
            }
        }

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status,
            'progress': self.progress,
            'fps': self.fps,
            'eta': self.eta,
            'error': self.error,
            'output_path': str(self.output_path) if self.output_path else None,
            'preview_url': self.thumbnail_url,
            'size': self.input_path.stat().st_size if self.input_path.exists() else 0,
            'seq': self.seq,
            'created': self.created,
            'metrics': copy.deepcopy(self.metrics),
            'execution_backend': self.execution_backend,
            'backend_reason': self.backend_reason,
            'backend_warning': self.backend_warning,
        }

    def reset_for_requeue(self):
        self.cancel_requested = False
        self.delete_requested = False
        self.error = None
        self.progress = 0
        self.fps = None
        self.eta = None
        self.output_path = None
        self.metrics = self._default_metrics()

# ═══════════════════════════════════════════════════════════
# PRESET STORE
# ═══════════════════════════════════════════════════════════
class PresetStore:
    def __init__(self): self.dir=PRESETS_DIR; self.dir.mkdir(parents=True,exist_ok=True)
    @staticmethod
    def _safe(name):
        s=re.sub(r'[^\w\s-]','',name.lower().strip()); return re.sub(r'[\s-]+','_',s) or 'unnamed'
    @staticmethod
    def _norm(data):
        data.setdefault('version',PRESET_SCHEMA_VERSION); data.setdefault('description','')
        data.setdefault('metadata',{}); data['metadata'].setdefault('created',datetime.now().isoformat())
        data['metadata'].setdefault('builtin',False); data['metadata']['updated']=datetime.now().isoformat()
        if 'encode' in data and 'encoder' not in data: data['encoder']=data.pop('encode')
        if 'params' in data and isinstance(data['params'],dict):
            p=data.pop('params')
            for s in ('color','fx','upscale','encoder','performance'):
                if s in p and s not in data: data[s]=p[s]
        return data
    def list(self):
        presets=[]
        for f in sorted(self.dir.glob('*.json'),key=lambda p: p.stem):
            try: d=json.loads(f.read_text()); d.setdefault('name',f.stem); presets.append(PresetStore._norm(d))
            except Exception as e: log.warning(f'Preset load fail {f.name}: {e}')
        return presets
    def save(self, preset):
        if not isinstance(preset,dict) or not preset.get('name'): return {'error':'name required','code':'INVALID_PRESET'}
        preset=PresetStore._norm(preset)
        (self.dir/f'{PresetStore._safe(preset["name"])}.json').write_text(json.dumps(preset,indent=2,ensure_ascii=False))
        return preset
    def delete(self, name):
        f=self.dir/f'{PresetStore._safe(name)}.json'
        if f.exists(): f.unlink(); return True
        return False
