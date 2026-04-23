import threading, queue, logging, time, shutil, cv2, copy
from pathlib import Path
from backend.config import log, log_event, RETRIES, TIMEOUTS, OUTPUT_DIR, TEMP_DIR
from backend.engines import gpu_kernels as gk, color_engine as ce
from backend.pipeline import gpu_pipeline as gp
from .job import Job, jsonl_logger

class JobQueue:
    VALID_TRANSITIONS = {
        'ready':{'queued','cancelled'},'queued':{'processing','cancelled'},
        'processing':{'muxing','completed','failed','cancelled'},'muxing':{'completed','failed','cancelled'},
        'failed':{'queued'},'completed':{'queued'},'cancelled':{'queued'},
    }
    def __init__(self):
        self._lock=threading.RLock(); self.jobs={}; self.queue=queue.Queue()
        self.worker=threading.Thread(target=self._worker,daemon=True); self.worker.start()

    def _set_status(self, job, new):
        cur=job.status
        if cur==new: return True
        if new not in self.VALID_TRANSITIONS.get(cur,set()):
            log_event(logging.WARNING,job.id,'status',f'invalid {cur}->{new}'); return False
        old=job.status; job.status=new
        try: jsonl_logger.log_job_status(job.id,old,new)
        except Exception: pass
        return True

    def add(self, job):
        with self._lock: self.jobs[job.id]=job
        self.queue.put(job.id)

    def register(self, job):
        with self._lock: job.status='ready'; self.jobs[job.id]=job

    def get(self, job_id):
        with self._lock:
            job = self.jobs.get(job_id)
            if job and job.delete_requested and job.status not in Job.TERMINAL_STATUSES:
                return None
            return job

    def list_all(self):
        with self._lock:
            return [
                j.to_dict() for j in self.jobs.values()
                if not (j.delete_requested and j.status not in Job.TERMINAL_STATUSES)
            ]

    def cancel(self, job_id):
        with self._lock:
            j=self.jobs.get(job_id)
            if not j or j.status in Job.TERMINAL_STATUSES: return False
            j.cancel_requested=True
            if j.status in ('ready','queued'):
                self._set_status(j,'cancelled')
                j.error='Cancelado'
            return True

    def _worker(self):
        log.info("Worker thread started.")
        while True:
            try:
                job_id = self.queue.get()
                log.info(f"Worker picked up job: {job_id}")
                
                with self._lock:
                    job = self.jobs.get(job_id)
                    if not job:
                        log.warning(f"Job {job_id} not found in store, skipping.")
                        self.queue.task_done()
                        continue
                    
                    if job.status in Job.TERMINAL_STATUSES:
                        log.info(f"Job {job_id} is already in terminal state ({job.status}), skipping.")
                        self.queue.task_done()
                        continue

                    if job.cancel_requested:
                        log.info(f"Job {job_id} was cancelled before starting.")
                        self._set_status(job, 'cancelled')
                        job.error = 'Cancelado antes do início'
                        self.queue.task_done()
                        continue

                # Process the job
                log.info(f"Starting processing for job {job_id}")
                self._process(job)
                log.info(f"Finished processing sequence for job {job_id}")

            except Exception as e:
                log.error(f"Critical error in worker loop: {e}", exc_info=True)
                time.sleep(1) # Prevent tight loop on persistent errors
            finally:
                try:
                    # Clean up after processing if needed
                    with self._lock:
                        job = self.jobs.get(job_id)
                        if job and job.delete_requested and job.status in Job.TERMINAL_STATUSES:
                            self._purge_job_files(job)
                            self.jobs.pop(job_id, None)
                except Exception as e:
                    log.warning(f"Cleanup error for job {job_id}: {e}")
                
                # IMPORTANT: task_done must be called to allow queue.join() if used
                try: self.queue.task_done()
                except: pass

    def _process(self, job):
        with self._lock:
            # Check status again inside lock right before processing
            if job.status in Job.TERMINAL_STATUSES or job.cancel_requested:
                if job.cancel_requested:
                    self._set_status(job, 'cancelled')
                return

            if not self._set_status(job, 'processing'):
                log.warning(f"Could not transition job {job.id} to processing state.")
                return

        try:
            if gk.GPU_OK:
                self._process_gpu(job)
            else:
                self._process_cpu(job)
        except Exception as e:
            log.error(f"Error during _process for job {job.id}: {e}", exc_info=True)
            with self._lock:
                self._set_status(job, 'failed')
                job.error = str(e)


    def _process_gpu(self, job):
        job.execution_backend='gpu'
        job.backend_reason='gpu_pipeline_v18'
        job.backend_warning=None
        # Use original_name if available, otherwise fall back to input_path stem
        base_name = Path(job.original_name).stem if job.original_name else Path(job.input_path).stem
        output_path=OUTPUT_DIR/f'{base_name}_{job.id}_upscaled.mkv'
        
        import torch
        
        def _progress(pct, fps, eta):
            job.progress=int(pct*100)
            job.fps=round(fps,2) if fps is not None else None
            if eta is not None and isinstance(eta, (int, float)):
                job.eta = f'{int(eta//60)}m{int(eta%60):02d}s'
            else:
                job.eta = eta
            if fps is not None: job.metrics['avg_fps']=round(fps,2)
            if eta is not None:
                try: job.metrics['eta_seconds']=int(eta) if isinstance(eta,(int,float)) else None
                except Exception: pass
            job.metrics['bottleneck']='gpu_pipeline'

        result=gp.process_gpu_pipeline(input_path=job.input_path,output_path=str(output_path),params=job.params,
                                   log_fn=lambda msg: log_event(logging.INFO,job.id,'gpu_pipeline',msg),
                                   progress_fn=_progress,cancel_check=lambda: job.cancel_requested)

        # Validate output file after processing
        output_valid = False
        if result.get('status') == 'completed' and not job.cancel_requested:
            if output_path.exists() and output_path.stat().st_size > 0:
                output_valid = True
                job.output_path=str(output_path)
                job.progress=100
                job.eta='0m00s'
                self._set_status(job,'completed')
            else:
                # Pipeline reported success but file is missing or empty
                file_size = output_path.stat().st_size if output_path.exists() else 0
                error_msg = f"Pipeline completed but output file is {'empty' if file_size == 0 else 'missing'}"
                log_event(logging.ERROR, job.id, 'output_validation', error_msg)
                job.error = error_msg
                result['status'] = 'failed'
                result['error'] = error_msg

        with self._lock:
            if result.get('status') == 'cancelled' or job.cancel_requested:
                self._set_status(job,'cancelled')
                job.error='Cancelado'
                try:
                    if output_path.exists():
                        output_path.unlink(missing_ok=True)
                except Exception as e:
                    log_event(logging.WARNING, job.id, 'cleanup', f'Partial output cleanup failed: {e}')
            elif not output_valid and not job.error:
                self._set_status(job,'failed')
                job.error=result.get('error','GPU pipeline failed')
        
        # Liberar memória GPU após conclusão do job
        try:
            from backend.engines.cache import _engine_cache
            _engine_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_event(logging.INFO, job.id, 'cleanup', 'GPU memory released')
        except Exception as e:
            log_event(logging.WARNING, job.id, 'cleanup', f'Memory cleanup failed: {e}')

    def _process_cpu(self, job):
        # Implementação simplificada para o módulo. 
        # A lógica CPU original do app.py era muito extensa e acoplada.
        # Em produção, o foco é o GPU Pipeline.
        with self._lock:
            self._set_status(job, 'failed')
            job.error = "CPU Processing not fully implemented in modular mode. Use GPU."

    @staticmethod
    def _purge_job_files(job):
        for path in (job.input_path, job.output_path):
            try:
                if path and Path(path).exists():
                    Path(path).unlink(missing_ok=True)
            except Exception:
                pass
