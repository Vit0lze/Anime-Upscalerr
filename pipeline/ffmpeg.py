import subprocess, threading, queue as queue_mod
from backend.config import log

class FFmpegPipeWriter:
    def __init__(self, output_path, W, H, fps, input_path, encoder_params, ram_buffer_gb=4):
        pix_fmt = encoder_params.get('pixFmt', 'yuv420p10le')
        cmd = ["ffmpeg","-y","-loglevel","error","-f","rawvideo","-vcodec","rawvideo",
               "-s",f"{W}x{H}","-pix_fmt","bgr24","-r",str(fps),"-i","pipe:0",
               "-err_detect","ignore_err","-i",str(input_path)]
        cmd += self._build_encoder_args(encoder_params)
        cmd += ["-map","0:v:0","-map","1:a?","-c:a","copy","-map","1:s?","-c:s","copy",
                "-map_chapters","1","-map_metadata","1","-pix_fmt",pix_fmt,str(output_path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._q = queue_mod.Queue(maxsize=max(4, int(ram_buffer_gb * 2)))
        self._th = threading.Thread(target=self._drain, daemon=True); self._th.start()
        self._error = None

    @staticmethod
    def _build_encoder_args(params):
        codec = params.get('codec','hevc_nvenc')
        crf = params.get('crf', 28)  # Higher CRF = smaller file, 28 is good balance
        preset = str(params.get('qualityPreset','quality'))  # Default to quality preset for better compression
        if 'nvenc' in codec:
            nvenc_presets={'performance':'p1','balanced':'p4','quality':'p5','ultra':'p6','near_lossless':'p7'}
            # Use CBR for more consistent bitrate, lower maxrate for compression
            return ["-c:v",codec,"-preset",nvenc_presets.get(preset,'p5'),"-rc","vbr","-cq",str(crf),"-maxrate","8M","-bufsize","16M"]
        elif codec == 'libx265':
            x265_presets = {'performance':'fast','balanced':'medium','quality':'slow','ultra':'slower','near_lossless':'veryslow'}
            return ["-c:v","libx265","-preset",x265_presets.get(preset,'slow'),"-crf",str(crf)]
        return ["-c:v","libx264","-preset","slow","-crf",str(crf)]

    def _drain(self):
        while True:
            data = self._q.get()
            if data is None: self._q.task_done(); break
            try:
                if self._error is None: self.proc.stdin.write(data)
            except Exception as e:
                if self._error is None: self._error = str(e)
            self._q.task_done()

    def write(self, frames_np): self._q.put(frames_np.tobytes())

    def close(self, force=False):
        if force:
            try: self.proc.terminate()
            except: pass
            try: self.proc.stdin.close()
            except: pass
            try: self.proc.stderr.close()
            except: pass
            self._q.put(None); self._th.join(timeout=3)
            try: self.proc.wait(timeout=5)
            except:
                try: self.proc.kill()
                except: pass
            return
        else:
            self._q.put(None); self._th.join(timeout=30)
            try: self.proc.stdin.close()
            except: pass
            stderr_data = None
            def _read_stderr():
                nonlocal stderr_data
                try: stderr_data = self.proc.stderr.read()
                except: pass
            stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
            stderr_thread.start()
            try: self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try: self.proc.wait(timeout=10)
                except: self.proc.kill()
            stderr_thread.join(timeout=5)
            if stderr_thread.is_alive():
                try: self.proc.stderr.close()
                except: pass
                stderr_thread.join(timeout=2)
            if stderr_data:
                log.warning(f"FFmpeg stderr: {stderr_data[:500]}")
            if self._error:
                raise RuntimeError(f"FFmpeg pipe write failed: {self._error}")
            if self.proc.returncode not in (0, None):
                raise RuntimeError(f"FFmpeg exited with code {self.proc.returncode}")
