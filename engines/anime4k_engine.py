import numpy as np
from . import gpu_kernels as gk
from backend.config import BLK

def _sf(val, default=0.0):
    """Safe float: handles None, empty string, non-numeric gracefully."""
    if val is None:
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)

class Anime4KEngineCUDA:
    def __init__(self, Hi, Wi, Ho, Wo, mb):
        self.Hi, self.Wi, self.Ho, self.Wo = Hi, Wi, Ho, Wo
        self.fpi, self.fpo = Hi * Wi, Ho * Wo
        self.mb = mb
        self.src = gk.cp.empty(mb * self.fpi * 3, dtype=gk.cp.float32)
        self.dst = gk.cp.empty(mb * self.fpo * 3, dtype=gk.cp.float32)
        self.lm = gk.cp.empty(mb * max(self.fpi, self.fpo), dtype=gk.cp.float32)
        self.gr = gk.cp.empty(mb * max(self.fpi, self.fpo), dtype=gk.cp.float32)
        self.ry = float(Hi) / float(Ho); self.rx = float(Wi) / float(Wo)

    def release(self):
        del self.src, self.dst, self.lm, self.gr
        if gk.mempool: gk.mempool.free_all_blocks()

    def _check(self, B):
        if B > self.mb: raise RuntimeError(f"A4K batch {B} > buffer {self.mb}")

    def apply_lines_only(self, frames_np, params):
        B = frames_np.shape[0]; self._check(B)
        npix = B * self.fpo; t = npix * 3; gp = (npix + BLK - 1) // BLK
        dst = self.dst[:t]; ucfg = params.get('upscale') or {}
        # Ensure numeric types for parameters (may come as strings from JSON)
        rs = _sf(ucfg.get('anime4kDeblur', 60)) / 100.0
        ds = _sf(ucfg.get('anime4kDarken', 100)) / 100.0
        ts = _sf(ucfg.get('anime4kThin', 0)) / 100.0
        dst_view = dst.reshape(B, self.Ho, self.Wo, 3)
        dst_view[:] = gk.cp.asarray(frames_np, dtype=gk.cp.float32) / 255.0
        gk._a4k_clamp((gp,),(BLK,),(dst,np.int32(self.Ho),np.int32(self.Wo),np.int32(npix)))
        gk._a4k_lg((gp,),(BLK,),(dst,self.lm,self.gr,np.int32(self.Ho),np.int32(self.Wo),np.int32(npix)))
        if rs>0: gk._a4k_restore((gp,),(BLK,),(dst,self.lm,self.gr,np.float32(rs),np.int32(self.Ho),np.int32(self.Wo),np.int32(npix)))
        gk._a4k_lg((gp,),(BLK,),(dst,self.lm,self.gr,np.int32(self.Ho),np.int32(self.Wo),np.int32(npix)))
        if ds>0: gk._a4k_darken((gp,),(BLK,),(dst,self.lm,self.gr,np.float32(ds),np.int32(self.Ho),np.int32(self.Wo),np.int32(npix)))
        if ts>0: gk._a4k_thin((gp,),(BLK,),(dst,self.lm,self.gr,np.float32(ts),np.int32(self.Ho),np.int32(self.Wo),np.int32(npix)))
        gk._a4k_clamp((gp,),(BLK,),(dst,np.int32(self.Ho),np.int32(self.Wo),np.int32(npix)))
        return gk.cp.asnumpy((gk.cp.clip(dst_view, 0.0, 1.0) * 255).astype(np.uint8))

    def process_upscale(self, frames_np, params):
        B = frames_np.shape[0]; self._check(B)
        np_i = B * self.fpi; np_o = B * self.fpo
        gp_i = (np_i+BLK-1)//BLK; gp_o = (np_o+BLK-1)//BLK
        ucfg = params.get('upscale') or {}
        # Ensure numeric types for parameters (may come as strings from JSON)
        rs = _sf(ucfg.get('anime4kDeblur', 60)) / 100.0
        ds = _sf(ucfg.get('anime4kDarken', 100)) / 100.0
        ts = _sf(ucfg.get('anime4kThin', 0)) / 100.0
        src = self.src[:np_i*3]; dst = self.dst[:np_o*3]
        src_v = src.reshape(B, self.Hi, self.Wi, 3)
        src_v[:] = gk.cp.asarray(frames_np, dtype=gk.cp.float32) / 255.0
        gk._a4k_clamp((gp_i,),(BLK,),(src,np.int32(self.Hi),np.int32(self.Wi),np.int32(np_i)))
        gk._a4k_lg((gp_i,),(BLK,),(src,self.lm,self.gr,np.int32(self.Hi),np.int32(self.Wi),np.int32(np_i)))
        if rs>0: gk._a4k_restore((gp_i,),(BLK,),(src,self.lm,self.gr,np.float32(rs),np.int32(self.Hi),np.int32(self.Wi),np.int32(np_i)))
        gk._a4k_clamp((gp_i,),(BLK,),(src,np.int32(self.Hi),np.int32(self.Wi),np.int32(np_i)))
        gk._a4k_lg((gp_i,),(BLK,),(src,self.lm,self.gr,np.int32(self.Hi),np.int32(self.Wi),np.int32(np_i)))
        gk._a4k_up((gp_o,),(BLK,),(src,dst,self.lm,self.gr,np.int32(self.Hi),np.int32(self.Wi),np.int32(self.Ho),np.int32(self.Wo),np.float32(self.ry),np.float32(self.rx),np.int32(np_o)))
        gk._a4k_lg((gp_o,),(BLK,),(dst,self.lm,self.gr,np.int32(self.Ho),np.int32(self.Wo),np.int32(np_o)))
        if ds>0: gk._a4k_darken((gp_o,),(BLK,),(dst,self.lm,self.gr,np.float32(ds),np.int32(self.Ho),np.int32(self.Wo),np.int32(np_o)))
        if ts>0: gk._a4k_thin((gp_o,),(BLK,),(dst,self.lm,self.gr,np.float32(ts),np.int32(self.Ho),np.int32(self.Wo),np.int32(np_o)))
        gk._a4k_clamp((gp_o,),(BLK,),(dst,np.int32(self.Ho),np.int32(self.Wo),np.int32(np_o)))
        dst_v = dst.reshape(B, self.Ho, self.Wo, 3)
        return gk.cp.asnumpy((gk.cp.clip(dst_v, 0.0, 1.0) * 255).astype(np.uint8))
