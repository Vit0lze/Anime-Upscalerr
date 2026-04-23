# ============================================
# CUDA KERNELS (CuPy)
# ============================================
GPU_OK = False
cp = None
mempool = None
pinned_pool = None

_color_grade = _bgr2hsv = _hsv2bgr = _hsl = _vibsat = _bilateral_denoise = None
_a4k_lg = _a4k_clamp = _a4k_restore = _a4k_darken = _a4k_thin = _a4k_up = None

def init_gpu(vram_pct=85):
    global GPU_OK, cp, mempool, pinned_pool
    global _color_grade, _bgr2hsv, _hsv2bgr, _hsl, _vibsat, _bilateral_denoise
    global _a4k_lg, _a4k_clamp, _a4k_restore, _a4k_darken, _a4k_thin, _a4k_up
    try:
        import cupy as _cp; cp = _cp
        mempool = cp.get_default_memory_pool(); pinned_pool = cp.get_default_pinned_memory_pool()
        mempool.set_limit(fraction=vram_pct / 100.0)
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
        if isinstance(gpu_name, bytes): gpu_name = gpu_name.decode()
        GPU_OK = True; print(f"✅ GPU Pipeline: {gpu_name} | VRAM limit: {vram_pct}%")
    except Exception as e: print(f"⚠️ GPU Pipeline disabled: {e}"); return False

    _color_grade = cp.RawKernel(r'''extern "C" __global__
    void color_grade(float*d,int np,float ev,float cv){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=np)return;int i=p*3;
    float b=d[i],g=d[i+1],r=d[i+2],e=1.f+ev;b*=e;g*=e;r*=e;
    float cf=1.f+cv*.01f;b=.5f+(b-.5f)*cf;g=.5f+(g-.5f)*cf;r=.5f+(r-.5f)*cf;
    d[i]=fmaxf(0.f,fminf(1.f,b));d[i+1]=fmaxf(0.f,fminf(1.f,g));d[i+2]=fmaxf(0.f,fminf(1.f,r));}''','color_grade')

    _bgr2hsv = cp.RawKernel(r'''extern "C" __global__
    void bgr2hsv(const float*s,float*h,int n){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=n)return;int i=p*3;
    float b=s[i],g=s[i+1],r=s[i+2],mx=fmaxf(fmaxf(r,g),b),mn=fminf(fminf(r,g),b),d=mx-mn;
    float V=mx*255.f,S=(mx>1e-6f)?(d/mx)*255.f:0.f,H=0.f;
    if(d>1e-8f){if(mx==r)H=30.f*fmodf(((g-b)/d)+6.f,6.f);else if(mx==g)H=30.f*(((b-r)/d)+2.f);else H=30.f*(((r-g)/d)+4.f);}
    if(H<0.f)H+=180.f;if(H>=180.f)H-=180.f;h[i]=H;h[i+1]=S;h[i+2]=V;}''','bgr2hsv')

    _hsv2bgr = cp.RawKernel(r'''extern "C" __global__
    void hsv2bgr(const float*h,float*s,int n){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=n)return;int i=p*3;
    float H=h[i],S=h[i+1]/255.f,V=h[i+2]/255.f,C=V*S,H2=H/30.f,X=C*(1.f-fabsf(fmodf(H2,2.f)-1.f)),m=V-C,r=m,g=m,b=m;
    if(H2<1.f){r+=C;g+=X;}else if(H2<2.f){r+=X;g+=C;}else if(H2<3.f){g+=C;b+=X;}else if(H2<4.f){g+=X;b+=C;}else if(H2<5.f){r+=X;b+=C;}else{r+=C;b+=X;}
    s[i]=fmaxf(0.f,fminf(1.f,b));s[i+1]=fmaxf(0.f,fminf(1.f,g));s[i+2]=fmaxf(0.f,fminf(1.f,r));}''','hsv2bgr')

    _hsl = cp.RawKernel(r'''extern "C" __global__
    void hsl(float*v,int n,float gh,float gs,float gl,float bh,float bs,float bl,float sh,float ss,float sl){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=n)return;int i=p*3;float H=v[i],S=v[i+1],V=v[i+2];
    if(H>=25.f&&H<=95.f){float m=fmaxf(0.f,fminf(fminf(H-30.f,90.f-H)/10.f,1.f));if(m>0.f){H+=gh*m;S=fmaxf(0.f,S+gs*m*2.f);V=fmaxf(0.f,V+gl*m*2.f);}}
    if(H>=80.f&&H<=140.f){float m=fmaxf(0.f,fminf(fminf(H-85.f,135.f-H)/10.f,1.f));if(m>0.f){H+=bh*m;S=fmaxf(0.f,S+bs*m*2.f);V=fmaxf(0.f,V+bl*m*2.f);}}
    if(H>=1.f&&H<=30.f){float m=fmaxf(0.f,fminf(fminf(H-2.f,28.f-H)/5.f,1.f));if(m>0.f){H+=sh*m;S=fmaxf(0.f,S+ss*m*2.f);V=fmaxf(0.f,V+sl*m*2.f);}}
    v[i]=fmaxf(0.f,fminf(180.f,H));v[i+1]=fmaxf(0.f,fminf(255.f,S));v[i+2]=fmaxf(0.f,fminf(255.f,V));}''','hsl')

    _vibsat = cp.RawKernel(r'''extern "C" __global__
    void vibsat(float*d,int n,float sv){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=n)return;int i=p*3;
    float b=d[i],g=d[i+1],r=d[i+2],gr=.299f*r+.587f*g+.114f*b;
    if(sv!=0.f){float f=1.f+sv*.01f;b=gr+f*(b-gr);g=gr+f*(g-gr);r=gr+f*(r-gr);}
    d[i]=fmaxf(0.f,fminf(1.f,b));d[i+1]=fmaxf(0.f,fminf(1.f,g));d[i+2]=fmaxf(0.f,fminf(1.f,r));}''','vibsat')

    _bilateral_denoise = cp.RawKernel(r'''extern "C" __global__
    void bilateral_denoise(const float*src,float*dst,int H,int W,int N,
                           float sigma_s,float sigma_r,int radius){
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    int total=N*H*W;
    if(p>=total)return;
    int batch=p/(H*W),lp=p%(H*W),y=lp/W,x=lp%W;
    int base=batch*H*W*3,ci=base+lp*3;
    float cb=src[ci],cg=src[ci+1],cr=src[ci+2];
    float sb=0.f,sg=0.f,sr=0.f,ws=0.f;
    float inv2ss=-0.5f/(sigma_s*sigma_s+1e-6f);
    float inv2sr=-0.5f/(sigma_r*sigma_r+1e-6f);
    for(int dy=-radius;dy<=radius;dy++){
        for(int dx=-radius;dx<=radius;dx++){
            int ny=max(0,min(H-1,y+dy)),nx=max(0,min(W-1,x+dx));
            int ni=base+(ny*W+nx)*3;
            float nb=src[ni],ng=src[ni+1],nr=src[ni+2];
            float ds=(float)(dx*dx+dy*dy);
            float db=nb-cb,dg=ng-cg,dr=nr-cr;
            float dc=db*db+dg*dg+dr*dr;
            float w=expf(ds*inv2ss)*expf(dc*inv2sr);
            sb+=nb*w;sg+=ng*w;sr+=nr*w;ws+=w;}}
    float iv=1.f/fmaxf(ws,1e-6f);
    dst[ci]=sb*iv;dst[ci+1]=sg*iv;dst[ci+2]=sr*iv;}''','bilateral_denoise')

    _a4k_lg = cp.RawKernel(r'''extern "C" __global__
    void a4k_lg(const float*bgr,float*luma,float*grad,int H,int W,int tp){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=tp)return;
    int b=p/(H*W),lp=p%(H*W),y=lp/W,x=lp%W,fo=b*H*W*3,lo=b*H*W,bi=fo+lp*3;
    float L=.299f*bgr[bi+2]+.587f*bgr[bi+1]+.114f*bgr[bi];luma[lo+lp]=L;
    float gx=0.f,gy=0.f;
    for(int dy=-1;dy<=1;dy++){for(int dx=-1;dx<=1;dx++){
    int ny=max(0,min(H-1,y+dy)),nx=max(0,min(W-1,x+dx)),nb=fo+(ny*W+nx)*3;
    float nL=.299f*bgr[nb+2]+.587f*bgr[nb+1]+.114f*bgr[nb],wx=(dx==-1)?-1.f:(dx==1)?1.f:0.f,wy=(dy==-1)?-1.f:(dy==1)?1.f:0.f;
    if(dy==0)wx*=2.f;if(dx==0)wy*=2.f;gx+=nL*wx;gy+=nL*wy;}}
    grad[lo+lp]=sqrtf(gx*gx+gy*gy);}''','a4k_lg')

    _a4k_clamp = cp.RawKernel(r'''extern "C" __global__
    void a4k_clamp(float*d,int H,int W,int tp){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=tp)return;
    int b=p/(H*W),lp=p%(H*W),y=lp/W,x=lp%W,fo=b*H*W*3,bi=fo+lp*3;
    float xb=-1e9f,xg=-1e9f,xr=-1e9f,nb_=1e9f,ng_=1e9f,nr_=1e9f;
    for(int dy=-1;dy<=1;dy++){for(int dx=-1;dx<=1;dx++){
    if(!dx&&!dy)continue;
    int ny=max(0,min(H-1,y+dy)),nx=max(0,min(W-1,x+dx)),ni=fo+(ny*W+nx)*3;
    xb=fmaxf(xb,d[ni]);xg=fmaxf(xg,d[ni+1]);xr=fmaxf(xr,d[ni+2]);
    nb_=fminf(nb_,d[ni]);ng_=fminf(ng_,d[ni+1]);nr_=fminf(nr_,d[ni+2]);}}
    d[bi]=fmaxf(nb_,fminf(xb,d[bi]));d[bi+1]=fmaxf(ng_,fminf(xg,d[bi+1]));d[bi+2]=fmaxf(nr_,fminf(xr,d[bi+2]));}''','a4k_clamp')

    _a4k_restore = cp.RawKernel(r'''extern "C" __global__
    void a4k_restore(float*d,const float*lm,const float*gr,float str,int H,int W,int tp){
    int b=blockDim.x*blockIdx.x+threadIdx.x;if(b>=tp)return;
    int ba=b/(H*W),lp=b%(H*W),y=lp/W,x=lp%W,fo=ba*H*W*3,lo=ba*H*W,bi=fo+lp*3;
    float cg=gr[lo+lp],cl=lm[lo+lp],sb=0,sg=0,sr=0,ws=0;
    for(int dy=-1;dy<=1;dy++){for(int dx=-1;dx<=1;dx++){
    int ny=max(0,min(H-1,y+dy)),nx=max(0,min(W-1,x+dx)),ni=ny*W+nx,nb=fo+ni*3;
    float nL=lm[lo+ni],ng=gr[lo+ni],ld=fabsf(nL-cl),gs=1.f/(1.f+fabsf(ng-cg)*10.f),sp=(!dx&&!dy)?2.f:1.f,w;
    if(cg<.1f)w=sp*expf(-ld*ld*50.f);else w=sp*gs*expf(-ld*ld*200.f);
    sb+=d[nb]*w;sg+=d[nb+1]*w;sr+=d[nb+2]*w;ws+=w;}}
    if(ws>.001f){float iv=1.f/ws;d[bi]+=str*(sb*iv-d[bi]);d[bi+1]+=str*(sg*iv-d[bi+1]);d[bi+2]+=str*(sr*iv-d[bi+2]);}}''','a4k_restore')

    _a4k_darken = cp.RawKernel(r'''extern "C" __global__
    void a4k_darken(float*d,const float*lm,const float*gr,float str,int H,int W,int tp){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=tp)return;
    int b=p/(H*W),lp=p%(H*W),y=lp/W,x=lp%W,lo=b*H*W,bi=b*H*W*3+lp*3;
    float cl=lm[lo+lp],cg=gr[lo+lp];if(cg<.03f)return;float mn=1e9f;
    for(int dy=-1;dy<=1;dy++){for(int dx=-1;dx<=1;dx++){
    mn=fminf(mn,lm[lo+max(0,min(H-1,y+dy))*W+max(0,min(W-1,x+dx))]);}}
    float df=cl-mn;if(df>.001f){float f=fmaxf(.6f,1.f-str*fminf(df*4.f,.25f)*fminf(cg*3.f,1.f));d[bi]*=f;d[bi+1]*=f;d[bi+2]*=f;}}''','a4k_darken')

    _a4k_thin = cp.RawKernel(r'''extern "C" __global__
    void a4k_thin(float*d,const float*lm,const float*gr,float str,int H,int W,int tp){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=tp)return;
    int b=p/(H*W),lp=p%(H*W),y=lp/W,x=lp%W,fo=b*H*W*3,lo=b*H*W,bi=fo+lp*3;
    float cl=lm[lo+lp],cg=gr[lo+lp];if(cg<.03f)return;float ml=-1.f;int mi=lp;
    for(int dy=-1;dy<=1;dy++){for(int dx=-1;dx<=1;dx++){
    if(!dx&&!dy)continue;
    int ni=max(0,min(H-1,y+dy))*W+max(0,min(W-1,x+dx));float nL=lm[lo+ni];
    if(nL>ml){ml=nL;mi=ni;}}}
    float df=ml-cl;if(df>.01f){float t=str*fminf(df*3.f,.35f)*fminf(cg*2.f,1.f);int nb=fo+mi*3;
    d[bi]+=t*(d[nb]-d[bi]);d[bi+1]+=t*(d[nb+1]-d[bi+1]);d[bi+2]+=t*(d[nb+2]-d[bi+2]);}}''','a4k_thin')

    _a4k_up = cp.RawKernel(r'''extern "C" __global__
    void a4k_up(const float*s,float*d,const float*lm,const float*gr,int sH,int sW,int dH,int dW,float rY,float rX,int tdp){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=tdp)return;
    int b=p/(dH*dW),lp=p%(dH*dW),dy_=lp/dW,dx_=lp%dW;
    float sy=(float)dy_*rY,sx=(float)dx_*rX;int iy=(int)floorf(sy),ix=(int)floorf(sx);float fy=sy-iy,fx=sx-ix;
    int so=b*sH*sW,sfo=b*sH*sW*3,di=(b*dH*dW+lp)*3;
    float cg=gr[so+max(0,min(sH-1,iy))*sW+max(0,min(sW-1,ix))];
    float sb=0,sg=0,sr=0,ws=0;
    for(int ky=-1;ky<=2;ky++){for(int kx=-1;kx<=2;kx++){
    int ny=max(0,min(sH-1,iy+ky)),nx=max(0,min(sW-1,ix+kx)),nb=sfo+(ny*sW+nx)*3;
    float ay=fabsf((float)ky-fy),ax=fabsf((float)kx-fx),wy,wx;
    if(ay<1.f)wy=1.1667f*ay*ay*ay-2.f*ay*ay+.8889f;else if(ay<2.f)wy=-.3889f*ay*ay*ay+2.f*ay*ay-3.3333f*ay+1.7778f;else wy=0;
    if(ax<1.f)wx=1.1667f*ax*ax*ax-2.f*ax*ax+.8889f;else if(ax<2.f)wx=-.3889f*ax*ax*ax+2.f*ax*ax-3.3333f*ax+1.7778f;else wx=0;
    float w=wy*wx;if(cg>.08f){w*=(.5f+.5f*expf(-fabsf(gr[so+ny*sW+nx]-cg)*5.f));}
    w=fmaxf(w,0.f);sb+=s[nb]*w;sg+=s[nb+1]*w;sr+=s[nb+2]*w;ws+=w;}}
    if(ws>.001f){float iv=1.f/ws;d[di]=fmaxf(0.f,fminf(1.f,sb*iv));d[di+1]=fmaxf(0.f,fminf(1.f,sg*iv));d[di+2]=fmaxf(0.f,fminf(1.f,sr*iv));}
    else{int nb=sfo+(max(0,min(sH-1,iy))*sW+max(0,min(sW-1,ix)))*3;d[di]=s[nb];d[di+1]=s[nb+1];d[di+2]=s[nb+2];}}''','a4k_up')

    return True
