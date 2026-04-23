# ============================================
# ANIME UPSCALERR — Modular Backend Orchestrator
# ============================================
import gradio as gr
from fastapi.responses import FileResponse, JSONResponse
import threading, time, signal, os
from backend.config import auto_configure_vram, HAS_GPU, OUTPUT_DIR, PREVIEW_DIR, INPUT_DIR, TEMP_DIR
from backend.engines import gpu_kernels as gk
from backend.api import endpoints as api

# Ignore SIGINT (Ctrl+C) during processing to prevent accidental cancellation
# Only allow graceful shutdown on SIGTERM
def _ignore_sigint(signum, frame):
    pass  # Ignore Ctrl+C silently

def _register_fastapi_routes(app):
    @app.get("/download/{job_id}")
    def download_output(job_id: str, authToken: str):
        if authToken != api.SESSION_TOKEN:
            return JSONResponse({'error': 'invalid token', 'code': 'UNAUTHORIZED'}, status_code=401)

        j = api.job_queue.get(job_id)
        if not j:
            return JSONResponse({'error': 'job not found', 'code': 'JOB_NOT_FOUND'}, status_code=404)
        if not j.output_path:
            return JSONResponse({'error': 'output not ready', 'code': 'OUTPUT_NOT_READY'}, status_code=409)

        output_path = os.path.abspath(str(j.output_path))
        if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
            return JSONResponse({'error': 'output file missing', 'code': 'OUTPUT_NOT_FOUND'}, status_code=404)

        return FileResponse(path=output_path, filename=os.path.basename(output_path), media_type='application/octet-stream')

def start_backend():
    # Install SIGINT handler to ignore Ctrl+C during processing
    signal.signal(signal.SIGINT, _ignore_sigint)
    print(" Initializing Anime Upscalerr Backend...")
    auto_configure_vram()
    
    if HAS_GPU:
        gk.init_gpu(vram_pct=85)
        # Install AI upscale engines (downloads models on first run)
        try:
            from backend.engines.realesrgan_engine import install_realesrgan
            install_realesrgan()
        except Exception as e:
            print(f"  Real-ESRGAN setup warning: {e}")
        try:
            from backend.engines.waifu2x_engine import install_waifu2x
            install_waifu2x()
        except Exception as e:
            print(f"  Waifu2x setup warning: {e}")
    
    with gr.Blocks(title='Anime Upscalerr — Neo-Manga Edition') as demo:
        gr.Markdown('# 🚀 Anime Upscalerr — Modular Backend')
        gr.Markdown(f'**Session Token:** `{api.SESSION_TOKEN}`')
        
        # API Endpoints for Frontend
        gr.Button(visible=False).click(fn=api.api_session_init, inputs=[gr.JSON(value={}, visible=False)], outputs=[gr.JSON()], api_name='session_init')
        gr.Button(visible=False).click(fn=api.api_get_system_stats, inputs=[gr.JSON(value={}, visible=False)], outputs=[gr.JSON()], api_name='get_system_stats')
        gr.Button(visible=False).click(fn=api.api_list_jobs, inputs=[gr.JSON(value={}, visible=False)], outputs=[gr.JSON()], api_name='list_jobs')
        gr.Button(visible=False).click(fn=api.api_create_job, inputs=[gr.File(visible=False), gr.Textbox(visible=False)], outputs=[gr.JSON()], api_name='create_job')
        gr.Button(visible=False).click(fn=api.api_get_job, inputs=[gr.Textbox(visible=False)], outputs=[gr.JSON()], api_name='get_job')
        gr.Button(visible=False).click(fn=api.api_cancel_job, inputs=[gr.Textbox(visible=False)], outputs=[gr.JSON()], api_name='cancel_job')
        gr.Button(visible=False).click(fn=api.api_delete_job, inputs=[gr.Textbox(visible=False)], outputs=[gr.JSON()], api_name='delete_job')
        gr.Button(visible=False).click(fn=api.api_cancel_all, inputs=[gr.JSON(value={}, visible=False)], outputs=[gr.JSON()], api_name='cancel_all')
        gr.Button(visible=False).click(fn=api.api_extract_frame, inputs=[gr.Textbox(visible=False)], outputs=[gr.JSON()], api_name='extract_frame')
        gr.Button(visible=False).click(fn=api.api_render_job, inputs=[gr.Textbox(visible=False)], outputs=[gr.JSON()], api_name='render_job')
        gr.Button(visible=False).click(fn=api.api_render_all, inputs=[gr.JSON(value={}, visible=False)], outputs=[gr.JSON()], api_name='render_all')
        gr.Button(visible=False).click(fn=api.api_list_presets, inputs=[gr.JSON(value={}, visible=False)], outputs=[gr.JSON()], api_name='list_presets')
        gr.Button(visible=False).click(fn=api.api_save_preset, inputs=[gr.Textbox(visible=False)], outputs=[gr.JSON()], api_name='save_preset')
        gr.Button(visible=False).click(fn=api.api_delete_preset, inputs=[gr.Textbox(visible=False)], outputs=[gr.JSON()], api_name='delete_preset')
        gr.Button(visible=False).click(fn=api.api_get_output, inputs=[gr.Textbox(visible=False)], outputs=[gr.JSON()], api_name='get_output')

    print(f'🔑 Token: {api.SESSION_TOKEN}')

    # Launch and get the server info
    demo.launch(
        share=True,
        server_port=7860,
        server_name='0.0.0.0',
        quiet=False,
        prevent_thread_lock=True,
        allowed_paths=[
            '/content/AnimeUpscaler',
            str(OUTPUT_DIR),
            str(PREVIEW_DIR),
            str(INPUT_DIR),
            str(TEMP_DIR),
        ],
    )
    if getattr(demo, "app", None) is not None:
        _register_fastapi_routes(demo.app)

    # Print prominent connection instructions after server starts
    print('\n' + '='*70)
    print('🎌 ANIME UPSCALERR - BACKEND READY')
    print('='*70)
    print('\n📌 COPY THIS URL AND PASTE IT IN THE WEBSITE:')
    share_url = getattr(demo, 'share_url', None) or 'https://<YOUR-GRADIO-SHARE-URL>.gradio.live'
    link_box_width = max(72, len(share_url) + 8)
    print('┌' + ('─' * link_box_width) + '┐')
    print('│' + ' ' * link_box_width + '│')
    print('│' + ' GRADIO PUBLIC URL '.center(link_box_width) + '│')
    print('│' + ' ' * link_box_width + '│')
    print('│' + share_url.center(link_box_width) + '│')
    print('│' + ' ' * link_box_width + '│')
    print('└' + ('─' * link_box_width) + '┘')
    print('\nURL only:')
    print(share_url)
    print('\n📋 INSTRUCTIONS:')
    print('   1. Open: https://anime-upscalerr.vercel.app/app.html')
    print('   2. Paste the URL above in "NODE TUNNEL ADDRESS"')
    print('   3. Click "ESTABLISH LINK"')
    print('   4. Upload your anime video and start upscaling!')
    print('='*70 + '\n')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
if __name__ == "__main__":
    start_backend()
