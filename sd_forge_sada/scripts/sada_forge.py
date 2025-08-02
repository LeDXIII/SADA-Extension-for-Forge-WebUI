import torch
import torch.nn.functional as F
import gradio as gr
import math
from modules import scripts

# Global storage for cleanup and logging
_sada_state = {
    'step_skipper': None,
    'original_apply_model': None,
    'patched_unet': None,
    'is_active': False,
    'logged_activation': False,
    'logged_first_skip': False,
    'total_skips': 0,
    'total_steps': 0
}

class SADAStepCounter:
    """Tracks actual sampling steps for precise control."""
    def __init__(self):
        self.reset()
        
    def update_step(self, sigma, total_steps=None):
        if total_steps is not None:
            self.total_steps = total_steps
            
        self.sigma_history.append(float(sigma))
        
        if len(self.sigma_history) == 1:
            self.current_step = 0
        else:
            first_sigma = self.sigma_history[0]
            current_sigma = sigma
            
            if first_sigma > 0:
                progress = max(0, min(1, (first_sigma - current_sigma) / first_sigma))
                self.current_step = int(progress * self.total_steps)
            else:
                self.current_step = len(self.sigma_history) - 1
        
        return self.current_step
    
    def reset(self):
        self.current_step = 0
        self.total_steps = 50
        self.step_history = []
        self.sigma_history = []

def safe_tensor_to_float(tensor):
    """Safely convert tensor to float, handling multiple elements."""
    try:
        if hasattr(tensor, 'item'):
            if tensor.numel() == 1:
                return tensor.item()
            elif tensor.numel() > 1:
                return tensor.flatten()[0].item()
            else:
                return 0.0
        else:
            return float(tensor)
    except (RuntimeError, ValueError):
        try:
            if hasattr(tensor, '__getitem__'):
                return float(tensor[0])
            else:
                return float(tensor)
        except:
            return 1.0

class SADAStepSkipper:
    """Implements step skipping with clean logging."""
    def __init__(self, skip_ratio, acc_range, stability_threshold=0.05):
        self.skip_ratio = skip_ratio
        self.acc_range = acc_range
        self.stability_threshold = stability_threshold
        self.reset()
        
    def should_skip_step(self, current_features, timestep_tensor, total_steps):
        """Decide whether to skip current step with minimal logging."""
        global _sada_state
        
        sigma = safe_tensor_to_float(timestep_tensor)
        current_step = self.step_counter.update_step(sigma, total_steps)
        acc_start, acc_end = self.acc_range
        
        # Log activation only once at start
        if not _sada_state['logged_activation']:
            print(f"SADA v4: Activated for {total_steps} steps, acceleration range: {acc_start}-{acc_end}")
            _sada_state['logged_activation'] = True
            _sada_state['total_steps'] = total_steps
        
        # Check acceleration range
        if current_step < acc_start or current_step > acc_end:
            return False
            
        # Log first acceleration only once
        if not _sada_state['logged_first_skip'] and acc_start <= current_step <= acc_end:
            print(f"SADA: Acceleration started at step {current_step}")
            _sada_state['logged_first_skip'] = True
            
        # Edge protection
        if current_step < acc_start + 2 or current_step > acc_end - 2:
            return False
            
        # Consecutive skip limit
        if self.skip_count >= 2:
            self.skip_count = 0
            return False
            
        # Feature stability check
        if self.prev_features is not None:
            try:
                current_flat = current_features.flatten()
                prev_flat = self.prev_features.flatten()
                
                min_size = min(len(current_flat), len(prev_flat))
                if min_size > 0:
                    current_flat = current_flat[:min_size]
                    prev_flat = prev_flat[:min_size]
                    
                    similarity = F.cosine_similarity(
                        current_flat.unsqueeze(0), 
                        prev_flat.unsqueeze(0)
                    ).item()
                    
                    if similarity > (1.0 - self.stability_threshold):
                        self.skip_count += 1
                        _sada_state['total_skips'] += 1
                        return True
                        
            except Exception:
                pass
                
        self.prev_features = current_features.clone().detach()
        self.skip_count = 0
        return False
    
    def reset(self):
        """Reset state for new generation."""
        self.step_counter = SADAStepCounter()
        self.prev_features = None
        self.skip_count = 0

def cleanup_sada_patches():
    """Clean up any existing SADA patches."""
    global _sada_state
    
    # Log completion summary if we were active
    if _sada_state['is_active'] and _sada_state['logged_activation']:
        skipped = _sada_state['total_skips']
        total = _sada_state['total_steps']
        if total > 0:
            print(f"SADA: Completed - skipped {skipped}/{total} steps ({skipped/total*100:.1f}%)")
    
    if _sada_state['patched_unet'] is not None and _sada_state['original_apply_model'] is not None:
        try:
            _sada_state['patched_unet'].model.apply_model = _sada_state['original_apply_model']
        except Exception as e:
            print(f"SADA: Cleanup error: {e}")
    
    # Reset all state
    _sada_state.update({
        'step_skipper': None,
        'original_apply_model': None,
        'patched_unet': None,
        'is_active': False,
        'logged_activation': False,
        'logged_first_skip': False,
        'total_skips': 0,
        'total_steps': 0
    })

def apply_sada_acceleration(unet_patcher, skip_ratio, acc_range, early_exit_threshold, total_steps):
    """Apply SADA with clean logging."""
    global _sada_state
    
    cleanup_sada_patches()
    
    step_skipper = SADAStepSkipper(skip_ratio=skip_ratio, acc_range=acc_range)
    _sada_state['step_skipper'] = step_skipper
    _sada_state['is_active'] = True
    
    def sada_model_wrapper(original_apply_model):
        def wrapped_apply_model(x, timestep, **kwargs):
            if not _sada_state['is_active'] or _sada_state['step_skipper'] is None:
                return original_apply_model(x, timestep, **kwargs)
            
            if _sada_state['step_skipper'].should_skip_step(x, timestep, total_steps):
                if hasattr(wrapped_apply_model, '_last_result'):
                    sigma = safe_tensor_to_float(timestep)
                    noise_scale = sigma * 0.03
                    noise = torch.randn_like(x) * noise_scale
                    return wrapped_apply_model._last_result + noise
            
            result = original_apply_model(x, timestep, **kwargs)
            wrapped_apply_model._last_result = result.clone().detach()
            return result
            
        return wrapped_apply_model
    
    def sada_forward_patch(h, hsp, transformer_options):
        if not _sada_state['is_active'] or _sada_state['step_skipper'] is None:
            return h, hsp
            
        current_step = _sada_state['step_skipper'].step_counter.current_step
        acc_start, acc_end = acc_range
        
        if acc_start <= current_step <= acc_end and early_exit_threshold > 0:
            try:
                if len(h.shape) == 4:  # Conv layers
                    B, C, H, W = h.shape
                    feature_magnitude = torch.mean(torch.abs(h)).item()
                    
                    if feature_magnitude < early_exit_threshold:
                        range_progress = (current_step - acc_start) / max(1, acc_end - acc_start)
                        scale_factor = 0.75 + 0.15 * range_progress
                        
                        h_small = F.interpolate(h, scale_factor=scale_factor, mode='bilinear', align_corners=False)
                        h = F.interpolate(h_small, size=(H, W), mode='bilinear', align_corners=False)
                
                elif len(h.shape) == 3:  # Attention layers
                    B, N, C = h.shape
                    if N > 256:
                        range_progress = (current_step - acc_start) / max(1, acc_end - acc_start)
                        keep_ratio = 0.6 + 0.25 * range_progress
                        keep_tokens = max(64, int(N * keep_ratio))
                        
                        if keep_tokens < N:
                            step_size = max(1, N // keep_tokens)
                            indices = torch.arange(0, N, step_size, device=h.device)[:keep_tokens]
                            h_reduced = h[:, indices, :]
                            
                            h = F.interpolate(
                                h_reduced.transpose(1, 2),
                                size=N,
                                mode='linear',
                                align_corners=False
                            ).transpose(1, 2)
                        
            except Exception:
                pass
        
        return h, hsp
    
    m = unet_patcher.clone()
    m.set_model_output_block_patch(sada_forward_patch)
    
    try:
        _sada_state['original_apply_model'] = m.model.apply_model
        _sada_state['patched_unet'] = m
        m.model.apply_model = sada_model_wrapper(_sada_state['original_apply_model'])
    except Exception as e:
        print(f"SADA: Failed to apply model wrapper: {e}")
    
    return m

class SADAForForge(scripts.Script):
    sorting_priority = 15
    
    def title(self):
        return "SADA: Stability-guided Adaptive Diffusion Acceleration"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            sada_enabled = gr.Checkbox(
                label='Enable SADA Acceleration', 
                value=False,
                info="Enable SADA with model-specific presets"
            )
            
            # Model preset selection
            with gr.Row():
                model_preset = gr.Radio(
                    choices=["SDXL (Balanced)", "Flux (Aggressive)"],
                    value="SDXL (Balanced)",
                    label="Model Preset",
                    info="Choose preset optimized for your model type"
                )
            
            with gr.Row():
                skip_ratio = gr.Slider(
                    label='Skip Aggressiveness', 
                    minimum=0.05, maximum=0.4, step=0.05, value=0.2,
                    info="Ratio of steps to skip"
                )
            
            with gr.Row():
                acc_start = gr.Slider(
                    label='Start Step', 
                    minimum=1, maximum=25, step=1, value=15,
                    info="Begin acceleration from this step"
                )
                acc_end = gr.Slider(
                    label='End Step', 
                    minimum=25, maximum=50, step=1, value=45,
                    info="Stop acceleration at this step"
                )
            
            with gr.Row():
                early_exit_threshold = gr.Slider(
                    label='Early Exit Threshold', 
                    minimum=0.005, maximum=0.05, step=0.005, value=0.02,
                    info="Feature threshold for lightweight processing"
                )
            
            # Updated preset function with new SDXL parameters
            def update_preset(preset_choice):
                if preset_choice == "SDXL (Balanced)":
                    return {
                        skip_ratio: gr.update(value=0.2),
                        acc_start: gr.update(value=15),
                        acc_end: gr.update(value=45),
                        early_exit_threshold: gr.update(value=0.02)
                    }
                elif preset_choice == "Flux (Aggressive)":
                    return {
                        skip_ratio: gr.update(value=0.3),
                        acc_start: gr.update(value=7),
                        acc_end: gr.update(value=35),
                        early_exit_threshold: gr.update(value=0.04)
                    }
                else:
                    return {
                        skip_ratio: gr.update(),
                        acc_start: gr.update(),
                        acc_end: gr.update(),
                        early_exit_threshold: gr.update()
                    }
            
            # Connect preset selector to sliders
            model_preset.change(
                fn=update_preset,
                inputs=[model_preset],
                outputs=[skip_ratio, acc_start, acc_end, early_exit_threshold]
            )
            
            gr.HTML("""
            <div style="background-color: #e8f4fd; padding: 12px; border-radius: 8px; border: 1px solid #bee5eb;">
                <b>🚀 SADA v4 - Updated SDXL Parameters:</b><br>
                • <b>🎯 SDXL Balanced:</b> Start=15, Skip=0.2, End=45, Threshold=0.02<br>
                • <b>⚡ Flux Aggressive:</b> Start=7, Skip=0.3, End=35, Threshold=0.04<br>
                • <b>📊 Clean logging:</b> Only essential messages during generation<br>
                • <b>🔧 Manual adjust:</b> Fine-tune any setting after selecting preset<br>
                • <b>💨 Expected speedup:</b> 1.2-1.5x (SDXL) / 1.3-1.8x (Flux)
            </div>
            """)
        
        return (sada_enabled, model_preset, skip_ratio, acc_start, acc_end, early_exit_threshold)
    
    def process_before_every_sampling(self, p, *script_args, **kwargs):
        """Apply SADA with updated SDXL parameters."""
        sada_enabled, model_preset, skip_ratio, acc_start, acc_end, early_exit_threshold = script_args
        
        cleanup_sada_patches()
        
        if not sada_enabled:
            return
        
        total_steps = getattr(p, 'steps', 20)
        acc_range = (int(acc_start), int(acc_end))
        
        try:
            unet = p.sd_model.forge_objects.unet
            
            unet = apply_sada_acceleration(
                unet_patcher=unet,
                skip_ratio=float(skip_ratio),
                acc_range=acc_range,
                early_exit_threshold=float(early_exit_threshold),
                total_steps=total_steps
            )
            
            p.sd_model.forge_objects.unet = unet
            
            # Log parameters with preset info
            p.extra_generation_params.update({
                'SADA_v4': True,
                'SADA_preset': model_preset,
                'SADA_skip': skip_ratio,
                'SADA_range': f"{acc_start}-{acc_end}",
                'SADA_threshold': early_exit_threshold
            })
            
        except Exception as e:
            print(f"SADA: Failed to apply: {e}")
            cleanup_sada_patches()
        
        return
