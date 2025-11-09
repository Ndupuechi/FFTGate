












# %%  

#####---------------------- NOTE NOVEL ACTIVATION FUCNTION NOTE -----------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
#################################### FFTGATE  ############################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-----------------------XXX ACTIVATION FUCNTION   XXX------------------------------------------------------------#####



########################################################################################################################
####-------| NOTE 1. IMPORTS LIBRARIES | XXX ------------------------------------------------------#####################
########################################################################################################################

# ✅ Import libraries
import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

########################################################################################################################
####-------| NOTE 1. DEFINE FFTGATE ACTIVATION CLASS | XXX -----------------------------------------####################
########################################################################################################################


num_epochs = 100



# ===============================================================
# 🔍 Helper: ---- Normalizer for FFT magnitudes (layer-based) --
# ===============================================================
class FreqMagNorm(nn.Module):
    def __init__(self, eps=1e-6, target_min=0.01, target_max=3.0):
        super().__init__()
        self.eps = eps
        self.target_min = target_min
        self.target_max = target_max

    def forward(self, x):
        # ❌ Channel-wise: worked on vector [C] with per-channel normalization
        # ✅ Layer-based: collapse across channels → single scalar mean
        mean_val = x.mean()   # scalar

        # ✅ For a scalar there is no min/max scaling.
        # 🔖 Just bound it into [target_min, target_max]
        return torch.clamp(mean_val, self.target_min, self.target_max)
















# ===============================================================
# 🔗==================== FFTGATE activation 🔑================🔗
# ===============================================================

class FFTGate(nn.Module):


    # ===============================================================
    # 🔧 Initialization
    # ===============================================================
    def __init__(self,
                 gamma1_init=1.5,        # ⬅️ reduce from 1.5 → 0.8
                 freq_init=0.3,          # ⬅️ reduce from 0.3 → 0.1                  
                 history_len=32,         # ✅ default: 32 
                 enable_history=True,

                 use_skip=False,         # ✅ default: use_skip=False
                 decay_mode="linear",    # ✅ toggle options: "linear"(default), "exp", "off"  
                 use_freq_gate=True,     # 🟡 toggle to enable/disable FFT-based modulation
                 use_smoothing=True,     # 🟡 toggle to enable/disable James–Stein smoothing   

                 layer_depth=None,       # ✅ accept "layer_depth" as argument
                 total_layers=6          # 👈 safe default        
                 ):        
        super().__init__()         

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ store 
        self.activation_history = None  # always exists, starts empty
        self.history_len = int(history_len)
        self.enable_history = enable_history
        self.use_skip = use_skip
        self.decay_mode = decay_mode
        self.use_freq_gate = use_freq_gate
        self.use_smoothing = use_smoothing   


        # ⚡ store layer depth for scaling
        if layer_depth is None:
            # ✅ Default to placeholder (will be updated later by find_activations_split)
            self.layer_depth = 1
            print("⚠️ FFTGate: layer_depth temporarily set to 1 (will be updated later)")
        else:
            self.layer_depth = layer_depth

        self.total_layers = total_layers








        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔧 trainable raw params 
        self.gamma1_raw = nn.Parameter(torch.full((1,), gamma1_init, device=self.device))
        self.freq_raw = nn.Parameter(torch.full((1,), freq_init, device=self.device))



        # # 🔧 Effective ranges | ⚙️ freqmag Normalizers (ConVeXT)  
        # self.gamma1_eff = lambda: 0.5 + (3.0 - 0.5) * torch.sigmoid(self.gamma1_raw)    # gamma1 ∈ [0.5, 3.0]          
        # self.freq_factor_eff = lambda: 0.3 + (0.8 - 0.3) * torch.sigmoid(self.freq_raw) # freq_factor ∈ [0.3, 0.8] 
        # self.norm_freqmag = FreqMagNorm(target_min=0.05, target_max=1.5) 


        # # 🔧 Transformer-calibrated ranges | ⚙️ freqmag Normalizers (Transformer) 
        # self.gamma1_eff = lambda: 0.5 + (1.8 - 0.5) * torch.sigmoid(self.gamma1_raw)    # γ₁ ∈ [0.5, 1.8]
        # self.freq_factor_eff = lambda: 0.1 + (0.6 - 0.1) * torch.sigmoid(self.freq_raw) # freq ∈ [0.1, 0.6]  
        # self.norm_freqmag = FreqMagNorm(target_min=0.01, target_max=0.4)



        # 🔧 Effective ranges (Cifar) | ⚙️ freqmag Normalizers (Cifar)    
        self.gamma1_eff = lambda: 0.5 + (3.0 - 0.5) * torch.sigmoid(self.gamma1_raw)    # γ₁ ∈ [0.5, 3.0]
        self.freq_factor_eff = lambda: 0.2 + (0.8 - 0.2) * torch.sigmoid(self.freq_raw) # freq ∈ [0.2, 0.8]
        self.norm_freqmag = FreqMagNorm(target_min=0.03, target_max=1.0)



        # 🔢 Local forward-step counter (resets each epoch) | Local forward-step counter
        # 🔖 Tracks how many forward passes this layer has done within the current epoch.
        # 🔖 Used for spectral warmup logic and internal history management — independent per FFTGate layer.
        self._step = 0


        # 🔍 Track last logged epoch for debug printing        
        self._last_logged_epoch = -1


        # 🔄 Buffer to store current epoch (updated each epoch in train loop)
        self.register_buffer("current_epoch", torch.tensor(0, dtype=torch.long), persistent=False)       
        # ─────────────────────────────────────────────────────────────────────────────────────────────────




    # ===============================================================
    # 🔍 Non-monotonic decay schedule
    # ===============================================================
    def sigmoid_blended_decay(self, epoch, t1=50, t2=70, k1=0.25, k2=0.5, a=0.980, b=0.995, c=0.9801):
        """📉 Sigmoid-based phase-decay transition"""
        s1 = 1 / (1 + math.exp(-k1 * (epoch - t1)))
        s2 = 1 / (1 + math.exp(-k2 * (epoch - t2)))
        return a * (1 - s1) + b * s1 * (1 - s2) + c * s2



    # ===============================================================
    # 🔄 Update history with batch-averaged activations
    # ===============================================================
    def update_history(self, x):
        if not self.enable_history:
            return

        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Reduce to per-channel vector (C,)
        if x.dim() == 4:  
            #❗CNN-style: [B, C, H, W] → average over batch + spatial
            x = x.mean(dim=(0, 2, 3))
        elif x.dim() == 3:  
            #❗Transformer-style [B, T, C] → average over batch + sequence length
            x = x.mean(dim=(0, 1))
        elif x.dim() == 2:  
            #❗Simple MLP-style: [B, C] → average over batch
            x = x.mean(dim=0)
        else:
            #❗If neither 2D/3D/4D, throw error
            raise ValueError(f"Unexpected input shape {x.shape}")
        # ─────────────────────────────────────────────────────────────────────────────────────────────────


        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🧩 Freeze this copy for history (no gradients stored)
        x_hist = x.detach().to(self.device).unsqueeze(0)

        # ✅ Manage history buffer
        if self.activation_history is None:
            # First entry → init buffer
            self.activation_history = x_hist

        elif self.activation_history.size(0) < self.history_len:
            # Warmup → keep stacking
            self.activation_history = torch.cat(
                [self.activation_history, x_hist], dim=0
            )
        else:
            # Full buffer → roll and append
            self.activation_history = torch.cat(
                [self.activation_history[1:], x_hist], dim=0
            )
        # ─────────────────────────────────────────────────────────────────────────────────────────────────



    # ===============================================================
    # 🔍 Decay stored history
    # ===============================================================
    def decay_spectral_history(self, epoch, num_epochs=num_epochs):
        if not self.enable_history:
            return
        with torch.no_grad():
            if self.decay_mode == "linear":
                decay = self.sigmoid_blended_decay(epoch)                             # 🟢 Use Non-Monotonic decay Strategy
                self.activation_history *= decay
            
            elif self.decay_mode == "exp":                                            # 🟢 Use Exponential decay Strategy
                decay_factor = 0.99 + 0.01 * math.cos(math.pi * epoch / num_epochs)
                self.activation_history *= decay_factor

            elif self.decay_mode == "off":                                            # 🚫 Keep full history intact (no temporal scaling)                
                pass

            else:
                raise ValueError(
                    f"[FFTGate] Unknown decay_mode: {self.decay_mode}. "
                    f"Valid options are 'linear', 'exp', or 'off'."
                )









    # ===============================================================
    # 🔗 == Forward pass (with tiered warmup for freq_magnitude)==🔗
    # ===============================================================
    def forward(self, x):

        self._step += 1


        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ read epoch from buffer (no need to pass in forward)
        epoch = int(self.current_epoch.item())  

 
        # 🔍 Sanity print: Log when epoch changes
        if epoch != self._last_logged_epoch:
            if not hasattr(self.__class__, "_epoch_log_cache"):
                self.__class__._epoch_log_cache = []
            
            self.__class__._epoch_log_cache.append(self.layer_depth)

            # When all layers have logged, print once and reset
            if len(self.__class__._epoch_log_cache) == self.total_layers:
                layers_str = ", ".join(str(i) for i in sorted(self.__class__._epoch_log_cache))
                print(f"🧭 [FFTGate] Epoch {epoch} | Layers {layers_str}/{self.total_layers} | Device: {self.device}")
                self.__class__._epoch_log_cache.clear()

            self._last_logged_epoch = epoch
        # ─────────────────────────────────────────────────────────────────────────────────────────────────




        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Ensure input lives on same device as module params/buffers
        x = x.to(self.device)

        if self.enable_history:
            self.update_history(x)
            if self.decay_mode != "off":  # 🚫 Skip decay call when decay_mode == "off"
                self.decay_spectral_history(epoch)
        # ─────────────────────────────────────────────────────────────────────────────────────────────────





        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔄 freq_magnitude (tiered warmup) 
        if self._step < self.history_len:
            # 🟡 Partial history available
            h = self.activation_history[-self._step:]  # only filled rows
            # ✅ Pad to full length to keep FFT safe
            pad_len = self.history_len - h.size(0)
            if pad_len > 0:
                pad = torch.zeros(pad_len, h.size(1), device=h.device)
                h = torch.cat([pad, h], dim=0)   # [history_len, C]
        else:
            # 🟢 Full buffer
            h = self.activation_history
        # ─────────────────────────────────────────────────────────────────────────────────────────────────


        
        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔄 Channel-wise spectrum with normalization + shrinkage | Safe FFT (always [history_len, C])
        freq_response  = torch.fft.fft(h, dim=0)
        # freq_mag_hist  = torch.abs(freq_response).mean(dim=0)   # [C]
        freq_mag_hist = torch.nan_to_num(torch.abs(freq_response), nan=0.0, posinf=1e6, neginf=-1e6).mean(dim=0)  # [C] | Safe magnitude extraction (prevent NaN/inf from ComplexHalf FFT overflow)
        freq_magnitude = self.norm_freqmag(freq_mag_hist)       # [C] | normalised



        # 🔧 Optional smoothing (James–Stein shrinkage) | Smoothing across channels to prevent spikes based
        if self.use_smoothing: 
            smoothing_factor = max(0.1, 1 / (epoch + 10))
            freq_magnitude = (1 - smoothing_factor) * freq_magnitude + smoothing_factor * freq_magnitude.mean()


        # ✅ reshape for broadcasting
        if x.dim() == 3:  
            # Transformer [T, B, C]
            freq_magnitude = freq_magnitude.view(1, 1, -1)   # ✅ [1,1,C]
        elif x.dim() == 4:  
            # CNN [B, C, H, W]
            freq_magnitude = freq_magnitude.view(1, -1, 1, 1)  # ✅ [1,C,1,1]
        elif x.dim() == 2:
            # MLP [B, C]
            freq_magnitude = freq_magnitude.view(1, -1)       # ✅ [1,C]
        else:
            raise ValueError(f"[FFTGate] Unsupported x shape {x.shape} for freq_magnitude reshape")
        # ─────────────────────────────────────────────────────────────────────────────────────────────────    





        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔄 Normalize trainable params to clamp 
        gamma1_b = self.gamma1_eff()       
        freq_b   = self.freq_factor_eff()  



        # ⚙️ Disable spectral modulation when use_freq_gate=False
        if not self.use_freq_gate:
            freq_magnitude = torch.zeros_like(freq_magnitude)



        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # # 🔄 Core gating❗️🔑
        # gate = torch.sigmoid(gamma1_b * x - freq_b * freq_magnitude)  


        # # 🧠 Normalize gate relative to x variance (ConVeXT)
        # x_var = x.var(unbiased=False) + 1e-5        
        # gate_var = gate.var(unbiased=False) + 1e-5  
        # gate = gate * (x_var.sqrt() / gate_var.sqrt())


        # # 🧠 Normalize gate relative to x variance (Transformer)
        # x_std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-3)
        # gate_std = gate.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-3)
        # gate = gate * (x_std / gate_std)


        # # 🧠 Normalize gate relative to x variance (VGG16-safe, per-channel)
        # x_var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False) + 1e-5
        # gate_var = gate.var(dim=(0, 2, 3), keepdim=True, unbiased=False) + 1e-5
        # gate = gate * (x_var.sqrt() / gate_var.sqrt())
        # ─────────────────────────────────────────────────────────────────────────────────────────────────





        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔄 Core gating❗️🔑
        pre_gate = gamma1_b * x - freq_b * freq_magnitude

        # 🔍 Debug: monitor pre_gate stats before clamping (milestone-based)
        milestone_epochs = {0, 1, 3, 5, 10, 20, 30, 50, 80, 95}
        detailed_steps = {0, 1, 2, 5, 10, 50, 70, 100, 200, 500, 700, 750}

        if (epoch in milestone_epochs) and (self._step in detailed_steps):
            with torch.no_grad():
                pre_gate_mean = pre_gate.mean().item()
                pre_gate_min = pre_gate.min().item()
                pre_gate_max = pre_gate.max().item()

                # ✅ Format the log line
                log_line = (
                    f"[FFTGate Debug] Epoch {epoch:>3d} | Step {self._step:>4d} "
                    f"| pre_gate mean={pre_gate_mean:+.3f} | min={pre_gate_min:+.3f} | max={pre_gate_max:+.3f}\n"
                )

                # ✅ Define file path
                gate_log_path = "./Results/FFTGate/FFTGate_gate.txt"
                import os
                os.makedirs(os.path.dirname(gate_log_path), exist_ok=True)

                # ✅ Clear previous log at the start of training (epoch 0)
                if epoch == 0 and self._step == 1:  # only once at training start
                    with open(gate_log_path, "w", encoding="utf-8") as log_file:
                        log_file.write("")  # clear old content

                # ✅ Append the new log line
                with open(gate_log_path, "a", encoding="utf-8") as f:
                    f.write(log_line)

        # 🧩 Clamp pre_gate safely before sigmoid to prevent overflow or NaN
        pre_gate = pre_gate.clamp(-20, 20)  # was (-30, 30)

        # 🔄 Compute final gate activation
        gate = torch.sigmoid(pre_gate)
        # ─────────────────────────────────────────────────────────────────────────────────────────────────







        #🔄 Final activation
        activation = x * gate
        # ─────────────────────────────────────────────────────────────────────────────────────────────────



         # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔧 Optional skip connection | Annealed skip connection (0.05 → 0.0 over first 20 epochs)
        if self.use_skip:
            skip_w = max(0.0, 0.05 * (1 - epoch / 20))
            activation = (1 - skip_w) * activation + skip_w * x
       # ─────────────────────────────────────────────────────────────────────────────────────────────────


        # ✅ Always return activation
        return activation











# %%






