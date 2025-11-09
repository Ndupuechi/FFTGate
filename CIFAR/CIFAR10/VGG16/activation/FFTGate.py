










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


# ✅ Import Custom libraries
import os, sys
Activation_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if Activation_PATH not in sys.path:
    sys.path.append(Activation_PATH)

from FreqMagNorm_FFTGateLogger import FreqMagNorm, GateLogger





# ==============================================================================================================================
# ==============================================================================================================================
# 🔗=======================================🔑 FFTGATE ACTIVATION FUNCTION 🔑================================================🔗
# ==============================================================================================================================
# ==============================================================================================================================

class FFTGate(nn.Module):


    # ===============================================================
    # 🔧 Initialization
    # ===============================================================
    def __init__(self,
                 gamma1_init=1.5,        # ⬅️ 1.5 → 0.8
                 freq_init=0.3,          # ⬅️ 0.3 → 0.1                  
                 history_len=32,         
                 num_epochs = 100,
                 enable_history=True,

                 use_skip=True,          
                 decay_mode="linear",     
                 use_freq_gate=True,     
                 use_smoothing=True, 
                 arch_type="classic",            

                 layer_depth=None,       
                 total_layers=6,          # 👈 default placeholder   

                 gate_log_path=None,
                 gate_history=None   
                 ):        
        super().__init__()         

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔒 Store configuration flags and state variables
        #     - activation_history : buffer for past activations used in FFT computation (initialized empty)
        #     - history_len        : number of steps retained in rolling spectral history
        #     - enable_history     : toggle for maintaining activation history
        #     - use_skip           : enables residual skip connection
        #     - decay_mode         : controls temporal decay strategy for spectral history ('linear' or 'off')
        #     - use_freq_gate      : enables frequency-based modulation in the gating function
        #     - use_smoothing      : enables James–Stein smoothing across channels
        #     -  arch_type         : select architecture "convnext", "transformer" or classic
        self.activation_history = None  
        self.history_len = int(history_len)
        self.enable_history = enable_history
        self.use_skip = use_skip
        self.decay_mode = decay_mode
        self.use_freq_gate = use_freq_gate
        self.use_smoothing = use_smoothing
        self.num_epochs = num_epochs
        self.arch_type = arch_type

        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # ⚡ Store layer depth for scaling
        if layer_depth is None:
            # ✅ Default to placeholder (will be updated later by find_activations_split)
            self.layer_depth = 1
            print("⚠️ FFTGate: layer_depth temporarily set to 1 (will be updated later)")
        else:
            self.layer_depth = layer_depth

        self.total_layers = total_layers
        # ─────────────────────────────────────────────────────────────────────────────────────────────────

        # ===============================================================
        # 🧾 Logging FFTGate Params setup
        # ===============================================================
        if gate_log_path is None:
            gate_log_path = {"gate_history": "./Results/FFTGate/FFTGate_gate.txt"}
        if gate_history is None:
            gate_history = []

        self.gate_log_path = gate_log_path
        self.gate_history = gate_history

        # 📜 Create GateLogger modules (import from your helper file)
        self.logger_raw  = GateLogger(self.gate_log_path, self.gate_history, label="RAW")
        self.logger_norm = GateLogger(self.gate_log_path, self.gate_history, label="NORM")


        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔧 trainable raw params 
        self.gamma1_raw = nn.Parameter(torch.full((1,), gamma1_init, device=self.device))
        self.freq_raw = nn.Parameter(torch.full((1,), freq_init, device=self.device))
        # ─────────────────────────────────────────────────────────────────────────────────────────────────


        # ─────────────────────────────────────────────────────────────────────────────────────────────────   
        # 🧩 Compute Effective (bounded) trainable parameters
        # 🔖 ensuring stable scaling and preventing unbounded growth during training
        if self.arch_type in ["convnext", "classic"]:
            # 🔧 CNN, etc
            self.gamma1_eff = lambda: 0.5 + (2.5 - 0.5) * torch.sigmoid(self.gamma1_raw)    # γ₁ ∈ [0.5, 3.0] 
            self.freq_factor_eff = lambda: 0.3 + (0.8 - 0.3) * torch.sigmoid(self.freq_raw) # freq ∈ [0.3, 0.8]
            self.norm_freqmag = FreqMagNorm(target_min=0.05, target_max=1.5)

        elif self.arch_type == "transformer":
            # 🔧 Transformer
            self.gamma1_eff = lambda: 0.5 + (1.8 - 0.5) * torch.sigmoid(self.gamma1_raw)    # γ₁ ∈ [0.5, 1.8]
            self.freq_factor_eff = lambda: 0.1 + (0.6 - 0.1) * torch.sigmoid(self.freq_raw) # freq ∈ [0.1, 0.6]
            self.norm_freqmag = FreqMagNorm(target_min=0.01, target_max=0.4)

        else:
            raise ValueError(f"Unknown arch_type: {self.arch_type}")
        # ─────────────────────────────────────────────────────────────────────────────────────────────────


        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔢 Forward-pass counters
        # 🔖 _step: Global forward count across training — used for rolling FFT window management.
        # 🔖 _epoch_step: Per-epoch batch counter (resets each epoch manually in the train loop).
        # 🔖 Both are tracked independently per FFTGate layer.
        self._step = 0           # 🔢 global forward count (for rolling FFT window)
        self._epoch_step = 0     # 🔢 per-epoch batch counter (for debug / milestone logs)

        # 🔍 Track last logged epoch for debug printing        
        self._last_logged_epoch = -1

        # 🔄 Buffer to store current epoch (updated each epoch in train loop)
        self.register_buffer("current_epoch", torch.tensor(0, dtype=torch.long), persistent=False)     
        # ─────────────────────────────────────────────────────────────────────────────────────────────────



    # ===============================================================
    # 🔍 Non-monotonic decay schedule
    # ===============================================================
    # ⚙️ Computes sigmoid-blended multi-phase decay rate used for non-monotonic history scaling
    def sigmoid_blended_decay(self, epoch,
                            t1=50, t2=70,
                            k1=0.25, k2=0.5,
                            a=0.980, b=0.995, c=0.9801):
        """📉 Sigmoid-based phase-decay transition"""

        # Rescale current epoch to fit 0–100 reference scale
        epoch_scaled = (epoch / self.num_epochs) * 100

        s1 = 1 / (1 + math.exp(-k1 * (epoch_scaled - t1)))
        s2 = 1 / (1 + math.exp(-k2 * (epoch_scaled - t2)))
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
    def decay_spectral_history(self, epoch):
        if not self.enable_history:
            return
        with torch.no_grad():
            if self.decay_mode == "linear":
                decay = self.sigmoid_blended_decay(epoch)     # 🟢 Use Non-Monotonic decay Strategy
                self.activation_history *= decay            

            elif self.decay_mode == "off":                    # 🚫 Keep full history intact (no temporal scaling)                
                pass

            else:
                raise ValueError(
                    f"[FFTGate] Unknown decay_mode: {self.decay_mode}. "
                    f"Valid options are 'linear' or 'off'."
                )


    # ===============================================================
    # ===============================================================
    # 🔗 == Forward pass (with tiered warmup for freq_magnitude)==🔗
    # ===============================================================
    # ===============================================================
    def forward(self, x):

        self._step += 1          # 🟢 keeps accumulating for rolling FFT
        self._epoch_step += 1    # 🟡 Resets each epoch (manually in train loop)

        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Read epoch from buffer (no need to pass in forward)
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
        freq_mag_hist = torch.nan_to_num(torch.abs(freq_response), nan=0.0, posinf=1e6, neginf=-1e6).mean(dim=0)  # [C] |⚠️ Safe magnitude extraction (prevent NaN/inf from ComplexHalf FFT overflow)
        freq_magnitude = self.norm_freqmag(freq_mag_hist)       # [C] | normalised

        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔧 Optional smoothing (James–Stein shrinkage) | Smoothing across channels to prevent spikes based
        if self.use_smoothing: 
            smoothing_factor = max(0.1, 1 / (epoch + 10))
            freq_magnitude = (1 - smoothing_factor) * freq_magnitude + smoothing_factor * freq_magnitude.mean()


        # ✅ Reshape for broadcasting
        if x.dim() == 3:  
            # Transformer [T, B, C]
            freq_magnitude = freq_magnitude.view(1, 1, -1)     # ✅ [1,1,C]
        elif x.dim() == 4:  
            # CNN [B, C, H, W]
            freq_magnitude = freq_magnitude.view(1, -1, 1, 1)  # ✅ [1,C,1,1]
        elif x.dim() == 2:
            # MLP [B, C]
            freq_magnitude = freq_magnitude.view(1, -1)        # ✅ [1,C]
        else:
            raise ValueError(f"[FFTGate] Unsupported x shape {x.shape} for freq_magnitude reshape")
        # ─────────────────────────────────────────────────────────────────────────────────────────────────    


        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔄 Compute effective (bounded) trainable parameters
        gamma1_b = self.gamma1_eff()       
        freq_b   = self.freq_factor_eff()  


        # ⚙️ Disable spectral modulation when use_freq_gate=False
        if not self.use_freq_gate:
            freq_magnitude = torch.zeros_like(freq_magnitude)
        # ─────────────────────────────────────────────────────────────────────────────────────────────────


        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔄 Gating 🔑🔑
        pre_gate = gamma1_b * x - freq_b * freq_magnitude

        # ────────
        # 1️⃣ Log RAW pre_gate statistics before normalization
        pre_gate_raw_print = self.logger_raw(pre_gate, epoch, self._epoch_step, self._step)  # 🧾 log only (no effect on forward output)
        # ────────
        # ─────────────────────────────────────────────
        # 🧠 Architecture-specific normalization
        if self.arch_type == "convnext":
            # 🟡 CNN (ConvNeXt)
            gate = torch.sigmoid(pre_gate)
            x_var = x.var(unbiased=False) + 1e-5        
            gate_var = gate.var(unbiased=False) + 1e-5  
            gate = gate * (x_var.sqrt() / gate_var.sqrt())

        elif self.arch_type == "transformer":
            # 🟡 Transformer
            gate = torch.sigmoid(pre_gate)
            x_std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-3)
            gate_std = gate.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-3)
            gate = gate * (x_std / gate_std)

        elif self.arch_type == "classic":
            # 🟡 Classic CNN (VGG, SENet), MLP, etc.
            pre_gate = pre_gate.clamp(-20, 20)
            gate = torch.sigmoid(pre_gate)

        else:
            raise ValueError(f"Unknown arch_type: {self.arch_type}")
        # ─────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────────────────────────────────

        # ────────
        # 2️⃣ Log NORM pre_gate statistics after normalization/clamping
        pre_gate_norm_print = self.logger_norm(pre_gate, epoch, self._epoch_step, self._step)  # 🧾 log only (no effect on forward output)
        # ────────


        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        #🔄 Final activation
        activation = x * gate

        # 🔧 Optional skip connection 
        if self.use_skip:
            activation = 0.95 * activation + 0.05 * x
       # ─────────────────────────────────────────────────────────────────────────────────────────────────


        # ✅ Always return activation
        return activation






# %%






