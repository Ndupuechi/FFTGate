









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
        # Just bound it into [target_min, target_max]
        return torch.clamp(mean_val, self.target_min, self.target_max)
















# ===============================================================
# 🔗==================== FFTGATE activation 🔑================🔗
# ===============================================================

class FFTGate(nn.Module):


    # ===============================================================
    # 🔧 Initialization
    # ===============================================================
    def __init__(self, num_channels, gamma1_init=1.5, freq_init=0.3, phi=0.1, history_len=32,                         # ⚡ Default: "gamma1_init=1.5, freq_init=0.3"  | # 🔒 FIX: allow smarter init                           
                 enable_history=True, use_skip=False, decay_mode="linear", layer_depth=None, total_layers=None):      # ⚡layer_depth=None, total_layers=None | 🟢  default for Test:use_skip=False, history_len=16
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.activation_history = None  # always exists, starts empty
        self.C = int(num_channels)
        self.history_len = int(history_len)
        self.enable_history = enable_history
        self.use_skip = use_skip
        self.decay_mode = decay_mode




        # ⚡store layer depth for scaling
        if layer_depth is None:
            raise ValueError("layer_depth must be provided by Block")
        self.layer_depth = layer_depth
        self.total_layers = total_layers





        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔧 trainable raw params 
        self.gamma1_raw = nn.Parameter(torch.full((1,), gamma1_init, device=self.device))
        self.freq_raw = nn.Parameter(torch.full((1,), freq_init, device=self.device))


        # 🔧 Effective ranges (ORIGINAL)    
        self.gamma1_eff = lambda: 0.5 + (3.0 - 0.5) * torch.sigmoid(self.gamma1_raw)    # gamma1 ∈ [0.5, 3.0]          
        self.freq_factor_eff = lambda: 0.3 + (0.8 - 0.3) * torch.sigmoid(self.freq_raw) # freq_factor ∈ [0.3, 0.8] 




        # 🔧 Normalizers 
        self.norm_freqmag = FreqMagNorm(target_min=0.05, target_max=1.5)    # was [0.01, 3.0] 


        # 🔄 Buffers            
        self.register_buffer('phi', torch.tensor(phi))  # 🔒 Fixed buffer phi (No gradient updates)
        self._step = 0
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

        # ✅ Reduce to per-channel vector (C,)
        if x.dim() == 4:
            x = x.mean(dim=(0, 2, 3))
        elif x.dim() == 2:
            x = x.mean(dim=0)
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")

        # ✅ Freeze this copy for history (no gradients stored)
        x_hist = x.detach().to(self.device).unsqueeze(0)

        # ✅ Manage history buffer
        if self.activation_history is None:
            # First entry → init buffer
            self.activation_history = x_hist
        elif self.activation_history.shape[1] != x.shape[0]:
            # 🚨 Channel mismatch
            raise RuntimeError(
                f"[FFTGate] Channel mismatch in update_history! "
                f"Expected {self.activation_history.shape[1]} channels, got {x.shape[0]}."
            )
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



    # ===============================================================
    # 🔍 Decay stored history
    # ===============================================================
    def decay_spectral_history(self, epoch, num_epochs=num_epochs):
        if not self.enable_history:
            return
        with torch.no_grad():
            if self.decay_mode == "linear":
                decay = self.sigmoid_blended_decay(epoch)                            # 🟢 Use Non-Monotonic decay Strategy

                # 🔵 exploration bump: reduce history weight mid-phase
                if 15 <= epoch <= 40:
                    decay *= 0.95   # 5% weaker decay → current signal has more say

                self.activation_history *= decay
            elif self.decay_mode == "exp":                                          # 🟢 Use Exponential decay Strategy
                decay_factor = 0.99 + 0.01 * math.cos(math.pi * epoch / num_epochs)
                self.activation_history *= decay_factor
            else:
                raise ValueError(f"Unknown decay_mode: {self.decay_mode}")




    # ===============================================================
    # 🔗 == Forward pass (with tiered warmup for freq_magnitude)==🔗
    # ===============================================================
    def forward(self, x, epoch=0):

        self._step += 1


        # # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # # ⚡ Debug print: collect γ₁_eff and freq_eff for ALL layers
        # if epoch in [0, 1, 5] and self._step in [1, 2, 8, 50, 1000, 4000]:
        #     # init buffer ONCE at start of batch
        #     if self.layer_depth == 1:
        #         FFTGate._debug_buffer = f"FFTGate-Forward|| Epoch={epoch} | Batch={self._step}"

        #     gamma1_eff_val = self.gamma1_eff().item()
        #     freq_eff_val   = self.freq_factor_eff().item()
            

        #     FFTGate._debug_buffer += (
        #         f" || Layer={self.layer_depth} : "
        #         f"γ₁={gamma1_eff_val:.4f}, f={freq_eff_val:.4f}"
        #     )

        #     # flush only at last layer
        #     if self.layer_depth == self.total_layers:
        #         print(FFTGate._debug_buffer)
        #         FFTGate._debug_buffer = None
        # # ⚡ End debug print
        # # ─────────────────────────────────────────────────────────────────────────────────────────────────



        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Ensure input lives on same device as module params/buffers
        x = x.to(self.device)

        if self.enable_history:
            self.update_history(x)
            self.decay_spectral_history(epoch)


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
        freq_mag_hist  = torch.abs(freq_response).mean(dim=0)   # [C]
        freq_magnitude = self.norm_freqmag(freq_mag_hist)       # [C] | normalised

        # 🔧 Optional smoothing (James–Stein shrinkage) | Smoothing across channels to prevent spikes based 
        smoothing_factor = max(0.1, 1 / (epoch + 10))
        freq_magnitude = (1 - smoothing_factor) * freq_magnitude + smoothing_factor * freq_magnitude.mean()

        # reshape for broadcasting
        freq_magnitude = freq_magnitude.view(1, -1, 1, 1)
        # ─────────────────────────────────────────────────────────────────────────────────────────────────    



        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # 🔄 Normalize trainable params to clamp 
        gamma1_b = self.gamma1_eff()       
        freq_b   = self.freq_factor_eff()  



        # 🔄 Core gating❗️🔑
        gate = torch.sigmoid(gamma1_b * x - freq_b * freq_magnitude)  

        # 🧠 Normalize gate relative to x variance (like GELU’s self-normalization) | Keeps output variance ≈ input variance
        x_var = x.var(unbiased=False) + 1e-5        # ✅ scalar variance
        gate_var = gate.var(unbiased=False) + 1e-5  # ✅ scalar variance
        gate = gate * (x_var.sqrt() / gate_var.sqrt())


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






