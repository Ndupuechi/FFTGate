




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


########################################################################################################################
####-------| NOTE 1. DEFINE FFTGATE ACTIVATION CLASS | XXX -----------------------------------------####################
########################################################################################################################

# ✅ Define num_epochs
num_epochs = 100    # Paper use "num_epochs = 300 "


# ✅ FFTGATE activation (per-channel params)
class FFTGate(nn.Module):
    """
    FFTGate Activation Function

    Parameters:
    ----------
    gamma1 : float
        Trainable scaling factor applied to pre-activation input (controls gating sharpness).
        Initialized as a learnable parameter.

    phi : float
        Frequency of the sinusoidal perturbation added to the gated activation output to prevent
        neuron saturation and improve learning smoothness.

    history_len : int
        Temporal activation history window length (T), i.e., number of past activation states stored
        for frequency-domain analysis.

    enable_history : bool
        Enables or disables tracking of temporal activation history. If False, FFT-based modulation is skipped.

    use_skip : bool
        Toggles an optional residual skip connection that blends the input and modulated output (0.95 * activation + 0.05 * input).

    decay_mode : str
        Temporal Activation History Decay Strategy:
        - "exp"    → Exponential decay strategy (cosine-based, monotonic)
        - "linear" → Non-monotonic decay strategy using sigmoid-blended multi-phase decay
    """

    def __init__(
            self,
            num_channels,               # 🟡 NEW: required channel count (C)
            gamma1_init=1.5,
            freq_init=0.3,
            phi=0.1,
            history_len=16,             
            enable_history=True,
            use_skip=True,
            decay_mode="linear"):       # 🟢  default for Test:use_skip=False, history_len=16

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.C = int(num_channels)        # 🟡 NEW: save C (channels)

        # 🔧 helper to map an effective init value in [lo,hi] → raw logit space
        def _to_raw(val, lo, hi, eps=1e-4):
            p = (float(val) - lo) / (hi - lo)
            p = min(max(p, eps), 1.0 - eps)
            return math.log(p / (1.0 - p))

        # 🔧 per-channel trainable RAW params((shape [C], not [1]))
        self.gamma1_raw = nn.Parameter(
            torch.full((self.C,), _to_raw(gamma1_init, 0.10, 13.0), device=self.device)
        )
        self.freq_raw = nn.Parameter(
            torch.full((self.C,), _to_raw(freq_init, 0.01, 1.30), device=self.device)
        )

        # convenient accessors for logging / regularization
        def _gamma1_eff():
            return 0.10 + 12.90 * torch.sigmoid(self.gamma1_raw)    # 13.0 - 0.10
        def _freq_eff():
            return 0.01 + 1.29 * torch.sigmoid(self.freq_raw)       # 1.30 - 0.01
        # bind as methods
        self.gamma1_eff = _gamma1_eff
        self.freq_factor_eff = _freq_eff

        # 🔒 Fixed buffer phi (No gradient updates)
        self.register_buffer('phi', torch.tensor(phi, device=self.device))

        # ✅ History settings
        self.history_len = history_len
        self.enable_history = enable_history
        self.activation_history_initialized = False

        # ⚙️ Optional skip connection toggle
        self.use_skip = use_skip

        # ✅ Save selected decay strategy
        self.decay_mode = decay_mode

        # 🟡🟡 UPDATE: hyperparams for differentiable frequency blend & numerical stability
        self.freq_alpha = 0.10              # live-current blend weight (0..1)
        self.freq_eps = 1e-6                # eps for std/denominators

    # 🔍 Computes sigmoid-blended multi-phase decay rate used for non-monotonic history scaling | non-monotonic decay strategy definition
    def sigmoid_blended_decay(self, epoch, t1=50, t2=70, k1=0.25, k2=0.5,
                              a=0.980, b=0.995, c=0.9801):
        """📉 Sigmoid-based phase-decay transition"""
        s1 = 1 / (1 + math.exp(-k1 * (epoch - t1)))
        s2 = 1 / (1 + math.exp(-k2 * (epoch - t2)))
        decay = a * (1 - s1) + b * s1 * (1 - s2) + c * s2
        return decay

    # 🔄 Update history with batch-averaged activations | 🔍 Tracks moving average of pre-activation inputs over a history window (per channel)
    def update_history(self, x):
        if not self.enable_history:
            return

        # Average over batch and spatial dims
        if x.dim() == 4:
            x = x.mean(dim=(0, 2, 3))
        elif x.dim() == 2:
            x = x.mean(dim=0)
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")

        x = x.detach()

        # 🔁  Perform buffer update without autograd tracking
        with torch.no_grad():
            # Reset buffer if shape changes
            if not self.activation_history_initialized:
                # 🟡 Initialize history buffer once as a registered buffer
                self.register_buffer(
                    "activation_history",
                    torch.zeros(self.history_len, x.shape[0], device=x.device, dtype=x.dtype)
                )
                self.activation_history_initialized = True
            elif self.activation_history.shape[1] != x.shape[0]:
                # 🟡 Avoid re-registering; just replace the tensor to keep it a buffer
                self.activation_history = torch.zeros(
                    self.history_len, x.shape[0], device=x.device, dtype=x.dtype
                )

            # 🔥 Roll buffer + append new value
            self.activation_history = torch.cat([self.activation_history[1:], x.unsqueeze(0)], dim=0)

    # 🔍 Decays the stored activation history based on selected temporal decay strategy
    def decay_spectral_history(self, epoch, num_epochs=num_epochs):
        if not (self.enable_history and self.activation_history_initialized):
            return

        with torch.no_grad():
            if self.decay_mode == "linear":                     # 🟢 Use Non-Monotonic decay Strategy
                decay_rate = self.sigmoid_blended_decay(epoch)
                self.activation_history *= decay_rate
            elif self.decay_mode == "exp":                      # 🟢 Use Exponential decay Strategy
                decay_factor = 0.99 + 0.01 * math.cos(math.pi * epoch / num_epochs)
                self.activation_history *= decay_factor
            else:
                raise ValueError(f"Unknown decay_mode: {self.decay_mode}. Use 'exp' or 'linear'.")

    def forward(self, x, epoch=0):
        self.saved_output = x.detach()

        if self.enable_history:
            self.update_history(x)

        # 🟡🟡 UPDATE: helper to z-normalize safely (no grad side effects)
        def _znorm(v):
            m = v.mean()
            s = v.std().clamp_min(self.freq_eps)
            return (v - m) / s, m, s

        # 🟡🟡 UPDATE: compute a differentiable current-step spectrum per channel
        if x.dim() == 4:
            # [B,C,H,W] → average over batch only; keep spatial to get 2D spectrum
            x_b = x.mean(dim=0, keepdim=False)                        # [C,H,W]
            X = torch.fft.rfft2(x_b, dim=(-2, -1), norm="ortho")
            cur_spec_raw = (X.abs() ** 2).mean(dim=(-2, -1))          # [C], differentiable wrt x
        elif x.dim() == 2:
            # [B,C] → use per-channel variance as a differentiable magnitude proxy
            cur_spec_raw = x.var(dim=0, unbiased=False)               # [C]
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")

        # 🟡🟡 UPDATE: history-based magnitude (detached) computed as before
        have_hist = self.enable_history and self.activation_history_initialized
        if have_hist:
            self.decay_spectral_history(epoch)
            freq_response  = torch.fft.fft(self.activation_history, dim=0)   # [T,C]
            hist_mag_raw   = torch.abs(freq_response).mean(dim=0)            # [C], no grad path anyway
            hist_mag_raw   = torch.clamp(hist_mag_raw, min=0.05, max=1.5)
        else:
            hist_mag_raw = None

        # 🟡🟡 UPDATE: scale current step to the *history* statistics, then blend
        if hist_mag_raw is not None:
            # z-normalize each source; scale current to history’s mean/std so units match
            cur_norm, cur_m, cur_s   = _znorm(cur_spec_raw)
            _, hist_m, hist_s        = _znorm(hist_mag_raw)
            cur_scaled               = cur_norm * hist_s + hist_m          # [C], differentiable
            freq_mag_blend           = (1.0 - self.freq_alpha) * hist_mag_raw.detach() + self.freq_alpha * cur_scaled
        else:
            # no history yet → just use current differentiable magnitude
            freq_mag_blend = cur_spec_raw

        # 🟡🟡 UPDATE: mild per-batch smoothing (same spirit as before), keep positive range
        smoothing_factor = max(0.1, 1 / (epoch + 10))
        freq_magnitude   = (1 - smoothing_factor) * freq_mag_blend + smoothing_factor * freq_mag_blend.mean()
        freq_magnitude   = torch.clamp(freq_magnitude, min=0.05, max=1.5)

        # 🟡 CHANGED: get per-channel vectors and reshape for broadcast
        gamma1_vec    = self.gamma1_eff()                                    # [C]
        freq_vec      = self.freq_factor_eff()                               # [C]
        if x.dim() == 4:
            gamma1_b  = gamma1_vec.view(1, -1, 1, 1)                         # 🟡
            freq_b    = freq_vec.view(1, -1, 1, 1)                           # 🟡
            freq_magnitude_b = freq_magnitude.view(1, -1, 1, 1)              # 🟡🟡 UPDATE
        else:  # (N,C)
            gamma1_b  = gamma1_vec.view(1, -1)                               # 🟡
            freq_b    = freq_vec.view(1, -1)                                 # 🟡
            freq_magnitude_b = freq_magnitude.view(1, -1)                    # 🟡🟡 UPDATE

        # 🔄 Gate using blended FFT-derived frequency magnitude (history⊕current, current carries grads)
        gate        = torch.sigmoid(gamma1_b * x - freq_b * freq_magnitude_b)  # 🟡 uses per-channel params

        # 🧠 Main activation logic: gated + sinusoidal regularization
        activation  = x * gate + 0.05 * torch.sin(self.phi * x)

        # 🔁 Optional: smart residual skip blend
        if self.use_skip:
            activation = 0.95 * activation + 0.05 * x

        return activation





# %%
