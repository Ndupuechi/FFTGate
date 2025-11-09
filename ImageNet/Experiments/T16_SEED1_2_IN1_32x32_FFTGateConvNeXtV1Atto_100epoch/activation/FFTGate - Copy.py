







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



num_epochs = 100
warmup_epochs = 5


class FFTGate(nn.Module):
    def __init__(self,
                 num_channels,
                 gamma1_init=1.5, freq_init=0.3, phi=0.1,
                 history_len=16, enable_history=True, use_skip=True,
                 decay_mode="linear"):
        super().__init__()
        self.C = int(num_channels)
        self.history_len = int(history_len)
        self.enable_history = enable_history
        self.use_skip = use_skip
        self.decay_mode = decay_mode


        # 🔧 per-channel trainable RAW params((shape [C], not [1]))
        self.gamma1_raw = nn.Parameter(torch.full((self.C,), gamma1_init))
        self.freq_raw   = nn.Parameter(torch.full((self.C,), freq_init))




        # 🔄 Buffers            
        self.register_buffer('phi', torch.tensor(phi))  # 🔒 Fixed buffer phi (No gradient updates)
        self.register_buffer("activation_history", torch.zeros(self.history_len, self.C))  # 🟡 initialize history & cached FFT on module (no device hops in forward)
        self._step = 0





    # 🔍 non-monotonic decay schedule (unchanged)
    def sigmoid_blended_decay(self, epoch, t1=50, t2=70, k1=0.25, k2=0.5, a=0.980, b=0.995, c=0.9801):
        """📉 Sigmoid-based phase-decay transition"""
        s1 = 1 / (1 + math.exp(-k1 * (epoch - t1)))
        s2 = 1 / (1 + math.exp(-k2 * (epoch - t2)))
        return a * (1 - s1) + b * s1 * (1 - s2) + c * s2




    # ===============================================================
    # 🟢 Helper: EMA Min–Max + smoothing normalization
    def ema_minmax_smooth(self, values, name, beta=0.9, smoothing=0.05):
        min_name, max_name = f"ema_min_{name}", f"ema_max_{name}"
        if not hasattr(self, min_name):
            self.register_buffer(min_name, values.clone())
            self.register_buffer(max_name, values.clone())

        ema_min = getattr(self, min_name)
        ema_max = getattr(self, max_name)

        ema_min = beta * ema_min + (1 - beta) * values
        ema_max = beta * ema_max + (1 - beta) * values

        setattr(self, min_name, ema_min)
        setattr(self, max_name, ema_max)

        normed = (values - ema_min) / (ema_max - ema_min + 1e-6)
        normed = torch.clamp(normed, 0.0, 1.0)

        normed = (1 - smoothing) * normed + smoothing * normed.mean()
        return normed
    # ===============================================================


    # 🔄 Update history with batch-averaged activations | 🔍 Tracks moving average of pre-activation inputs over a history window (per channel)
    def update_history(self, x):
        if not self.enable_history:
            return
        if x.dim() == 4:
            if x.shape[1] == self.C:      # NCHW
                x_mean = x.mean(dim=(0, 2, 3))
            elif x.shape[-1] == self.C:   # NHWC
                x_mean = x.mean(dim=(0, 1, 2))
            else:
                raise ValueError(f"Unexpected 4D shape {x.shape} for C={self.C}")
        elif x.dim() == 2 and x.shape[1] == self.C:
            x_mean = x.mean(dim=0)
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")
        x_mean = x_mean.detach()
        with torch.no_grad():
            self.activation_history = torch.cat(
                [self.activation_history[1:], x_mean.unsqueeze(0)],
                dim=0
            )







    # 🔍 Decays the stored activation history based on selected temporal decay strategy
    def decay_spectral_history(self, epoch, num_epochs=num_epochs):
        if not self.enable_history:
            return
        with torch.no_grad():
            if self.decay_mode == "linear":
                decay = self.sigmoid_blended_decay(epoch)                          # 🟢 Use Non-Monotonic decay Strategy

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
    # 🟢🔴 Forward pass (with tiered warmup for freq_magnitude)🔴🟢
    # ===============================================================
    def forward(self, x, epoch=0):
        self._step += 1

        if self.enable_history:
            self.update_history(x)
            self.decay_spectral_history(epoch)

        # ======= layout detection =======
        if x.dim() == 4 and x.shape[1] == self.C:
            layout = "NCHW"; red_dims = (0, 2, 3)
        elif x.dim() == 4 and x.shape[-1] == self.C:
            layout = "NHWC"; red_dims = (0, 1, 2)
        elif x.dim() == 2 and x.shape[1] == self.C:
            layout = "NC"; red_dims = (0,)
        else:
            raise ValueError(f"Unexpected input shape {x.shape} for C={self.C}")

        # ======= per-channel current statistic ======= 
        x_cur_stat = x.mean(dim=red_dims).abs()

        # ======= feature_norm = current batch ======= 
        feature_norm = self.ema_minmax_smooth(x_cur_stat, name="x")




        # ======= freq_magnitude (tiered warmup) =======
        if self._step == 1:
            # 🔵 First step → no history, just use feature_norm
            freq_magnitude = feature_norm
            if epoch <= 2:  # limit spam                
                print(f"[Epoch {epoch} | Step {self._step}] Warmup-Init → using feature_norm only | "
                    f"freq_magnitude: shape={freq_magnitude.shape}, min={freq_magnitude.min().item():.4f}, "
                    f"max={freq_magnitude.max().item():.4f}")

        elif self._step < self.history_len:
            # 🟡 Partial history available → FFT on shorter buffer
            h = self.activation_history[-self._step:]  # only use filled rows
            freq_response  = torch.fft.fft(h, dim=0)
            freq_mag_hist  = torch.abs(freq_response).mean(dim=0)
            freq_magnitude = self.ema_minmax_smooth(freq_mag_hist, name="freq")
            if epoch <= 2 and self._step in {2, 4, 8}:  # sample some steps
                print(f"[Epoch {epoch} | Step {self._step}] Warmup-Partial → FFT on {self._step} rows | "
                    f"freq_mag_hist: shape={freq_mag_hist.shape}, min={freq_mag_hist.min().item():.4f}, "
                    f"max={freq_mag_hist.max().item():.4f} | "
                    f"freq_magnitude: min={freq_magnitude.min().item():.4f}, "
                    f"max={freq_magnitude.max().item():.4f}")

        else:
            # 🟢 Full buffer → FFT on complete history
            h = self.activation_history
            freq_response  = torch.fft.fft(h, dim=0)
            freq_mag_hist  = torch.abs(freq_response).mean(dim=0)
            freq_magnitude = self.ema_minmax_smooth(freq_mag_hist, name="freq")
            if epoch in {17, 20} and self._step % 2000 == 0:  # periodic check
                print(f"[Epoch {epoch} | Step {self._step}] Full-FFT → using {self.history_len} rows | "
                    f"freq_mag_hist: shape={freq_mag_hist.shape}, min={freq_mag_hist.min().item():.4f}, "
                    f"max={freq_mag_hist.max().item():.4f} | "
                    f"freq_magnitude: min={freq_magnitude.min().item():.4f}, "
                    f"max={freq_magnitude.max().item():.4f}")






        # ======= normalize trainable params =======
        gamma1_b = self.ema_minmax_smooth(self.gamma1_raw, name="gamma1")
        freq_b   = self.ema_minmax_smooth(self.freq_raw, name="freq_b")

        # ======= reshape for broadcasting =======
        if layout == "NCHW":
            gamma1_b       = gamma1_b.view(1, self.C, 1, 1)
            freq_b         = freq_b.view(1, self.C, 1, 1)
            freq_magnitude = freq_magnitude.view(1, self.C, 1, 1)
            feature_norm   = feature_norm.view(1, self.C, 1, 1)
        elif layout == "NHWC":
            gamma1_b       = gamma1_b.view(1, 1, 1, self.C)
            freq_b         = freq_b.view(1, 1, 1, self.C)
            freq_magnitude = freq_magnitude.view(1, 1, 1, self.C)
            feature_norm   = feature_norm.view(1, 1, 1, self.C)
        else:  # "NC"
            gamma1_b       = gamma1_b.view(1, self.C)
            freq_b         = freq_b.view(1, self.C)
            freq_magnitude = freq_magnitude.view(1, self.C)
            feature_norm   = feature_norm.view(1, self.C)

        # 🔄 gated activation
        gate = torch.sigmoid(gamma1_b * feature_norm - freq_b * freq_magnitude)



        # 🧠 sinusoidal perturbation ❗️
        w_floor, w_max, T = 0.02, 0.05, 40   # floor, start, fade length
        t = min(epoch / T, 1.0)
        w = w_floor + 0.5 * (w_max - w_floor) * (1 + math.cos(math.pi * t))
        
        activation = feature_norm * gate + w * torch.sin(self.phi * feature_norm)


        # 🔁 Optional skip connection
        if self.use_skip:
            activation = 0.95 * activation + 0.05 * feature_norm

        return activation










# %%






