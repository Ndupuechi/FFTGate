







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








# ---------------- RMSNorm Rescale (per-channel) XXX OLD XXX----------------
class RMSNormRescale(nn.Module):
    def __init__(self, eps=1e-6, target_min=0.01, target_max=20.0):
        super().__init__()
        self.eps = eps
        self.target_min = target_min
        self.target_max = target_max

    def forward(self, x):
        # x shape: [C]
        rms = torch.sqrt(torch.mean(x**2, dim=0) + self.eps)
        normed = x / (rms + self.eps)   # scale around ~1
        # map to [0.01, 6.0]
        return self.target_min + (normed - normed.min()) / (normed.max() - normed.min() + self.eps) * (self.target_max - self.target_min)








# # ---------------- RMSNorm Rescale (per-channel) XXX NEW XXX----------------
# class RMSNormRescale(nn.Module):
#     def __init__(self, eps=1e-6, target_min=0.01, target_max=6.0, momentum=0.9):
#         super().__init__()
#         self.eps = eps
#         self.target_min = target_min
#         self.target_max = target_max
#         self.momentum = momentum

#         # running min/max for temporal smoothing
#         self.register_buffer("running_min", torch.tensor(float("inf")))
#         self.register_buffer("running_max", torch.tensor(float("-inf")))
#         self.initialized = False

#     def forward(self, x):
#         # x shape: [C]
#         rms = torch.sqrt(torch.mean(x**2, dim=0) + self.eps)
#         normed = x / (rms + self.eps)

#         batch_min, batch_max = normed.min(), normed.max()

#         if not self.initialized:
#             self.running_min = batch_min.detach()
#             self.running_max = batch_max.detach()
#             self.initialized = True
#         else:
#             self.running_min = (
#                 self.momentum * self.running_min + (1 - self.momentum) * batch_min.detach()
#             )
#             self.running_max = (
#                 self.momentum * self.running_max + (1 - self.momentum) * batch_max.detach()
#             )

#         scale = self.running_max - self.running_min + self.eps
#         normed = (normed - self.running_min) / scale
#         normed = torch.clamp(normed, 0.0, 1.0)

#         return self.target_min + normed * (self.target_max - self.target_min)












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

        # self.clamp_min = float(target_min)
        # self.clamp_max = float(target_max)

        # # 🔧 per-channel trainable RAW params((shape [C], not [1]))
        # self.gamma1_raw = nn.Parameter(torch.full((self.C,), gamma1_init))
        # self.freq_raw   = nn.Parameter(torch.full((self.C,), freq_init))



        # 🔧 per-channel trainable RAW params (direct init)
        self.gamma1_raw = nn.Parameter(torch.full((self.C,), gamma1_init))
        self.freq_raw   = nn.Parameter(torch.full((self.C,), freq_init))

        # effective mapping (still bounded for stability): [0.01, 6.0]
        self.gamma1_eff      = lambda: 0.01 + (20.0 - 0.01) * torch.sigmoid(self.gamma1_raw)
        self.freq_factor_eff = lambda: 0.01 + (20.0 - 0.01) * torch.sigmoid(self.freq_raw)




        # normalizers (all rescaled to [0.01, 6.0])
        self.norm_feature = RMSNormRescale()
        self.norm_freqmag = RMSNormRescale()
        self.norm_gamma1 = RMSNormRescale()
        self.norm_freq = RMSNormRescale()



        # 🔄 Buffers            
        self.register_buffer('phi', torch.tensor(phi))  # 🔒 Fixed buffer phi (No gradient updates)
        self.register_buffer("activation_history", torch.zeros(self.history_len, self.C))  # 🟡 history buffer
        self._step = 0


    # 🔍 non-monotonic decay schedule (unchanged)
    def sigmoid_blended_decay(self, epoch, t1=50, t2=70, k1=0.25, k2=0.5, a=0.980, b=0.995, c=0.9801):
        """📉 Sigmoid-based phase-decay transition"""
        s1 = 1 / (1 + math.exp(-k1 * (epoch - t1)))
        s2 = 1 / (1 + math.exp(-k2 * (epoch - t2)))
        return a * (1 - s1) + b * s1 * (1 - s2) + c * s2







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
        x_cur_stat = x.mean(dim=red_dims)                      # [C]

        # ======= feature_norm = current batch ======= 
        feature_norm = self.norm_feature(x_cur_stat.abs())




        # ======= freq_magnitude (tiered warmup) =======
        if self._step < self.history_len:
            # 🔵 Warmup phase → just use feature_norm (FFT not stable yet)
            freq_magnitude = feature_norm
        else:
            # 🟢 Full buffer → FFT on complete history
            h = self.activation_history
            freq_response  = torch.fft.fft(h, dim=0)
            freq_mag_hist  = torch.abs(freq_response).mean(dim=0)
            freq_magnitude = self.norm_freqmag(freq_mag_hist)

            # if epoch in {17, 20} and self._step % 1500 == 0:  # periodic check
            #     print(f"[Epoch {epoch} | Step {self._step}] Full-FFT → using {self.history_len} rows | "
            #         f"freq_mag_hist: shape={freq_mag_hist.shape}, min={freq_mag_hist.min().item():.4f}, "
            #         f"max={freq_mag_hist.max().item():.4f} | "
            #         f"freq_magnitude: min={freq_magnitude.min().item():.4f}, "
            #         f"max={freq_magnitude.max().item():.4f}")

            if epoch <= 2 and self._step in {20, 100, 1000, 4000}:  # sample some steps
                print(
                    f"[Epoch {epoch} | Step {self._step}] Full-FFT → FFT on {self._step} rows | "
                    f"freq_mag_hist: shape={freq_mag_hist.shape}, "
                    f"min={freq_mag_hist.min().item():.4f}, "
                    f"max={freq_mag_hist.max().item():.4f}, "
                    f"mean={freq_mag_hist.mean().item():.4f} | "
                    f"freq_magnitude: min={freq_magnitude.min().item():.4f}, "
                    f"max={freq_magnitude.max().item():.4f}, "
                    f"mean={freq_magnitude.mean().item():.4f}"
                )








        # ======= normalize trainable params =======
        gamma1_b = self.gamma1_eff()
        freq_b   = self.freq_factor_eff()


        # 🔍 Debug: print stats for investigation (only early epochs/steps to avoid spam)
        if epoch <= 2 and self._step in {20, 100, 1000, 4000}:
            print(
                f"[Epoch {epoch} | Step {self._step}] "
                f"gamma1_b: shape={gamma1_b.shape}, "
                f"min={gamma1_b.min().item():.4f}, max={gamma1_b.max().item():.4f}, mean={gamma1_b.mean().item():.4f} | "
                f"freq_b: min={freq_b.min().item():.4f}, max={freq_b.max().item():.4f}, mean={freq_b.mean().item():.4f}"
            )




        # ======= reshape for broadcasting❗️=======
        if layout == "NCHW":
            gamma1_b       = gamma1_b.view(1, self.C, 1, 1)
            freq_b         = freq_b.view(1, self.C, 1, 1)
            freq_magnitude = freq_magnitude.view(1, self.C, 1, 1)
        elif layout == "NHWC":
            gamma1_b       = gamma1_b.view(1, 1, 1, self.C)
            freq_b         = freq_b.view(1, 1, 1, self.C)
            freq_magnitude = freq_magnitude.view(1, 1, 1, self.C)
        else:  # "NC"
            gamma1_b       = gamma1_b.view(1, self.C)
            freq_b         = freq_b.view(1, self.C)
            freq_magnitude = freq_magnitude.view(1, self.C)




        # 🔄 gated activation
        gate = torch.sigmoid(gamma1_b * x - freq_b * freq_magnitude)


        # 🧠 sinusoidal perturbation 
        w_floor, w_max, T = 0.02, 0.05, 40   # floor, start, fade length
        t = min(epoch / T, 1.0)
        w = w_floor + 0.5 * (w_max - w_floor) * (1 + math.cos(math.pi * t))
        

        activation = x * gate + w * torch.sin(self.phi * x)


        # 🔁 Optional skip connection
        if self.use_skip:
            activation = 0.95 * activation + 0.05 * x

        return activation










# %%






