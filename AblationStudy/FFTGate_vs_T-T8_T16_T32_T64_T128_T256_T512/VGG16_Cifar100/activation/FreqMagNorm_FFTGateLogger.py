












# %%  

#####----------------------------- NOTE FFTGATE LIBRARIES NOTE ------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
#################################### FFTGATE LIBRARIES ###################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####---------------------------XXX ACTIVATION FUCNTION   XXX--------------------------------------------------------#####





# ✅ Import libraries
import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import os





# ==============================================================================================================================
# ==============================================================================================================================
# 🔗=================================🔑 FFTGATE ACTIVATION CUSTOM LIBRARIES🔑================================================🔗
# ==============================================================================================================================
# ==============================================================================================================================




# ===============================================================
# 🔍 Helper 1: -- Normalizer for FFT magnitudes (layer-based) --
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
# 🔍 Helper 2: ---- logger for FFTGate statistics --------------
# ===============================================================

class GateLogger(nn.Module):
    """
    Lightweight logger for FFTGate statistics.
    Logs whatever tensor it receives (raw or normalized),
    based on where it is called in forward().
    """

    def __init__(self, gate_log_path, gate_history, label="RAW"):
        super().__init__()
        self.gate_log_path = gate_log_path
        self.gate_history = gate_history
        self.label = label  # e.g., "RAW" or "NORM"

        # Predefined debug checkpoints
        self.milestone_epochs = {0, 1, 3, 5, 10, 20, 30, 50, 80, 95}
        self.detailed_steps   = {0, 1, 2, 5, 10, 50, 70, 100, 200, 500, 700, 750}

    def forward(self, tensor, epoch, epoch_step, rolling_step):
        """Logs basic stats for the given tensor if at milestone points."""
        log_line = None
        if (epoch in self.milestone_epochs) and (epoch_step in self.detailed_steps):
            with torch.no_grad():
                mean_val = tensor.mean().item()
                std_val  = tensor.std().item()
                min_val  = tensor.min().item()
                max_val  = tensor.max().item()

                log_line = (
                    f"[FFTGate Debug] Epoch {epoch:>3d} || Step {epoch_step:>4d} || Rolling {rolling_step:>4d} "
                    f"|| {self.label} ==> || Mean={mean_val:+.3f} | Std={std_val:+.3f} | "
                    f"Min={min_val:+.3f} | Max={max_val:+.3f}\n"
                )
                self.gate_history.append(log_line)

        # Clear old logs only once
        if epoch == 0 and rolling_step == 1:
            os.makedirs(os.path.dirname(self.gate_log_path["gate_history"]), exist_ok=True)
            open(self.gate_log_path["gate_history"], "w").close()

        # Append new line
        if log_line is not None:
            with open(self.gate_log_path["gate_history"], "a", encoding="utf-8") as f:
                f.write(log_line)

        return tensor  # Return unchanged







# %%
