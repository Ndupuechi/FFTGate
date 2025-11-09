






#####-------- NOTE 1. ACTIVATION FUCNTION NOTE ----------------------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
############################## ACTIVATION FUCNTION 3  ####################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-----------------------XXX ACTIVATION FUCNTION   XXX------------------------------------------------------------#####


import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt

class MYActivation3(nn.Module):
    def __init__(self, gamma1=1.5, phi=0.1, history_len=10, enable_history=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Trainable parameters
        self.gamma1 = nn.Parameter(torch.tensor(gamma1, dtype=torch.float32, device=self.device))
        self.phi = nn.Parameter(torch.tensor(phi, dtype=torch.float32, device=self.device))

        self.history_len = history_len
        self.activation_history_initialized = False
        self.beta = 0.85  # Momentum factor for smoothing
        self.enable_history = enable_history  # ✅ Allows history updates to be disabled for plotting

    def update_history(self, x):
        """ Update spectral activation history """
        if not self.enable_history:  # ✅ Skip history update if disabled (for plotting)
            return
        
        if x.dim() == 4:
            x = x.mean(dim=(0, 2, 3))  # Average over batch & spatial dims -> [channels]
        elif x.dim() == 2:
            x = x.mean(dim=0)  # Average over batch -> [features]
        else:
            raise ValueError(f"Unexpected input shape {x.shape}, expected 2D or 4D.")

        x = x.detach().to(self.device)

        if not self.activation_history_initialized:
            self.register_buffer("activation_history", torch.zeros(self.history_len, x.shape[0], device=self.device))
            self.activation_history_initialized = True

        if x.shape[0] != self.activation_history.shape[1]:
            raise RuntimeError(f"Feature count mismatch: got {x.shape[0]} features, expected {self.activation_history.shape[1]} features")

        self.activation_history = torch.cat([self.activation_history[1:], x.unsqueeze(0)])



    def decay_spectral_history(self, epoch):
        """ ✅ Adaptive decay prevents overfitting while allowing learning """
        if self.enable_history and self.activation_history_initialized:
            with torch.no_grad():
                if epoch < 40:
                    self.activation_history *= 0.98  # 🔹 Stronger decay early for faster adaptation
                elif epoch < 70:
                    self.activation_history *= 0.99  # 🔹 Balanced decay
                else:
                    self.activation_history *= 0.995  # 🔹 Slow decay in later training




    def forward(self, x, epoch=0):
        x = x.to(self.device)
        self.saved_output = x.clone().detach()

        # ✅ Only update history during normal training (not plotting)
        if self.enable_history:
            self.update_history(x)

        # 🔹 Compute FFT-based frequency magnitude (only when history is enabled)
        if self.enable_history and self.activation_history_initialized:
            freq_response = torch.fft.fft(self.activation_history, dim=0)
            freq_magnitude = torch.abs(freq_response).mean(dim=0)
            freq_magnitude = torch.clamp(freq_magnitude, min=0.05, max=1.5)
            smoothing_factor = max(0.1, 1 / (epoch + 10))
            freq_magnitude = (1 - smoothing_factor) * freq_magnitude + smoothing_factor * freq_magnitude.mean()
            freq_magnitude = freq_magnitude.view(1, -1, 1, 1)
        else:
            freq_magnitude = torch.zeros_like(x)

        # 🔹 Improved Non-Linearity and Spectral Influence
        gate = torch.sigmoid(self.gamma1 * x - 0.8 * freq_magnitude)

        # 🔹 Frequency-Aware Activation (FFT + Exp decay for stability)
        activation = x * gate + torch.sin(self.phi * x) * torch.exp(-0.5 * torch.abs(x))

        return activation
