






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

class MYActivation3(nn.Module):
    def __init__(self, gamma1=1.5, phi=0.1, history_len=10):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep only two trainable parameters
        self.gamma1 = nn.Parameter(torch.tensor(gamma1, dtype=torch.float32, device=self.device))
        self.phi = nn.Parameter(torch.tensor(phi, dtype=torch.float32, device=self.device))

        self.history_len = history_len
        self.activation_history_initialized = False
        self.beta = 0.85  # 🔹 Momentum factor for smoothing

    def update_history(self, x):
        """ Update spectral activation history """
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

    def decay_spectral_history(self):
        """ Apply spectral decay to prevent overfitting to high-frequency components """
        with torch.no_grad():
            self.activation_history *= 0.99  # Gradual memory decay to avoid spectral locking

    def forward(self, x, epoch=0):
        x = x.to(self.device)
        self.saved_output = x.clone().detach()
        self.update_history(x)

        # Compute FFT-based frequency magnitude (only first half of frequencies)
        freq_response = torch.fft.fft(self.activation_history, dim=0)
        freq_magnitude = torch.abs(freq_response).mean(dim=0)

        # 🔹 Apply Soft Spectral Clamping
        freq_magnitude = torch.clamp(freq_magnitude, min=0.05, max=1.5)  

        # 🔹 Apply More Adaptive Smoothing
        smoothing_factor = max(0.1, 1 / (epoch + 10))  
        freq_magnitude = (1 - smoothing_factor) * freq_magnitude + smoothing_factor * freq_magnitude.mean()

        # Reshape for broadcasting
        freq_magnitude = freq_magnitude.view(1, -1, 1, 1)

        # 🔹 New Soft-Gated Activation (Stability Improved)
        gate = torch.tanh(0.7 * self.gamma1 * x - 0.8 * freq_magnitude)

        # 🔹 Adjust Adaptive Noise Injection (Reduce After Epoch 50)
        if self.training:
            noise_scale = 0.02 if epoch < 50 else 0.005  
            noise = torch.randn_like(x) * torch.clamp(noise_scale * (1 - self.gamma1.detach()), min=1e-6)
            x = x + noise

        # 🔹 Compute Final Activation
        activation = x * gate + torch.sin(0.5 * self.phi * torch.clamp(x, min=-1.5, max=1.5))  

        return activation
