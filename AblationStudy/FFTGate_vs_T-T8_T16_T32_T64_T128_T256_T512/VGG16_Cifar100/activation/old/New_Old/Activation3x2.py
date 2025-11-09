




# %% 

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


        # ✅ 📍 Strategy 1: Dynamic Frequency Response Adaptation
        if 30 <= epoch < 75:
            freq_scale = 1.0 + 0.01 * (epoch - 30)  # Gradually increasing frequency sensitivity
        else:
            freq_scale = 1.0

        # 🔹 Improved Non-Linearity and Spectral Influence with dynamic freq_scale
        gate = torch.sigmoid(self.gamma1 * x - freq_scale * freq_magnitude)

        # 🔹 Frequency-Aware Activation (FFT + Exp decay for stability)
        activation = x * gate + torch.sin(self.phi * x) * torch.exp(-0.5 * torch.abs(x))

        return activation
# ✅ Updated Plot Function for MYActivation3
def plot_activation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 Generate input values as [Batch, Features]
    x = torch.linspace(-5, 5, 400, device=device).view(1, -1)  # Shape: [1, 400]

    fig, ax = plt.subplots(figsize=(8, 6))

    # 🔹 Test different values of Gamma1 and Phi
    param_sets = [(1.5, 0.1), (2.0, 0.2), (3.0, 0.5)]  

    with torch.no_grad():  # ✅ Disable gradient tracking
        for gamma1, phi in param_sets:
            act_fn = MYActivation3(gamma1=gamma1, phi=phi, enable_history=False).to(device)  # ✅ Disable history
            y = act_fn(x).cpu().detach().numpy().squeeze()  # ✅ Ensure proper shape
            ax.plot(x.cpu().numpy().flatten(), y, label=f"Gamma1={gamma1:.1f}, Phi={phi:.1f}")

    # 🔹 Improve visualization
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("MYActivation3 Function Visualization", fontsize=14)
    ax.set_xlabel("Input (x)", fontsize=12)
    ax.set_ylabel("Activation Output", fontsize=12)
    ax.legend()
    
    plt.show()

# 🔥 Run the updated plot function
plot_activation()



# %%
