




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

class MYActivation4(nn.Module):
    def __init__(self, gamma1=1.5, phi=0.1, history_len=10, enable_history=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Single Trainable parameter (Improved stability)
        self.gamma1 = nn.Parameter(torch.tensor(gamma1, device=self.device))

        # Fixed buffer phi (No gradient updates)
        self.register_buffer('phi', torch.tensor(phi, device=self.device))

        self.history_len = history_len
        self.activation_history_initialized = False
        self.enable_history = enable_history

    def update_history(self, x):
        if not self.enable_history:
            return
        
        if x.dim() == 4:
            x = x.mean(dim=(0, 2, 3))
        elif x.dim() == 2:
            x = x.mean(dim=0)
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")

        x = x.detach().to(self.device)

        if not self.activation_history_initialized:
            self.register_buffer("activation_history", torch.zeros(self.history_len, x.shape[0], device=self.device))
            self.activation_history_initialized = True

        self.activation_history = torch.cat([self.activation_history[1:], x.unsqueeze(0)])

    def decay_spectral_history(self, epoch):
        if self.enable_history and self.activation_history_initialized:
            decay_rate = 0.98 if epoch < 50 else 0.995
            with torch.no_grad():
                self.activation_history *= decay_rate

    def forward(self, x, epoch=0):
        x = x.to(self.device)
        self.saved_output = x.clone().detach()

        if self.enable_history:
            self.update_history(x)

        if self.enable_history and self.activation_history_initialized:
            freq_response = torch.fft.fft(self.activation_history, dim=0)
            freq_magnitude = torch.abs(freq_response).mean(dim=0)
            freq_magnitude = torch.clamp(freq_magnitude, min=0.05, max=1.5)
            smoothing_factor = max(0.1, 1 / (epoch + 10))
            freq_magnitude = (1 - smoothing_factor) * freq_magnitude + smoothing_factor * freq_magnitude.mean()
            freq_magnitude = freq_magnitude.view(1, -1, 1, 1)
        else:
            freq_magnitude = torch.zeros_like(x)

        # Adaptive freq_factor for training epochs
        freq_factor = min(0.3 + 0.007 * epoch, 0.8)
        gate = torch.sigmoid(self.gamma1 * x - freq_factor * freq_magnitude)

        activation = x * gate + 0.05 * torch.sin(self.phi * x)  # reduced oscillation
        activation = 0.95 * activation + 0.05 * x  # mild gradient-friendly skip connection

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
            act_fn = MYActivation4(gamma1=gamma1, phi=phi, enable_history=False).to(device)  # ✅ Disable history
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