






#####-------- NOTE 1. ACTIVATION FUCNTION NOTE ----------------------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
############################## ACTIVATION FUCNTION   #####################################################################
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


# ✅ FFTGATE activation
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


    def __init__(self, gamma1=1.5, phi=0.1, history_len=15,
                 enable_history=True, use_skip=False, decay_mode="exp"):    
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # 🔧 Trainable scaling factor
        # self.gamma1 = nn.Parameter(torch.tensor(gamma1, device=self.device))
        self.gamma1 = nn.Parameter(torch.full((1,), gamma1, device=self.device))

        # 🔒 Fixed buffer phi (No gradient updates)
        self.register_buffer('phi', torch.tensor(phi, device=self.device))

        self.history_len = history_len
        self.enable_history = enable_history
        self.activation_history_initialized = False

        # ⚙️ Optional skip connection toggle
        self.use_skip = use_skip

        # ✅ Save selected decay strategy
        self.decay_mode = decay_mode

    
    # 🔍 Computes sigmoid-blended multi-phase decay rate used for non-monotonic history scaling | non-monotonic decay stratey defination
    def sigmoid_blended_decay(self, epoch, t1=50, t2=70, k1=0.25, k2=0.5,
                              a=0.980, b=0.995, c=0.9801):
        """📉 Sigmoid-based phase-decay transition"""
        s1 = 1 / (1 + math.exp(-k1 * (epoch - t1)))
        s2 = 1 / (1 + math.exp(-k2 * (epoch - t2)))
        decay = a * (1 - s1) + b * s1 * (1 - s2) + c * s2
        return decay


    # 🔍 Tracks moving average of pre-activation inputs over a history window (per channel)
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

        x = x.detach().to(self.device)


        # 🔁 Reset buffer if shape changes | 🔥 Dynamically reset history buffer if shape changes
        if not self.activation_history_initialized or self.activation_history.shape[1] != x.shape[0]:
            self.register_buffer("activation_history", torch.zeros(self.history_len, x.shape[0], device=self.device))
            self.activation_history_initialized = True            

        self.activation_history = torch.cat([self.activation_history[1:], x.unsqueeze(0)])


    # 🔍 Decays the stored activation history based on selected temporal decay strategy 
    def decay_spectral_history(self, epoch, num_epochs=100):
        if not (self.enable_history and self.activation_history_initialized):
            return

        with torch.no_grad():
            if self.decay_mode == "linear":                     # ✅ Use Non-Monotonic decay Strategy    
                decay_rate = self.sigmoid_blended_decay(epoch)
                self.activation_history *= decay_rate

            elif self.decay_mode == "exp":                      # ✅ Use Exponential decay Strategy 
                decay_factor = 0.99 + 0.01 * math.cos(math.pi * epoch / num_epochs)
                self.activation_history *= decay_factor

            else:
                raise ValueError(f"Unknown decay_mode: {self.decay_mode}. Use 'exp' or 'linear'.")


    def forward(self, x, epoch=0):
        x = x.to(self.device)
        self.saved_output = x.clone().detach()

        if self.enable_history:
            self.update_history(x)


        # 🔍 Applies FFT to decayed activation history and computes average magnitude
        if self.enable_history and self.activation_history_initialized:
            freq_response = torch.fft.fft(self.activation_history, dim=0)   
            freq_magnitude = torch.abs(freq_response).mean(dim=0)
            freq_magnitude = torch.clamp(freq_magnitude, min=0.05, max=1.5)

            # 🔧 Smoothing across channels to prevent spikes        
            smoothing_factor = max(0.1, 1 / (epoch + 10))
            freq_magnitude = (1 - smoothing_factor) * freq_magnitude + smoothing_factor * freq_magnitude.mean()
            freq_magnitude = freq_magnitude.view(1, -1, 1, 1)
        else:
            freq_magnitude = torch.zeros_like(x)


        # ✅ Clamp gamma1 internally for stability
        gamma1 = torch.clamp(self.gamma1, min=0.1, max=6.0)

        # 🔄 Gate using FFT-derived frequency magnitude
        freq_factor = min(0.3 + 0.007 * epoch, 0.8)
        gate = torch.sigmoid(gamma1 * x - freq_factor * freq_magnitude)

        # ✅ Main activation logic: gated + sinusoidal regularization
        activation = x * gate + 0.05 * torch.sin(self.phi * x)  


        # 🔁 Optional: smart residual skip blend
        if self.use_skip:
            activation = 0.95 * activation + 0.05 * x


        return activation
