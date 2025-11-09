
# %%





#####-------- NOTE 1. ACTIVATION FUCNTION NOTE ----------------------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
############################## ACTIVATION FUCNTION 4  ####################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-----------------------XXX ACTIVATION FUCNTION   XXX------------------------------------------------------------#####

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class MYActivation4(nn.Module):
    def __init__(self, gamma1=1.5, gamma2=0.5, skew=0.5, phi=0.1, history_len=10):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma1 = nn.Parameter(torch.tensor(gamma1, dtype=torch.float32, device=self.device))
        self.gamma2 = nn.Parameter(torch.tensor(gamma2, dtype=torch.float32, device=self.device))
        self.skew = nn.Parameter(torch.tensor(skew, dtype=torch.float32, device=self.device))
        self.phi = nn.Parameter(torch.tensor(phi, dtype=torch.float32, device=self.device))

        self.history_len = history_len
        self.register_buffer("activation_history", torch.zeros(history_len, device=self.device))

    def update_history(self, x):
        x = x.mean().detach().to(self.device)
        self.activation_history = torch.cat([self.activation_history[1:], x.unsqueeze(0)])

    def forward(self, x):
        x = x.to(self.device)
        self.update_history(x)

        freq_response = torch.fft.fft(self.activation_history)
        freq_magnitude = torch.abs(freq_response).mean()

        gate = torch.sigmoid(self.gamma1 * x - self.gamma2 * freq_magnitude - self.skew)
        activation = x * gate + torch.sin(self.phi * x)
        return activation

# Initialize and manually feed history
activation = MYActivation4()

x = torch.linspace(-10, 10, 100)
y = []

for xi in x:
    yi = activation(xi.unsqueeze(0))  # Feed 1 input at a time to mimic batch sequence
    y.append(yi.item())

# Plot
plt.plot(x.numpy(), y)
plt.grid(True)
plt.title('MYActivation4 (History-Simulated)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# %%