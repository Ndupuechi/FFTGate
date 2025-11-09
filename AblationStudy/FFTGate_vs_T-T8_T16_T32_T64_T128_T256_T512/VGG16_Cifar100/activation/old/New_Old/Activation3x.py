





#####-------- NOTE 1. ACTIVATION FUCNTION NOTE ----------------------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
############################## ACTIVATION FUCNTION 3  ####################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-----------------------XXX ACTIVATION FUCNTION   XXX------------------------------------------------------------#####

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.fft

class MYActivation3(nn.Module):
    def __init__(self, gamma1=1.5, gamma2=0.5, skew=0.5, phi=0.1, history_len=10, device=None):
        super().__init__()
        self.gamma1 = nn.Parameter(torch.tensor(gamma1, dtype=torch.float32))
        self.gamma2 = nn.Parameter(torch.tensor(gamma2, dtype=torch.float32))
        self.skew = nn.Parameter(torch.tensor(skew, dtype=torch.float32))
        self.phi = nn.Parameter(torch.tensor(phi, dtype=torch.float32))

        self.history_len = history_len

        # Automatically detect device or use specified one
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer("activation_history", torch.zeros(history_len, device=self.device))

    def update_history(self, x):
        if x.dim() == 4:
            x = x.mean(dim=(0, 2, 3))  # Average over batch and spatial dims
        elif x.dim() == 2:
            x = x.mean(dim=0)  # Average over batch

        x = x.detach().to(self.device)
        self.activation_history = torch.cat([self.activation_history[1:], x.unsqueeze(0)])

    def forward(self, x):
        x = x.to(self.device)
        self.update_history(x)

        freq_response = torch.fft.fft(self.activation_history)
        freq_magnitude = torch.abs(freq_response).mean()

        gate = torch.sigmoid(self.gamma1 * x - self.gamma2 * freq_magnitude - self.skew)
        activation = x * gate + torch.sin(self.phi * x)

        return activation









# ####XXX create custom dataset XXX########        

# # Test the optimized activation function
# x = torch.linspace(-10, 10, 100)
# activation = MYActivation3()
# y = activation(x).detach()

# # Plot the function to inspect its shape
# plt.plot(x.numpy(), y.numpy())
# plt.grid(True)
# plt.title('MY2 Activation')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


# # %%




# #####-------- NOTE 1. ACTIVATION FUCNTION NOTE ----------------------------------------------------------------------#####
# ##########################################################################################################################
# ######################|--------------------------------------------------------------|####################################
# ############################## ACTIVATION FUCNTION 4  ####################################################################
# ######################|--------------------------------------------------------------|####################################
# ##########################################################################################################################
# #####-----------------------XXX ACTIVATION FUCNTION   XXX------------------------------------------------------------#####


# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn

# class MYActivation3(nn.Module):
#     def __init__(self, gamma1=1.5, gamma2=0.5, skew=0.5, phi=0.1, history_len=10, device=None):
#         super().__init__()
#         self.gamma1 = nn.Parameter(torch.tensor(gamma1, dtype=torch.float32))
#         self.gamma2 = nn.Parameter(torch.tensor(gamma2, dtype=torch.float32))
#         self.skew = nn.Parameter(torch.tensor(skew, dtype=torch.float32))
#         self.phi = nn.Parameter(torch.tensor(phi, dtype=torch.float32))
#         self.history_len = history_len

#         # Automatically detect device or use specified one
#         self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.register_buffer("activation_history", torch.zeros(history_len, device=self.device))

#         # Move parameters to device
#         self.to(self.device)

#     def update_history(self, x):
#         x = x.mean().detach().to(self.device)  # Simple aggregation as in your original code
#         self.activation_history = torch.cat([self.activation_history[1:], x.unsqueeze(0)])

#     def forward(self, x):
#         x = x.to(self.device)
#         self.update_history(x)

#         freq_response = torch.fft.fft(self.activation_history)
#         freq_magnitude = torch.abs(freq_response).mean()

#         gate = torch.sigmoid(self.gamma1 * x - self.gamma2 * freq_magnitude - self.skew)
#         activation = x * gate + torch.sin(self.phi * x)
#         return activation

# # Initialize and manually feed history
# activation = MYActivation3()

# x = torch.linspace(-10, 10, 100)
# y = []

# for xi in x:
#     yi = activation(xi.unsqueeze(0))  # Feed 1 input at a time to mimic batch sequence
#     y.append(yi.item())

# # Plot
# plt.plot(x.numpy(), y)
# plt.grid(True)
# plt.title('MYActivation3 (History-Simulated)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()




# # %%





