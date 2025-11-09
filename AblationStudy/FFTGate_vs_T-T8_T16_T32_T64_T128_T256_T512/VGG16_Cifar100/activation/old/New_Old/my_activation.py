








#####-------------------------------- NOTE I: LOAD LIBRARIES ENSURING REPRODUCIBILITY NOTE ---------------------------------------------------------------------------------------------------#####
###################################################################################################################################################################################################
#####-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#####
################################# Python code by Emeka Ndupuechi ##################################################################################################################################
###################################################################################################################################################################################################
############################################# 2025 ################################################################################################################################################
#####-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#####
###################################################################################################################################################################################################
#####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#####
#####-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#####




#####-------- NOTE I.A. LOAD REQUIRED LIBRARIES NOTE ----------------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
########################################## REQUIRED LIBRARIES ############################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####------------------------------- XXX LOAD REQUIRED LIBRARIES XXX-------------------------------------------------#####

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import random_split
import numpy as np




# pylint: disable=import-error
# from activation.class_entropy_log import log_per_class_entropy  # type: ignore # ✅ Import correctly








#####-------- NOTE 00. CHECK GPU CONNECTION NOTE --------------------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
############################## CHECK GPU CONNECTION  #####################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-----------------------XXX CHECK GPU CONNECTION  XXX------------------------------------------------------------#####




print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"Current Device: {torch.cuda.current_device()}")
print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")







import torch
import torch.nn as nn

class MYActivation2(nn.Module):
    def __init__(self, beta_positive=1.0, beta_negative=0.2, gamma1=1.5, gamma2=0.5, temperature=1.0, skew=0.5, epsilon=0.01):
        super(MYActivation2, self).__init__()

        # ✅ Track device explicitly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ Trainable Parameters (Moved to Correct Device)
        self.beta_positive = nn.Parameter(torch.tensor(beta_positive, dtype=torch.float32, device=self.device))
        self.beta_negative = nn.Parameter(torch.tensor(beta_negative, dtype=torch.float32, device=self.device))
        self.gamma1 = nn.Parameter(torch.tensor(gamma1, dtype=torch.float32, device=self.device))
        self.gamma2 = nn.Parameter(torch.tensor(gamma2, dtype=torch.float32, device=self.device))
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32, device=self.device))
        self.skew = nn.Parameter(torch.tensor(skew, dtype=torch.float32, device=self.device))  # ✅ Skew Factor

        # ✅ Register epsilon as a fixed buffer (Not trainable but moves with model)
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32, device=self.device))


        # ✅ **Clamp Initial Values to Ensure Proper Ranges**
        with torch.no_grad():
            self.beta_positive.data.clamp_(0.1, 10.0)
            self.beta_negative.data.clamp_(0.001, 5.0)
            self.gamma1.data.clamp_(0.1, 10.0)
            self.gamma2.data.clamp_(0.001, 5.0)
            self.temperature.data.clamp_(0.1, 10.0)
            self.skew.data.clamp_(-5.0, 5.0)  # ✅ Keeps skew factor in range


        # 🔴🔴🔴 Track last activations for entropy regularization
        self.last_activations = None



    def gaussian_cdf(self, x, skew_factor):
        """Computes Skewed Gaussian CDF using erf function."""
        return 0.5 * (1.0 + torch.erf((x + skew_factor) / (torch.sqrt(torch.tensor(2.0, dtype=torch.float32, device=self.device)) * self.temperature)))

    # def forward(self, x, epoch=None, train_accuracy=None):
    def forward(self, x, epoch=None, train_accuracy=None, targets=None):  # 🔴🔴🔴 Add `targets` argument to capture labels (needed for per-class entropy)
    
        """
        Ensure epoch and train_accuracy are never None before using them.
        """
        if epoch is None:
            epoch = 0  # Ensure epoch has a valid value
        if train_accuracy is None:
            train_accuracy = 0.0  # Ensure validation accuracy has a valid value

        # ✅ Ensure input `x` is on the same device
        x = x.to(self.device)



        # 🔴🔴🔴 Store activations + targets for per-class entropy logging
        self.last_activations = x.clone().detach()
        # self.last_targets = targets.clone().detach() if targets is not None else None  # 🔴🔴🔴 Save targets if passed



        # ✅ Prevent Multiple Prints Per Epoch
        if "printed_epochs" not in globals():
            globals()["printed_epochs"] = set()
        if epoch not in globals()["printed_epochs"]:
            # print(f"🚀 [DEBUG] Epoch {epoch}: Train Accuracy Passed to Activation = {train_accuracy:.2f}%")
            globals()["printed_epochs"].add(epoch)

        batch_mean = torch.mean(x)
        batch_std = torch.std(x) + 1e-6  








        # 🔴🔴🔴 Compute per-channel statistics correctly (handles both conv & linear)

        if x.dim() == 2:
            # Fully connected (final layer input) — no per-channel spatial dims
            per_channel_mean = x.mean(dim=0, keepdim=True)
            per_channel_std = x.std(dim=0, keepdim=True)
            per_channel_skew = ((x - per_channel_mean)**3).mean(dim=0) / (per_channel_std**3 + 1e-6)
            per_channel_skew = per_channel_skew.mean()  # Take average skew across features
        else:
            # Standard Conv feature map case
            per_channel_mean = x.mean(dim=[0, 2, 3], keepdim=True)
            per_channel_std = x.std(dim=[0, 2, 3], keepdim=True)
            per_channel_skew = ((x - per_channel_mean)**3).mean(dim=[0, 2, 3]) / (per_channel_std**3 + 1e-6)
            per_channel_skew = per_channel_skew.mean()  # Average skew across channels



        # 🔴🔴🔴 Batch-wise adaptive beta (this already exists)
        adaptive_beta_positive = self.beta_positive * (1 + 0.1 * per_channel_skew.abs().mean())
        adaptive_beta_negative = self.beta_negative * (1 + 0.05 * per_channel_skew.abs().mean())








        # ✅ **Clamp Parameters Before Computation to Ensure Stability**
        with torch.no_grad():
            self.beta_positive.data.clamp_(0.1, 10.0)
            self.beta_negative.data.clamp_(0.001, 5.0)
            self.gamma1.data.clamp_(0.1, 10.0)
            self.gamma2.data.clamp_(0.001, 5.0)
            self.temperature.data.clamp_(0.1, 10.0)
            self.skew.data.clamp_(-5.0, 5.0)  # ✅ Keeps skew factor in range



        # ✅ Adaptive Temperature Scaling
        temperature_scaled = torch.clamp(
            self.temperature * (1.0 + 0.02 * torch.sigmoid(self.temperature) * batch_std),
            0.1,
            10.0
        )

        # ✅ Adaptive Skew Scaling (Dynamic Skew for Positive and Negative Values)
        skew_factor_positive = torch.sigmoid(self.skew) * 2.0  # Scale to [0,2] for stronger effect
        skew_factor_negative = torch.sigmoid(-self.skew) * 2.0  # Inverse effect for negatives

        # ✅ Asymmetric Scaling for Positive/Negative Inputs
        gamma1_scaled = torch.clamp(self.gamma1 / (1.0 + 0.05 * batch_std), min=0.1) * (1 + skew_factor_positive)
        gamma2_scaled = torch.clamp(self.gamma2 / (1.0 + 0.05 * batch_std), min=0.002) * (1 + skew_factor_negative)



        # ✅ Gaussian Mixture for Flexible Activation Response
        phi1 = self.gaussian_cdf((adaptive_beta_positive * (x - batch_mean) / temperature_scaled), skew_factor_positive)
        phi2 = self.gaussian_cdf((1.5 * adaptive_beta_positive * (x - batch_mean) / temperature_scaled), skew_factor_positive)


        # ✅ Positive Activation: Skewed Gaussian Mixture
        positive = gamma1_scaled * (x - batch_mean) * (0.5 * phi1 + 0.5 * phi2) * (x > 0).float()

        # ✅ Negative Activation: Skewed Gaussian Mixture + Exponential
        phi_neg1 = self.gaussian_cdf((-adaptive_beta_negative * (x - batch_mean) / temperature_scaled), skew_factor_negative)
        phi_neg2 = self.gaussian_cdf((-2.0 * adaptive_beta_negative * (x - batch_mean) / temperature_scaled), skew_factor_negative)



        negative = (
            gamma2_scaled * (x - batch_mean) * (0.5 * phi_neg1 + 0.5 * phi_neg2) 
            + self.epsilon * torch.exp(-torch.abs(x))
        ) * (x <= 0).float()



        out = positive + negative



        return out








############--------------------------------------------------------################








