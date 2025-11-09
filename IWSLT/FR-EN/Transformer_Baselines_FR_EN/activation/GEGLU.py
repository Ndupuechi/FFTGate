




# %%  

#####---------------------- NOTE BASELINE ACTIVATION FUNCTION NOTE --------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
#################################### GEGLU  ##############################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-----------------------XXX ACTIVATION FUCNTION   XXX------------------------------------------------------------#####




########################################################################################################################
####-------| NOTE 1. IMPORTS LIBRARIES | XXX ------------------------------------------------------#####################
########################################################################################################################

# ✅ Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


########################################################################################################################
####-------| NOTE 1. DEFINE GEGLU ACTIVATION CLASS | XXX -------------------------------------------####################
########################################################################################################################


# ===============================================================
# 🔗==================== GEGLU Activation 🔑================🔗
# ===============================================================
# GEGLU (Gated Exponential Linear Unit)
# Reference: "GLU Variants Improve Transformer" – Noam Shazeer, 2020
# ---------------------------------------------------------------
# Formula:
#   Split input along last dimension into (x1, x2)
#   Output = x1 * GELU(x2)
# ---------------------------------------------------------------
# ⚠️ In Fairseq, the feedforward input dim (e.g., 2048) is not doubled.
# So we project internally to 2×dim before splitting.
# This version safely falls back to gated GELU when dim is not divisible by 2.
# ===============================================================



class GEGLU(nn.Module):
    def __init__(self, dim_in=None, dim_hidden=None):
        super().__init__()
        self.gelu = nn.GELU()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.linear = None
        self._warned = False

    def forward(self, x):
        # 🔹 Initialize projection lazily once we know the input dimension
        if self.linear is None:
            in_dim = x.size(-1)
            out_dim = in_dim * 2  # double for GEGLU split
            self.linear = nn.Linear(in_dim, out_dim, bias=True).to(x.device)
            print(f"⚙️ [GEGLU] Created projection layer: {in_dim} → {out_dim}")

        # 🔹 Project to 2×dim
        x_proj = self.linear(x)

        # 🔹 Split into halves
        x1, x2 = x_proj.chunk(2, dim=-1)

        # 🔹 Apply GELU gating
        gated = self.gelu(x2)
        out = x1 * gated

        return out



# %%






