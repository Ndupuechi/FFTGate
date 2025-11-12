










# %%  

#####---------------------- NOTE COMPLEXITY VS BLEU NOTE ------------------------------------------------------------#####
##########################################################################################################################
######################|--------------------------------------------------------------|####################################
#################################### Peak VRAM vs BLEU  ##################################################################
######################|--------------------------------------------------------------|####################################
##########################################################################################################################
#####-----------------------XXX COMPLEXITY VS BLEU  XXX--------------------------------------------------------------#####



####################################################################################################
#####---------------------- NOTE PEAK VRAM VS BLEU --------------------------------------------#####
####################################################################################################
# 🔗====================================🔑 COMPLEXITY VS BLEU 🔑==================================🔗
####################################################################################################


import matplotlib.pyplot as plt
import numpy as np
import os



# ✅ === Ensure correct working directory ===
import sys
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github4\Complexity_RunTime_Memory\Comparisons\Plot"
if os.getcwd() != Project_PATH:
    os.chdir(Project_PATH)
print(f"✅ Current working directory: {os.getcwd()}")


# ======================================================================================================
# ✅ === Standard libraries ===
# ======================================================================================================

import re
import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
import matplotlib.patheffects as path_effects

# ======================================================================================================
# ✅ === Custom parser ===
# ======================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Dataset information
exp_parser = argparse.ArgumentParser("ImageNet Experiment Config")
exp_parser.add_argument('--dataset_name', default="ConvNeXtV1_ATTO_ImageNet_1K_B256_LR0.004_AdamW_Seed1_2_INs32", type=str,
    help="Choose dataset direction: 'ImageNet")  

exp_parser.add_argument('--dataset_namefftgate', default="ConvNeXtV1_ATTO_ImageNet_1K_B256_LR0.004_AdamW_T12_Seed1_2_INs32", type=str,
    help="Choose dataset direction: 'ImageNet") 

# exp_parser.add_argument('--translation_direction', default="EN⟶FR", type=str)
exp_parser.add_argument('--translation_direction', default=r"FR$\rightarrow$EN", type=str)

# ✅ Global font settings
exp_parser.add_argument('--base_font_size', default=13, type=int)        # Default: 11   
exp_parser.add_argument('--spine_width', default=1.0, type=float)        # Default: 1.2
exp_parser.add_argument('--legend_title_font', default=12, type=int)     # Default: 10
exp_parser.add_argument('--legend_font', default=12, type=int)           # Default: 9

exp_args = exp_parser.parse_args([])   # ← for naming/logging
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ===============================================================
# 🔗============= GLOBAL FONT SETTINGs 🔑=====================🔗
# ===============================================================
# ─────────────────────────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({

    # === FONT SETTINGS ===
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage{bm}\boldmath",  # makes all LaTeX text bold

    # === Colors ===
    "text.color": "#000000",               # solid black
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "axes.edgecolor": "#000000",
    "axes.titlecolor": "#000000",


    "font.size": exp_args.base_font_size,
    "font.weight": "normal",
    "axes.titlesize": exp_args.base_font_size + 1,
    "axes.titleweight": "normal",
    "axes.labelsize": exp_args.base_font_size + 2,
    "axes.labelweight": "medium",
    "legend.fontsize": exp_args.base_font_size - 1,
    "legend.title_fontsize": exp_args.base_font_size,
    "xtick.labelsize": exp_args.base_font_size,
    "ytick.labelsize": exp_args.base_font_size,

    # === COLOR CONSISTENCY ===
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "axes.edgecolor": "black",


    # === AXES & SPINES ===
    "axes.linewidth": exp_args.spine_width,   # ✅ ensures ALL future figures use this spine width        
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.axisbelow": False,   # ensures lines/markers are above grid

    # === PDF / SVG EXPORT QUALITY ===
    "pdf.fonttype": 42,        # editable text in PDF
    "ps.fonttype": 42,         # editable text in PS
    "svg.fonttype": 'none',    # editable text in SVG
})

print(f"✅ Publication style applied: Bold fonts, black ticks, clean spines (base font size={exp_args.base_font_size} | width={exp_args.spine_width}).")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
 

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Data (from your table)
activations = ["ReLU", "GELU", "TanhExp", "GEGLU", "FFTGate (T=32)"]

# Peak VRAM (MB)
vram_enfr = np.array([2622.52, 2977.62, 3518.08, 4293.92, 3518.8])
vram_fren = np.array([2569.00, 2934.99, 3508.26, 4296.11, 3503.94])

# BLEU scores
bleu_enfr = np.array([36.66, 36.58, 36.35, 35.52, 37.54])
bleu_fren = np.array([36.32, 36.12, 35.77, 35.25, 37.40])

# ────────────────────────────────────────────────────────────────────────────────────────────────
# 🎨 Color palette
COLORS = {
    "ReLU":     "#8338EC",
    "GELU":     "#FFD60A",
    "TanhExp":  "#06D6A0",
    "GEGLU":    "#EF476F",
    "FFTGate":  "#071DE7C5",
}

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Create figure
fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

# Plot En→Fr
for i, act in enumerate(activations):
    color = COLORS["FFTGate"] if "FFTGate" in act else COLORS.get(act, "gray")
    ax.scatter(vram_enfr[i], bleu_enfr[i],
               color=color, edgecolor='black', s=75, zorder=3, marker='o')



    # # Center annotation directly above each marker (LaTeX bold)
    # label_text = act.replace(" (T=32)", "")
    # ax.annotate(rf"\textbf{{{label_text}}}",
    #             xy=(vram_enfr[i], bleu_enfr[i]),
    #             xytext=(0, 7),  # vertical offset
    #             textcoords="offset points",
    #             ha='center', va='bottom',
    #             fontsize=exp_args.base_font_size, color=color)  #10





    # Center annotation directly above each marker (LaTeX bold + black outline)
    label_text = act.replace(" (T=32)", "")
    text = ax.annotate(rf"\textbf{{{label_text}}}",
                    xy=(vram_enfr[i], bleu_enfr[i]),
                    xytext=(0, 7),  # vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=exp_args.base_font_size,
                    color=color)

    # Add a subtle black outline for better visibility
    text.set_path_effects([
        path_effects.Stroke(linewidth=0.5, foreground='black'),
        path_effects.Normal()
    ])





# Plot Fr→En
for i, act in enumerate(activations):
    color = COLORS["FFTGate"] if "FFTGate" in act else COLORS.get(act, "gray")
    ax.scatter(vram_fren[i], bleu_fren[i],
               color=color, edgecolor='black', s=75, zorder=3, marker='^')

# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Axis styling
ax.set_xlabel(r"\textbf{Peak VRAM (MB)}", labelpad=6)
ax.set_ylabel(r"\textbf{BLEU Score}", labelpad=6)
ax.grid(True, linestyle="--", alpha=0.35)


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Custom Axis Styling (Publication Style)
ax.set_xlim(2450, 4550)                  
ax.set_xticks([2500, 3000, 3500, 4000, 4500])                 # X-ticks at every 10 epochs

ax.set_ylim(34.9, 38.1)                  
ax.set_yticks([35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0,])    # Y-ticks for clarity
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Legend (fixed and clean)
legend_handles = [
    plt.Line2D([], [], marker='o', color='w', markerfacecolor='k',
               markeredgecolor='k', label=r"\textbf{En$\rightarrow$Fr}", markersize=7),
    plt.Line2D([], [], marker='^', color='w', markerfacecolor='k',
               markeredgecolor='k', label=r"\textbf{Fr$\rightarrow$En}", markersize=7),
]
# ax.legend(handles=legend_handles, loc='upper left', frameon=False, fontsize=10)
ax.legend(
    handles=legend_handles, 
    loc='upper left', 
    frameon=True, 
    columnspacing=0.8,
    handlelength=1.0,
    handletextpad=0.2,
    fontsize=exp_args.legend_font,
    )

# # ✅ Title
# ax.set_title(r"\textbf{Trade-off Between GPU Memory and BLEU Score}", pad=10)

# ✅ Save & show
os.makedirs("./Plots", exist_ok=True)
plt.savefig("./Plots/PeakVRAM_vs_BLEU_Centered.pdf",
            format="pdf",
            bbox_inches="tight", 
            facecolor="white",
            dpi=600)
plt.show()



# %%  
