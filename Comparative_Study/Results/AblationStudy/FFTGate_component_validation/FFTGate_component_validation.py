



# %% Imports and Setup


# ===============================================================
# 🔗==================== ABLATION STUDY 🔑====================🔗
# ===============================================================


########################################################################################################################
####-------| NOTE 1. IMPORTS LIBRARIES | XXX -------------------------------------------------------####################
########################################################################################################################



# ✅ === Enable flexible CUDA memory allocation to reduce fragmentation ===
# Must be set before importing torch!
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Alternative for memory split limits:
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"


# ✅ === Ensure correct working directory ===
import sys
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github4\IWSLT\Comparative_Study\Results\AblationStudy\FFTGate_component_validation"
if os.getcwd() != Project_PATH:
    os.chdir(Project_PATH)
print(f"✅ Current working directory: {os.getcwd()}")


# ✅ === Define core project paths ===
PROJECT_PATH = Project_PATH
MODELS_PATH = os.path.join(Project_PATH, "models")
ACTIVATION_PATH = os.path.join(Project_PATH, "activation")


# ✅ === Add essential paths to sys.path ===
for path in [PROJECT_PATH, MODELS_PATH, ACTIVATION_PATH]:
    if path not in sys.path:
        sys.path.append(path)

print("✅ sys.path updated:")
for path in sys.path:
    print("   📂", path)





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


# ======================================================================================================
# ✅ === Custom parser ===
# ======================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Dataset information
exp_parser = argparse.ArgumentParser("CIFAR-100 FFTGate_componen-Effect Experiment Config")
exp_parser.add_argument('--dataset_name', default="VGG16_CIFAR100", type=str,
    help="Choose dataset: VGG16_CIFAR100")  
# exp_parser.add_argument('--translation_direction', default="EN⟶FR", type=str)
# exp_parser.add_argument('--translation_direction', default=r"EN$\rightarrow$FR", type=str)

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









######################################## 1️⃣ 2️⃣ 3️⃣ Ⓐ Ⓑ Ⓒ 🅐 🅑 🅒 🅓 ##################################################
########################################################################################################################
####--🔴--| NOTE: BLEU Score Comparison (ReLU, GELU, TanhExp, GEGLU, FFTGate) | XXX ----------------####################
########################################################################################################################
# 🔗========================================== AblationStudy 🔑=====================================================🔗


# ======================================================================================================
# ✅ =======================🔖 TEST ACCURACY SCORE 🔖==================================================
# ======================================================================================================


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅐 Define Test log file paths
No_FFT_based_gating_test_log     = fr'./1-No_FFT_based_gating/Case1-Test_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_Ablation_no_fft_based_gating.txt'
no_fft_term_test_log     = fr'./2-No_FFT_term/Case2-Test_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_Ablation_no_fft.txt'
Base_FFTGate_test_log  = fr'./3-Base_FFTGate/Case3-Test_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_Ablation_fft_based_gatingONLY.txt'
Full_FFTGate_test_log    = fr'./4-Full_FFTGate/Case7-Test_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_Ablation_fftgate-WITH-residual_sm-james_stein_sm-targetReg-decay.txt'
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅑 Define Train log file paths
No_FFT_based_gating_Train_log     = fr'./1-No_FFT_based_gating/Case1-Train_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_Ablation_no_fft_based_gating.txt'
no_fft_term_Train_log     = fr'./2-No_FFT_term/Case2-Train_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_Ablation_no_fft.txt'
Base_FFTGate_Train_log  = fr'./3-Base_FFTGate/Case3-Train_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_Ablation_fft_based_gatingONLY.txt'
Full_FFTGate_Train_log    = fr'./4-Full_FFTGate/Case7-Train_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_Ablation_fftgate-WITH-residual_sm-james_stein_sm-targetReg-decay.txt'
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# ===============================================================
# 🔗=============== READ LOG FUNCTIONS (Updated for CIFAR logs) 🔑
# ===============================================================



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ Read test loss and accuracy
def read_test_metrics(file_path):
    epochs, test_losses, test_accs = [], [], []
    best_acc = None

    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return epochs, test_losses, test_accs, best_acc

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Example: "Epoch 0 | Test Loss: 4.332 | Test Acc: 3.270%"
            match = re.search(r"Epoch\s+(\d+)\s*\|\s*Test Loss:\s*([\d.]+)\s*\|\s*Test Acc:\s*([\d.]+)%", line)
            if match:
                epochs.append(int(match.group(1)))
                test_losses.append(float(match.group(2)))
                test_accs.append(float(match.group(3)))

            # Example: "🏆 Best Test Accuracy: 67.200%"
            best_match = re.search(r"Best Test Accuracy:\s*([\d.]+)%", line)
            if best_match:
                best_acc = float(best_match.group(1))
    return epochs, test_losses, test_accs, best_acc
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ Read train loss and accuracy
def read_train_metrics(file_path):
    epochs, train_losses, train_accs = [], [], []
    best_acc = None

    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return epochs, train_losses, train_accs, best_acc

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Example: "Epoch 0 | Train Loss: 4.635 | Train Acc: 1.623%"
            match = re.search(r"Epoch\s+(\d+)\s*\|\s*Train Loss:\s*([\d.]+)\s*\|\s*Train Acc:\s*([\d.]+)%", line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                train_accs.append(float(match.group(3)))

            # Example: "🏆 Best Training Accuracy: 98.472%"
            best_match = re.search(r"Best Training Accuracy:\s*([\d.]+)%", line)
            if best_match:
                best_acc = float(best_match.group(1))
    return epochs, train_losses, train_accs, best_acc
# ─────────────────────────────────────────────────────────────────────────────────────────────────






# ===============================================================
# 🔗=================== READ TEXTS 🔑=========================🔗
# ===============================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ Read Test Metrics (Loss + Accuracy)
No_FFT_based_gating_epochs_test, No_FFT_based_gating_test_loss, No_FFT_based_gating_test_acc, No_FFT_based_gating_best_acc = read_test_metrics(No_FFT_based_gating_test_log)
no_fft_term_epochs_test, no_fft_term_test_loss, no_fft_term_test_acc, no_fft_term_best_acc = read_test_metrics(no_fft_term_test_log)
Base_FFTGate_epochs_test, Base_FFTGate_test_loss, Base_FFTGate_test_acc, Base_FFTGate_best_acc = read_test_metrics(Base_FFTGate_test_log)
Full_FFTGate_epochs_test, Full_FFTGate_test_loss, Full_FFTGate_test_acc, Full_FFTGate_best_acc = read_test_metrics(Full_FFTGate_test_log)
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ Read Train Metrics (Loss + Accuracy)
No_FFT_based_gating_epochs_train, No_FFT_based_gating_train_loss, No_FFT_based_gating_train_acc, No_FFT_based_gating_best_train = read_train_metrics(No_FFT_based_gating_Train_log)
no_fft_term_epochs_train, no_fft_term_train_loss, no_fft_term_train_acc, no_fft_term_best_train = read_train_metrics(no_fft_term_Train_log)
Base_FFTGate_epochs_train, Base_FFTGate_train_loss, Base_FFTGate_train_acc, Base_FFTGate_best_train = read_train_metrics(Base_FFTGate_Train_log)
Full_FFTGate_epochs_train, Full_FFTGate_train_loss, Full_FFTGate_train_acc, Full_FFTGate_best_train = read_train_metrics(Full_FFTGate_Train_log)
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ✅ Optional — Print summary of best accuracies
print("\n📊 === Best Test Accuracies ===")
print(f"No FFT-based gating: {No_FFT_based_gating_best_acc:.2f}% | "
      f"No FFT term: {no_fft_term_best_acc:.2f}% | "
      f"Base FFTGate: {Base_FFTGate_best_acc:.2f}% | "
      f"Full FFTGate: {Full_FFTGate_best_acc:.2f}%")

print("\n📈 === Best Train Accuracies ===")
print(f"No FFT-based gating: {No_FFT_based_gating_best_train:.2f}% | "
      f"No FFT term: {no_fft_term_best_train:.2f}% | "
      f"Base FFTGate: {Base_FFTGate_best_train:.2f}% | "
      f"Full FFTGate: {Full_FFTGate_best_train:.2f}%")






# ===============================================================
# 🔗==================== PLOT FUNCTION 🔑=====================🔗
# ===============================================================

def plot_fftgate_ablation(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)


    # ────────────────────────────────────────────────────────────────
    # 🎨 Consistent, high-contrast palette for ablation variants
    COLORS = {
        "No_FFT_based_gating": "#8338EC",  # Gray — disabled FFT module
        "no_fft_term":         "#EF476F",  # Blue — no frequency term
        "Base_FFTGate":        "#06D6A0",  # Teal — only FFT-based gating
        "Full_FFTGate":        "#E49B0F"   # Gold — full mechanism (highlight)
    }                 #  "T64":  "#E49B0F", 
    # ────────────────────────────────────────────────────────────────



    # ===============================================================
    # 1️⃣ TEST ACCURACY vs EPOCH
    # ===============================================================
    fig_acc, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # 🔹 Baselines
    ax.plot(No_FFT_based_gating_epochs_test, No_FFT_based_gating_test_acc, label="No FFT-based gating",
            color=COLORS["No_FFT_based_gating"], linewidth=1.8, alpha=1.0)
    ax.plot(no_fft_term_epochs_test, no_fft_term_test_acc, label="No FFT term",
            color=COLORS["no_fft_term"], linewidth=1.8, alpha=1.0)
    ax.plot(Base_FFTGate_epochs_test, Base_FFTGate_test_acc, label="Base FFTGate",
            color=COLORS["Base_FFTGate"], linewidth=1.8, alpha=1.0)

    # 🔷 Highlighted curve — Full FFTGate
    ax.plot(
        Full_FFTGate_epochs_test,
        Full_FFTGate_test_acc,
        label="Full FFTGate",
        color=COLORS["Full_FFTGate"],
        linewidth=2.0,
        alpha=1.0,
        marker='o',
        markersize=4.5,
        markerfacecolor='white',
        markeredgecolor=COLORS["Full_FFTGate"],
        markeredgewidth=1.4,
        markevery=5,
        zorder=10
    )
    # ────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────
    ax.set_xlabel(r"$\mathbf{Epoch}$")
    ax.set_ylabel(r"$\mathbf{Test\ Accuracy\ (\%)}$")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(-2, max(No_FFT_based_gating_epochs_test)+2)
    # ────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────
    # ✅ Custom Axis Styling (Publication Style)
    ax.set_xlim(-4, 104)                  # 👈 add some padding before and after epochs
    ax.set_xticks(range(0, 105, 20))      # 👈 tick spacing every 20 epochs
    ax.set_ylim(56, 70)                   # 👈 Y-axis tuned to your accuracy range
    # ax.set_yticks(range(60, 73, 2))       # 👈 tick spacing every 2%
    ax.set_yticks([60, 65, 70])  # Y-ticks for clarity
    # ────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────
    # 🔧 Fine-tuning options
    leg = ax.legend(
        fontsize=exp_args.legend_font,
        loc="upper left",
        # bbox_to_anchor=(0.5, 1.16),  # 🔹 try 1.14 (closer) or 1.20 (higher)
        bbox_to_anchor=(-0.02, 1.03),   # 👈 y=1.05 moves legend slightly above default
        ncol=1,
        frameon=False,         # 🚫 removes the box frame
        facecolor="white",
        edgecolor="gray",
        # edgecolor="none",   # ✅ hides border but keeps white background
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.6,
    )
    leg._legend_box.align = "left"
    # ────────────────────────────────────────────────────────────────
    # ✅ Make legend text bold (LaTeX text mode, not math)
    for text in leg.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")

    # ✅ Draw canvas for legend coordinates (needed if you add a title later)
    fig_acc.canvas.draw()
   # ────────────────────────────────────────────────────────────────




   # ────────────────────────────────────────────────────────────────
    # ✅ Save figure
    fig_acc.savefig(
        os.path.join(save_dir, f"FFTGateAblation_TestAccuracy_vs_Epoch_{exp_args.dataset_name}.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    plt.show()



    # ===============================================================
    # 2️⃣ TEST LOSS vs EPOCH
    # ===============================================================
    fig_loss, ax_loss = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    ax_loss.plot(No_FFT_based_gating_epochs_test, No_FFT_based_gating_test_loss, label="No FFT-based gating",
                 color=COLORS["No_FFT_based_gating"], linewidth=1.8)
    ax_loss.plot(no_fft_term_epochs_test, no_fft_term_test_loss, label="No FFT term",
                 color=COLORS["no_fft_term"], linewidth=1.8)
    ax_loss.plot(Base_FFTGate_epochs_test, Base_FFTGate_test_loss, label="Base FFTGate",
                 color=COLORS["Base_FFTGate"], linewidth=1.8)

    ax_loss.plot(
        Full_FFTGate_epochs_test,
        Full_FFTGate_test_loss,
        label="Full FFTGate",
        color=COLORS["Full_FFTGate"],
        linewidth=2.0,
        marker='o',
        markersize=4.8,
        markerfacecolor='white',
        markeredgecolor=COLORS["Full_FFTGate"],
        markeredgewidth=1.5,
        markevery=5,
        zorder=10
    )

    ax_loss.set_xlabel(r"$\mathbf{Epoch}$")
    ax_loss.set_ylabel(r"$\mathbf{Test\ Loss}$")
    ax_loss.grid(True, linestyle="--", alpha=0.35)
    ax_loss.set_xlim(-4, 104)
    ax_loss.set_xticks(range(0, 105, 20))
    ax_loss.legend(fontsize=exp_args.legend_font, loc="upper right",
                   frameon=True, facecolor="white", edgecolor="gray")

    fig_loss.savefig(
        os.path.join(save_dir, f"FFTGateAblation_TestLoss_vs_Epoch_{exp_args.dataset_name}.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    plt.show()



    # ===============================================================
    # 3️⃣ TRAIN LOSS vs EPOCH
    # ===============================================================
    fig_train, ax_train = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    ax_train.plot(No_FFT_based_gating_epochs_train, No_FFT_based_gating_train_loss, label="No FFT-based gating",
                  color=COLORS["No_FFT_based_gating"], linewidth=1.8)
    ax_train.plot(no_fft_term_epochs_train, no_fft_term_train_loss, label="No FFT term",
                  color=COLORS["no_fft_term"], linewidth=1.8)
    ax_train.plot(Base_FFTGate_epochs_train, Base_FFTGate_train_loss, label="Base FFTGate",
                  color=COLORS["Base_FFTGate"], linewidth=1.8)

    ax_train.plot(
        Full_FFTGate_epochs_train,
        Full_FFTGate_train_loss,
        label="Full FFTGate",
        color=COLORS["Full_FFTGate"],
        linewidth=2.2,
        marker='o',
        markersize=4.8,
        markerfacecolor='white',
        markeredgecolor=COLORS["Full_FFTGate"],
        markeredgewidth=1.5,
        markevery=5,
        zorder=10
    )

    ax_train.set_xlabel(r"$\mathbf{Epoch}$")
    ax_train.set_ylabel(r"$\mathbf{Train\ Loss}$")
    ax_train.grid(True, linestyle="--", alpha=0.35)
    ax_train.set_xlim(-4, 104)
    ax_train.set_xticks(range(0, 105, 20))
    ax_train.legend(fontsize=exp_args.legend_font, loc="upper right",
                    frameon=True, facecolor="white", edgecolor="gray")

    fig_train.savefig(
        os.path.join(save_dir, f"FFTGateAblation_TrainLoss_vs_Epoch_{exp_args.dataset_name}.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    plt.show()

    print(f"✅ Saved all FFTGate ablation plots in: {save_dir}")




# 🔷 Call the function
plot_fftgate_ablation()

# %%


