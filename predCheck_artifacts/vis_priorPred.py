import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


if __name__=='__main__':

    # Load Std. Normal
    stdNorm_1 = pkl.load(open("predCheck_stdNorm_1.pkl", "rb"))
    stdNorm_2 = pkl.load(open("predCheck_stdNorm_2.pkl", "rb"))

    # Load He
    heScale_1 = pkl.load(open("predCheck_HeScaling_1.pkl", "rb"))
    heScale_2 = pkl.load(open("predCheck_HeScaling_2.pkl", "rb"))

    # Load PredCP
    ## TO DO
    ## TO DO

    vals1 = heScale_1
    vals2 = heScale_2


    dist_colors = ["#6d0303", "#E74C3C", "#F1948A"]

    fig, axs = plt.subplots(2, 1, figsize=(4, 2))

    axs[0].plot([x-5 for x in range(20)], [1/10. for x in range(20)], 'k--', lw=1)
    axs[1].plot([x-5 for x in range(20)], [1/10. for x in range(20)], 'k--', lw=1)

    axs[0].bar([x+1 for x in range(10)], vals1)
    axs[1].bar([x+1 for x in range(10)], vals2)

    axs[0].set_ylim([0,1])
    axs[0].set_xlim([0.4,10.6])
    axs[0].set_yticks(np.arange(0, 1.1, step=0.5))
    axs[0].set_xticks(np.arange(1, 10.1, step=1.))

    axs[1].set_ylim([0,1])
    axs[1].set_xlim([0.4,10.6])
    axs[1].set_yticks(np.arange(0, 1.1, step=0.5))
    axs[1].set_xticks(np.arange(1, 10.1, step=1.))
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
    
