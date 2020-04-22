import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


if __name__=='__main__':

    # Load Std. Normal
    stdNorm_1 = pkl.load(open("predCheck_stdNorm_1_wBN.pkl", "rb"))
    stdNorm_2 = pkl.load(open("predCheck_stdNorm_2_wBN.pkl", "rb"))
    stdNorm_3 = pkl.load(open("predCheck_stdNorm_3_wBN.pkl", "rb"))

    # Load He
    heScale_1 = pkl.load(open("predCheck_HeScaling_1_wBN.pkl", "rb"))
    heScale_2 = pkl.load(open("predCheck_HeScaling_2_wBN.pkl", "rb"))
    heScale_3 = pkl.load(open("predCheck_HeScaling_3_wBN.pkl", "rb"))

    # Load PredCP
    ## TO DO
    ## TO DO

    vals1 = stdNorm_1 #heScale_1
    vals2 = stdNorm_2 #heScale_2
    vals3 = stdNorm_3 #heScale_3


    dist_colors = ["#6d0303", "#E74C3C", "#F1948A"]
    n_plots = 3

    fig, axs = plt.subplots(n_plots, 1, figsize=(4, 4))

    axs[0].bar([x+1 for x in range(10)], vals1)
    axs[1].bar([x+1 for x in range(10)], vals2)
    axs[2].bar([x+1 for x in range(10)], vals3)

    for idx in range(n_plots):
        axs[idx].plot([x-5 for x in range(20)], [1/10. for x in range(20)], 'k--', lw=1)
        
        axs[idx].set_ylim([0,1])
        axs[idx].set_xlim([0.4,10.6])
        axs[idx].set_yticks(np.arange(0, 1.1, step=0.5))
        axs[idx].set_xticks(np.arange(1, 10.1, step=1.))
        axs[idx].set_ylabel("Class Probability", fontsize=10)

    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
    
