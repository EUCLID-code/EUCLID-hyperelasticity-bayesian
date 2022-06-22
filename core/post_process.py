import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
from sklearn.metrics import r2_score
matplotlib.pyplot.rcParams["font.family"] = "serif"
matplotlib.pyplot.rcParams["mathtext.fontset"] = "dejavuserif"

import seaborn as sns
import numpy as np
from features_library import *

def print_solution(theta):
    """
    
    """
    print(np.expand_dims(theta.squeeze(),axis=1))

def post_proc(chain, theta_gt, feature_filter, fem_mat, energy_func, fig_title, fig_title2, fig_dir = None, plotting=True, interactive_job=True):
    """
    Making output plots containing a summary of the chains, and the corresponding predicted energies

    _Input Arguments_

    -`chain` - object of `Chain` class (see `core_spike_slab` file)

    -`theta_gt` - The true set of feature coefficients for the benchmark material

    -`feature_filter` - The list of features to retain for constructing the Markov chain. Suppressed features will be highlighted with a red patch in the plot

    -`fem_mat` - The name of the benchmark material to be tested

    -`energy_func` - The label of the function used to predict energy evolution for the discovered material along 6 different deformation paths

    -`fig_title` - Title displayed on the figure

    -`fig_title2` - Filename of the saved figure (.png format)

    ---

    """
    
    if interactive_job==False:
        matplotlib.use('Agg')
        plotting=False

    fig=plt.figure(figsize=(20,10))

    gs=GridSpec(3,5)

    axes = []
    axes.append(fig.add_subplot(gs[0:2,0:3]))
    axes.append(fig.add_subplot(gs[2,0:3]))
    axes.append(fig.add_subplot(gs[0,3:4]))
    axes.append(fig.add_subplot(gs[1,3:4]))
    axes.append(fig.add_subplot(gs[2,3:4]))
    axes.append(fig.add_subplot(gs[0,4:5]))
    axes.append(fig.add_subplot(gs[1,4:5]))
    axes.append(fig.add_subplot(gs[2,4:5]))

    plt.tight_layout(pad=6)
    plt.rcParams.update({'font.size':12})

    fig.suptitle(fig_title, fontsize=24, x=0.5, y=1.00)
    num_feats= getNumberOfFeatures()
    x_tik2 = list(range(num_feats+1));
    x_tik2.pop(0);
    x_tikfilt = [x_tik2[i] for i in feature_filter];
    plottheta = np.zeros([chain.theta.shape[0],num_feats])
    plotz = np.zeros([chain.theta.shape[0],num_feats],dtype=int)
    for i in range(chain.theta.shape[1]):
        plottheta[:,(x_tikfilt[i]-1)]=chain.theta[:,i]
        plotz[:,(x_tikfilt[i]-1)]=chain.z[:,i]


    def plot_gt(id, gt = theta_gt):
        for i in range(chain.theta.shape[1]):
            sns.lineplot(x=[(x_tikfilt[i]-1)-0.4,(x_tikfilt[i]-1)+0.4],y=[gt[i]]*2, color='red', linewidth=1, ax=axes[id])

    def plot_gt_highlight(id):
        for i in range(chain.theta.shape[1]):
            if np.abs(theta_gt[i]) > 1e-4:
                axes[id].add_patch(patches.Rectangle(((x_tikfilt[i]-1)-0.4, -2),0.8,4.5, facecolor='cyan', edgecolor='none', alpha=0.4, zorder=0))
        for i in range(num_feats):
            if i not in feature_filter:
                axes[id].add_patch(patches.Rectangle((i-0.4, -2),0.8,4.5, facecolor='red', edgecolor='none', alpha=0.4, zorder=1))

    def set_lims(id, ylims):
        axes[id].set_ylim(ylims[0], ylims[1])
        axes[id].set_xlim(-1, num_feats)

    def set_labels(id,x,y):
        plt.sca(axes[id])
        plt.xticks(list(range(num_feats)),x_tik2,rotation=0,fontsize=15)
        axes[id].tick_params(axis='y', labelsize=15)
        axes[id].set_xlabel(x,fontsize =16)
        axes[id].set_ylabel(y,fontsize =16)


    def plot_trace(id):
        for i in range(chain.theta.shape[0]):
            sns.lineplot(x=list(range(num_feats)),y=list(plottheta[i,:]), color='black', dashes=[(2,2)], linewidth=0.5, alpha=0.02, ax=axes[id])

    cyan_patch = patches.Patch(color='cyan',alpha=0.4,label='True feature')
    red_patch = patches.Patch(color='red',alpha=0.4,label='Suppressed feature')
    blue_patch = patches.Patch(color='blue',alpha=1,label='Feature activity')
    #violin plots
    sns.violinplot(data=[d for d in plottheta.T],bw=0.2,cut=0,scale='count',split=False,xlabel='add',ax=axes[0])
    plot_gt(0)
    plot_gt_highlight(0)
    plot_trace(0)
    set_lims(0, ylims=[-0.25,2.5])
    set_labels(0,'Feature index', r'Probability density')
    axes[0].text(-0.5,2.55,r'(a) Posterior  probability: $\mathbf{\theta}$',fontsize = 16)
    if chain.theta.shape[1]<num_feats:
        axes[0].legend(handles=[Line2D([0],[0],color='r',lw=2,label='True value'),cyan_patch,red_patch, Line2D([0],[0],color='grey',lw=1,label='Posterior samples')],fontsize=15, ncol=4, loc=1, facecolor='white', framealpha=1, edgecolor='black')
    else:
        axes[0].legend(handles=[Line2D([0],[0],color='r',lw=2,label='True value'),cyan_patch, Line2D([0],[0],color='grey',lw=1,label='Posterior samples')],fontsize=15, ncol=3, loc=1, facecolor='white', framealpha=1, edgecolor='black')


    #bar plots for activity
    sns.barplot(x=np.arange(num_feats),y=plotz.mean(axis=0), ax=axes[1], color="blue")
    plot_gt(1, gt = (np.abs(theta_gt)>1e-8)*1.0)
    plot_gt_highlight(1)
    set_lims(1, ylims=[0.,1.4])
    plt.sca(axes[1])
    plt.yticks([0.0,0.5,1.0],[0.0,0.5,1.0],fontsize=15)
    set_labels(1,"Feature index", "Avg. activity")
    axes[1].text(-0.5,1.45,'(b) Avg. activity of features',fontsize = 16)
    if chain.theta.shape[1]<num_feats:
        axes[1].legend(handles=[blue_patch,cyan_patch,red_patch],fontsize=15, ncol=3, loc=1, facecolor='white', framealpha=1, edgecolor='black')
    else:
        axes[1].legend(handles=[blue_patch,cyan_patch],fontsize=15, ncol=2, loc=1, facecolor='white', framealpha=1, edgecolor='black')

    alid = ['(c)','(d)','(e)','(f)','(g)','(h)']
    def energy_plot(id, deformation):
        gamma, W_mean, W_plus, W_minus, W_gt, W_all = energy_func(chain, theta_gt, fem_mat, feature_filter, deformation)
        # GT
        sns.lineplot(x=gamma, y=W_gt,  ax=axes[id], color='red',dashes=[(2, 2)], label='True')
        # # mean
        sns.lineplot(x=gamma, y=W_mean,  ax=axes[id], color='black', label='Mean')
        # confidence intervals
        R2 = r2_score(W_gt,W_mean)
        temp = str(R2)
        axes[id].fill_between(gamma, W_minus, W_plus, color='silver', label='95-perc.')
        #labels
        axes[id].set_xlabel('Deformation ($\gamma$)',fontsize=16)
        axes[id].set_ylabel('W($\gamma$)',fontsize=16)
        axes[id].set_title(alid[id-2] + ' ' + deformation.title().replace("_"," "), fontsize=16)
        axes[id].tick_params(axis='x',labelsize =12)
        axes[id].tick_params(axis='y',labelsize =12)
        axes[id].set_ylim(0,1.3*np.max(W_gt))
        handles, labels = axes[id].get_legend_handles_labels()
        axes[id].legend(handles=[handles[0],handles[1],handles[2]],
            labels=[labels[0],labels[1],labels[2]],
            fontsize=14, ncol=1, loc=2, facecolor='white', framealpha=0.,borderpad=0.,labelspacing=0.2,handlelength=0.7)
        axes[id].text(0.24,0.54,'R$^2$ = '+temp[0:5],ha='center',va='center',transform=axes[id].transAxes,zorder=2,fontsize=13)

    energy_plot(2,'tension')
    energy_plot(3,'simple_shear')
    energy_plot(4,'pure_shear')
    energy_plot(5,'biaxial_tension')
    energy_plot(6,'compression')
    energy_plot(7,'biaxial_compression')

    if fig_dir is not None:
        if(fig_dir[-1]=='/'):
            fig_dir = fig_dir[0:-1]
        plt.savefig(fig_dir+'/'+fig_title2+'.png')

    if plotting:
        plt.show()
        plt.close('all')
