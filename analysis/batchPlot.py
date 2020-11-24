from itertools import cycle, islice
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import sys

#---format---
# task_id  prec1 prec5  loss_avg loss_var euc_mean_avg euc_mean_var cos_mean_avg cos_mean_var euc_pair_avg euc_pair_var cos_pair_avg cos_pair_var


def breakLine(line,results,metric_list):

    task_id=int(line[1])
    prec1=float(line[3])
    prec5=float(line[5])
    loss_avg=float(line[7])
    loss_var=float(line[9])
    euc_mean_avg=float(line[11])
    euc_mean_var=float(line[13])
    cos_mean_avg=float(line[15])
    cos_mean_var=float(line[17])
    euc_pair_avg=float(line[19])
    euc_pair_var=float(line[21])
    cos_pair_avg=float(line[23])
    cos_pair_var=float(line[25])

    for i in range(len(metric_list)):
        metric=metric_list[i]
        results[task_id][metric].append(eval(metric))

def readClassFeature(line,results,metric_list):
    for index,metric in enumerate(metric_list):
        results[metric].append(line[index*2+1])


def readFile(filename):
    metric_list=['prec1','prec5','loss_avg','loss_var','euc_mean_avg','euc_mean_var','cos_mean_avg','cos_mean_var','euc_pair_avg','euc_pair_var','cos_pair_avg','cos_pair_var']
    class_metric_list=['class_euc_pair_avg','class_euc_pair_var','class_cos_pair_avg','class_cos_pair_var','class_var']



    # change the dir path if we want to check results of different settings
    dir='../'+'difficulty_batch_10_lr_0.025'

    path=os.path.join(dir,filename)
    f=open(path,'r')
    results=dict()

    n_task=5
    if 'cifar100' in filename:
        n_task=20
    for i in range(n_task):
        results[i]=dict()
        for metric in metric_list:
            results[i][metric]=[]
        for v in class_metric_list:
            results[v]=[]
        results['avg_acc']=[]
        

    counter=0
    while True:
        line=f.readline()
        if not line:
            break

        line=line.strip().split()
        if len(line)==26:
            breakLine(line,results,metric_list)
        if len(line)==8 or len(line)==12:
            results['avg_acc'].append(float(line[-1]))
        if len(line)==10:
            counter+=1
            if counter%10==0:
                readClassFeature(line,results,class_metric_list)


    f.close()
    return results




def batchPlot():

    dataset='cifar10'
    
    #strategy_list=['MOF','reservoir_class','kmeans','AR']
    #showname_list=['MOF','reservoir','coreset','AR']
    pretrain_list=['None','cifar10','cifar100']
    metric_list=['prec1','prec5','loss_avg','loss_var','euc_mean_avg','euc_mean_var','cos_mean_avg','cos_mean_var','euc_pair_avg','euc_pair_var','cos_pair_avg','cos_pair_var']
    class_metric_list=['class_euc_pair_avg','class_euc_pair_var','class_cos_pair_avg','class_cos_pair_var','class_var']
    color_list=['green','skyblue','blue','orange','red','cyan','black','magenta','gray','pink','olive','violet']

    #strategy_list=['Loss_triplet_no_hard_mining_MOF','Loss_triplet_MOF']
    #alpha_list=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]


    strategy_list=['MOF','reservoir_class','kmeans','Loss_triplet_no_hard_mining_reservoir_0.7','Loss_triplet_reservoir_0.7']

    showname_list=['MOF','reservoir','coreset','Triplet_Easy','Triplet_Hard']



    file_list=[]
    results_list=[]

    n_task=5
    if 'cifar100' in dataset:
        n_task=20

    plot_metric_list=['class_var']


    H_plot=len(pretrain_list)
    W_plot=len(strategy_list)    


    # test parameter alpha
    '''
    alpha_list=[0.6,0.7,0.8,0.9]
    for index,strategy in enumerate(strategy_list):
        print(strategy)
        for pretrain in pretrain_list:
            print(pretrain)
            for alpha in alpha_list:
                showname=showname_list[index]+'_'+str(alpha)+'_'+pretrain
                file_list.append(showname)
                filename='difficulty_'+dataset+'_'+strategy+'_'+str(alpha)+'_'+pretrain
                results=readFile(filename)
                results_list.append(results)
                print(results['avg_acc'][-1],end =", ")

            print()
        print()

    for index,strategy in enumerate(strategy_list):
        print(strategy)
        for pretrain in pretrain_list:
            print(pretrain)
            showname=showname_list[index]+'_'+pretrain
            file_list.append(showname)
            filename='difficulty_'+dataset+'_'+strategy+'_'+pretrain
            results=readFile(filename)
            results_list.append(results)
            print(results['avg_acc'][-1],end =", ")

            print()
        print()
    '''



    for pretrain in pretrain_list:
        for index,strategy in enumerate(strategy_list):
            file_list.append(showname_list[index]+'_'+pretrain)
            filename='difficulty_'+dataset+'_'+strategy+'_'+pretrain
            results=readFile(filename)
            results_list.append(results)

    for metric in plot_metric_list:
        print(metric)
        fig, axes=plt.subplots(W_plot,H_plot,sharex=True,sharey=True,figsize=(W_plot * 2 + 4, H_plot * 2 + 1))
        for index,results in enumerate(results_list):
            plt.subplot(H_plot, W_plot, index+1)  
            plt.title(file_list[index])

            if metric in metric_list:
                for task_id in range(n_task):
                    value=results[task_id][metric]
                    length=len(value)
                    x=list(range(1,length+1))
                    plt.plot(x,value,label='task'+str(task_id+1))

                    if metric=='prec1':
                        value=results['avg_acc']
                        new_x=list(range(1,len(value)+1))
                        plt.plot(new_x,value,label='avg_acc')
                        print(file_list[index],' ',results['avg_acc'][-1])

            elif metric in class_metric_list:
                value=results[metric]
                print(value)
                length=len(value)
                x=list(range(1,length+1))
                plt.plot(x,value)

                value=results['avg_acc']
                print(file_list[index],' ',results['avg_acc'][-1])


            #plt.tick_params()

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        ncol=n_task+1 if metric=='prec1' else n_task
        fig.legend(lines, labels, loc = 'upper center',ncol=ncol)        
        label_fig=axes[0,0].figure
        if metric in metric_list:
            label_fig.text(0.5,0.01,'batch_num', ha="center", va="center",fontsize=14)
        elif metric in class_metric_list:
            label_fig.text(0.5,0.01,'task_num', ha="center", va="center",fontsize=14)

        label_fig.text(0.01,0.5,metric, ha="center", va="center", rotation=90,fontsize=14)
        fig.tight_layout()
        plt.show()




def batchCorrelation():

    dataset='cifar10'
    strategy_list=['MOF','reservoir','kmeans','hash','single']
    showname_list=['MOF','reservoir','coreset','LSH','finetune']
    pretrain_list=['None','cifar100','cifar10']
    metric_list=['prec1','prec5','loss_avg','loss_var','euc_mean_avg','euc_mean_var','cos_mean_avg','cos_mean_var','euc_pair_avg','euc_pair_var','cos_pair_avg','cos_pair_var']
    file_list=[]
    results_list=[]
    n_task=5

    H_plot=2
    W_plot=3   

    # it is really difficult to identify any useful correlation
    for pretrain in pretrain_list:
        for index,strategy in enumerate(strategy_list):
            file_list.append(showname_list[index]+'_'+pretrain)
            filename='difficulty_'+dataset+'_'+strategy+'_'+pretrain
            results=readFile(filename)
            results_list.append(results)


    index_list=[2,3,4,14]
    for index,results in enumerate(results_list):
        if not index in index_list:
            continue
        print(file_list[index])
        fig, axes=plt.subplots(W_plot,H_plot,sharex=True,sharey=True,figsize=(W_plot * 2 + 4, H_plot * 2 + 3))
        sns.set(font_scale=0.7)

        for task_id in range(n_task):
            results[task_id]['avg_acc']=results['avg_acc']
            class_data=pd.DataFrame.from_dict(results[task_id])
            plt.subplot(H_plot, W_plot, task_id+1)  
            plt.title(str(task_id+1))
            sns.heatmap(class_data.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=1, linecolor='black',cbar=False)

        plt.show()


if __name__ == "__main__":

    batchPlot()