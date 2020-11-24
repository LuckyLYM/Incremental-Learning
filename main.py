import sys
sys.path.append("./model")
import importlib # for dynamic class
import pickle
import datetime
import argparse
import random
import uuid
import time
import os
import pdb
import numpy as np
import logging
import torch
import torch.nn as nn
from metrics.metrics import confusion_matrix
from model import pretrained_cifar
import utils
from torch.autograd import Variable



def load_datasets(args):
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)

    # tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()])
    # tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])

    if args.tasks_to_preserve>0:
        d_tr=d_tr[:args.tasks_to_preserve]
        d_te=d_te[:args.tasks_to_preserve]

    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max().item())
        n_outputs = max(n_outputs, d_te[i][2].max().item())
    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)

class Continuum:

    def __init__(self, data, args):
        self.data = data
        self.batch_size = args.batch_size
        n_tasks = len(data)
        task_permutation = range(n_tasks)
        if args.shuffle_tasks == 'yes':
            task_permutation = torch.randperm(n_tasks).tolist()
        sample_permutations = []
        samples_per_task= args.samples_per_task

        # data format: [(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()]
        n=0
        for t in range(n_tasks):
            N=data[t][1].size(0)
            if samples_per_task<0:
                n=N
            else:
                n=min(N,samples_per_task)
            p = torch.randperm(data[t][1].size(0))[0:n]
            sample_permutations.append(p)
        
        self.permutation = []
        for t in range(n_tasks):
            task_t = task_permutation[t]
            for _ in range(args.n_epochs):
                task_p = [[task_t, i] for i in sample_permutations[task_t]]
                random.shuffle(task_p)
                self.permutation += task_p
        self.length = len(self.permutation)
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            ti = self.permutation[self.current][0]  # task id
            j = []
            i = 0
            # do not excced length of the stream, the batch size, the task id
            while (((self.current + i) < self.length) and
                   (self.permutation[self.current + i][0] == ti) and
                   (i < self.batch_size)):
                j.append(self.permutation[self.current + i][1])
                i += 1
            self.current += i
            j = torch.LongTensor(j)
            # return a batch of data and label
            return self.data[ti][1][j], ti, self.data[ti][2][j]


def dist_matrix(x,y=None):
    if y is None:
        y=x.clone()
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)
    return dist


def normalize(x):
    if x.shape[0]==1:
        return x/torch.norm(x,p=2)
    else:
        return x/torch.norm(x,p=2,dim=1).unsqueeze(1)

def cos_matrix(x,y=None):
    if y is None:
        x=normalize(x)
        y=x.clone()
    else:
        x=normalize(x)
        y=normalize(y)

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = (x*y).sum(2)
    return dist


# We modify the eval_task function here
def eval_tasks(model,tasks,current_task,args,criterion,file=None):
    model.eval()
    total_size=0
    total_pred=0
    current_avg_acc = 0
    class_feature_mean=None


    # enumerate each task
    for i, task in enumerate(tasks):
        t = i
        x = task[1]
        y = task[2]
        with torch.no_grad():
            if args.cuda:
                x = Variable(x).cuda()
                y = Variable(y).cuda()
        

        features,output=model.feature_and_output(x)

        if args.log_difficulty:
            # validation loss                
            loss=criterion(output,y)
            loss_avg=torch.mean(loss)
            loss_var=torch.var(loss)
            
            # feature mean distance
            feature_mean=torch.mean(features,dim=0,keepdim=True)
            euclidean_dist=dist_matrix(features,feature_mean) # N*1
            euc_mean_avg=torch.mean(euclidean_dist)
            euc_mean_var=torch.var(euclidean_dist)

            cos_dist=cos_matrix(features,feature_mean)
            cos_mean_avg=torch.mean(cos_dist)
            cos_mean_var=torch.var(cos_dist)

            # pairwise distance
            euc_pair=dist_matrix(features)
            euc_pair_avg=torch.mean(euc_pair)
            euc_pair_var=torch.var(euc_pair)

            cos_pair=cos_matrix(features)
            cos_pair_avg=torch.mean(cos_pair)
            cos_pair_var=torch.var(cos_pair)

            # new feature, feature mean
            if class_feature_mean is None:
                class_feature_mean=feature_mean
            else:
                class_feature_mean=torch.cat((class_feature_mean,feature_mean),dim=1)

        prec1, prec5 = utils.accuracy(output, y, topk=(1, 5))
        total_size+= x.size(0)
        total_pred+=round(float(prec1)*x.size(0))
        if t == current_task:
            current_avg_acc=total_pred / total_size


        if args.print_log:
            if args.log_difficulty:
                print('task_id: %d prec1: %f prec5: %f loss_avg: %f loss_var: %f euc_mean_avg: %f euc_mean_var: %f cos_mean_avg: %f cos_mean_var: %f euc_pair_avg: %f euc_pair_var: %f cos_pair_avg: %f cos_pair_var: %f'%(i,prec1,prec5,loss_avg,loss_var,euc_mean_avg,euc_mean_var,cos_mean_avg,cos_mean_var,euc_pair_avg,euc_pair_var,cos_pair_avg,cos_pair_var))
            else:
                print('task_id: %d prec1: %f prec5: %f'%(i,prec1,prec5))                

        if args.log_difficulty:
            file.write('task_id: %d prec1: %f prec5: %f loss_avg: %f loss_var: %f euc_mean_avg: %f euc_mean_var: %f cos_mean_avg: %f cos_mean_var: %f euc_pair_avg: %f euc_pair_var: %f cos_pair_avg: %f cos_pair_var: %f\n'%(i,prec1,prec5,loss_avg,loss_var,euc_mean_avg,euc_mean_var,cos_mean_avg,cos_mean_var,euc_pair_avg,euc_pair_var,cos_pair_avg,cos_pair_var))
    

    # end of enumerate each task 
    average_acc=total_pred/total_size
    if args.log_difficulty:
        class_euc_pair=dist_matrix(class_feature_mean)
        class_euc_pair_avg=torch.mean(euc_pair)
        class_euc_pair_var=torch.var(euc_pair)
        class_cos_pair=cos_matrix(class_feature_mean)
        class_cos_pair_avg=torch.mean(cos_pair)
        class_cos_pair_var=torch.var(cos_pair)
        class_var=torch.sum(torch.var(class_feature_mean,dim=1))
        file.write("class_euc_pair_avg: %f class_euc_pair_var: %f class_cos_pair_avg: %f class_cos_pair_var: %f class_var: %f\n"%(class_euc_pair_avg,class_euc_pair_var,class_cos_pair_avg,class_cos_pair_var,class_var))

    #result=[class_euc_pair_avg,class_euc_pair_var,class_cos_pair_avg,class_cos_pair_var,class_var]

    return average_acc,current_avg_acc

def eval_tasks_NEM(model,tasks,current_task,args,pretrained,criterion,file=None):
    result = []
    total_size=0
    total_pred=0
    current_result = []
    current_avg_acc = 0

    for i, task in enumerate(tasks):
        t = i
        x = task[1]
        y = task[2]
        rt = 0
        eval_bs = x.size(0)
        x=x.cuda()
        y=y.cuda()

        features = pretrained.feature_extractor(x).data
        features = features/torch.norm(features, p=2,dim=1).unsqueeze(1)
        dist=dist_matrix(features, model.mean_features)
        pb=torch.argmin(dist,dim=1)
        rt += (pb == y).float().sum()

        if args.print_log:
            print('task_id: %d test_size: %d correct_num: %d'%(i,eval_bs, rt))

        result.append(rt / x.size(0))
        total_size+= x.size(0)
        total_pred+=rt

        if t == current_task:
            current_result=[res for res in result]
            current_avg_acc=total_pred / total_size

    return result,total_pred/total_size,current_result,current_avg_acc


def eval_on_memory(args):
    model.eval()
    acc_on_mem=0
    if 'yes' in args.eval_memory:
        for x,y in zip(model.sampled_memory_data,model.sampled_memory_labs):
            if args.cuda:
                x = x.cuda()
            _, pb = torch.max(model(x.unsqueeze(0)).data.cpu(), 1, keepdim=False)
            acc_on_mem += (pb == y.data.cpu()).float()
        acc_on_mem=(acc_on_mem/model.sampled_memory_data.size(0))
    return acc_on_mem

def life_experience(model, continuum, x_te, args):

    batch_id=0
    current_task = 0
    time_start = time.time()
    train_time=0
    pretrained=None
    criterion = nn.CrossEntropyLoss(reduction='none')  # used for validation set only
    criterion = criterion.cuda()

    if args.pretrained:
        pretrained=pretrained_cifar.cifar_resnet20(pretrained='cifar100')
        pretrained.cuda()
        for param in pretrained.parameters():
            param.requires_grad = False
        pretrained.eval()

    log_file=None
    model_name=None
    dataset=args.data_file[:-3]
    if args.model!='Loss' and args.model!='SDC':
        model_name=args.model+'_'+args.mode
    else:
        model_name=args.model+'_'+args.mode+'_'+args.loss+'_'+args.memory_update+'_'+str(args.alpha)
    model_name=model_name+'_'+args.finetune
    
    if args.log_difficulty:
        filename='difficulty_'+dataset+'_'+model_name
        filename=os.path.join('log',filename)
        log_file=open(filename,"w")


    for (i, (x, t, y)) in enumerate(continuum):
        batch_id=i

        ############################# results on validation set ###############################
        if((i!=0 and (i % args.log_every) == 0) or (t != current_task)):
            average_acc,current_avg_acc=eval_tasks(model, x_te,current_task, args,criterion,log_file)
            current_task = t
            if args.log_difficulty:
                log_file.write("###################### Task_ID: %d ##########################\n"%current_task)
                log_file.write('task_id: %d batch_id: %d train_time: %.6f  total_time: %.6f average_acc: %f current_avg_acc: %f\n\n'%(current_task,batch_id,train_time,time.time()-time_start,average_acc,current_avg_acc))
            if args.print_log:
                print("###################### Task_ID: %d ##########################"%current_task)
                print('task_id: %d batch_id: %d train_time: %.6f  total_time: %.6f average_acc: %f current_avg_acc: %f\n\n'%(current_task,batch_id,train_time,time.time()-time_start,average_acc,current_avg_acc))

            if args.model=='adaptive':
                model.update_results()

        ###################################### train model ####################################
        v_x = x.view(x.size(0), -1)
        v_y = y.long()
        if args.cuda:
            v_x = v_x.cuda()
            v_y = v_y.cuda()
        train_start=time.time()
        model.train()
        model.observe(v_x, t, v_y, pretrained)
        train_time+=time.time()-train_start



    ###################################### finish model training ####################################
    batch_id+=1
    average_acc,current_avg_acc=eval_tasks(model, x_te,current_task, args,criterion,log_file)
    if args.log_difficulty:
        log_file.write("###################### Task_ID: %d ##########################\n"%current_task)
        log_file.write('task_id: %d batch_id: %d train_time: %.6f  total_time: %.6f average_acc: %f current_avg_acc: %f\n\n'%(current_task,batch_id,train_time,time.time()-time_start,average_acc,current_avg_acc))
    print("FINAL model: %s n_batch: %d current_avg_acc: %.6f  train_time: %.6f  total_time: %.6f"%(model_name,batch_id,current_avg_acc,train_time,time.time()-time_start))
    if args.log_difficulty:
        log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continuum learning')

    # model parameters
    parser.add_argument('--model', type=str, default='single',
                        help='model to train')
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--shared_head', type=str, default='yes',
                        help='shared head between tasks')
    parser.add_argument('--bias', type=int, default='1',
                        help='do we add bias to the last layer? does that cause problem?')
    parser.add_argument('--pretrained', type=str, default='yes',
                        help='Use pretrained model for feature extraction?')

    # memory parameters
    parser.add_argument('--n_memories', type=int, default=0,
                        help='number of memories per task')
    parser.add_argument('--n_sampled_memories', type=int, default=0,
                        help='number of sampled_memories per task')
    parser.add_argument('--n_constraints', type=int, default=0,
                        help='number of constraints to use during online training')
    parser.add_argument('--b_rehearse', type=int, default=0,
                        help='if 1 use mini batch while rehearsing')
    parser.add_argument('--tasks_to_preserve', type=int, default=-1,
                        help='number of tasks to preserve. If -1, use all the tasks')
    parser.add_argument('--change_th', type=float, default=0.0,
                        help='gradients similarity change threshold for re-estimating the constraints')
    parser.add_argument('--slack', type=float, default=0.01,
                        help='slack for small gradient norm')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')


    # optimizer parameters
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--n_iter', type=int, default=1,
                        help='Number of iterations per batch')
    parser.add_argument('--repass', type=int, default=0,
                        help='make a repass over the previous data')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--mini_batch_size', type=int, default=0,
                        help='mini batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='SGD learning rate')

    # experiment parameters
    parser.add_argument('--cuda', type=str, default='no',
                        help='Use GPU?')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models at the end of training')
    parser.add_argument('--output_name', type=str, default='',
                        help='special output name for the results?')
    parser.add_argument('--gpu_id', type=int, default=3,
                        help='id of gpu we use for training')
    parser.add_argument('--disable_random', type=str, default='yes',
                        help='not to use random seed?')  
    parser.add_argument('--print_log', type=str, default='yes',
                        help='whether print the log?')
    parser.add_argument('--normalize', type=str, default='yes',
                        help='whether to normalize the feature embedding of samples?')
    parser.add_argument('--mode', type=str, default='double',
                        help='single batch update or two batch update')




    # model distillation
    parser.add_argument('--distill', type=str, default='no',
                        help='Use cross distillation?')
    parser.add_argument('--T', type=float, default=2, help='Tempreture used for softening the targets')
    parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight given to new classes vs old classes in the loss; high value of alpha will increase perfomance on new classes at the expense of older classes.')
    parser.add_argument('--feature_update', type=str, default='all',
                        help='we update the feature of all data or the latest task only')
    parser.add_argument('--memory_update', type=str, default='reservoir',
                        help='buffer update strategy')
    parser.add_argument('--loss', type=str, default='triplet',
                        help='the loss function used in memory replay')
    parser.add_argument('--margin', type=float, default='0.0',
                        help='the margin used in the triplet loss function')
    parser.add_argument('--num_instances', type=int, default=4,
                        help='the number of triplet pair for an anchor point')

    # This is the important part, check the loss function here.
    #criterion = losses.create(args.loss, margin=args.margin, num_instances=args.num_instances).cuda()

    # data parameters
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--data_file', default='mnist_permutations.pt',
                        help='data file')
    parser.add_argument('--samples_per_task', type=int, default='-1',
                        help='training samples per task (all if negative)')
    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')
    parser.add_argument('--eval_memory', type=str, default='no',
                        help='compute accuracy on memory')
    parser.add_argument('--age', type=float, default=1,
                        help='consider age for sample selection')
    parser.add_argument('--subselect', type=int, default=1,
                        help='first subsample from recent memories')

    # will update the value according to  the specific dataset we use
    parser.add_argument('--class_per_task', type=int, default='-1',
                        help='the number of classes per task')
    parser.add_argument('--total_task', type=int, default='-1',
                        help='the number of task')
    parser.add_argument('--n_class', type=int, default='-1',
                        help='the number of total classes')

    # used for exstream algorithm
    parser.add_argument('--pseudo', type=str, default='no',
                        help='generate psuedo samples when using exstream clustering?')

    # used for hash clustering
    parser.add_argument('--projected_dim', type=int, default=4,
                        help='the dimension of hash projected vector')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='the dimension of extracted features')
    parser.add_argument('--n_hash_table', type=int, default=3,
                        help='the dimension of extracted features')
    parser.add_argument('--finetune', type=str, default='None',
                        help='finetune model or train model from scratch?')
    parser.add_argument('--distance', type=str, default='Euclidean',
                        help='the distance measure used to calculate the similarity')
    parser.add_argument('--log_difficulty', type=str, default='no',
                        help='log the information we needed to find the relationship between task difficulty and data selection strategy')


    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    args.normalize = True if args.normalize == 'yes' else False
    args.shared_head = True if args.shared_head == 'yes' else False
    args.pretrained = True if args.pretrained == 'yes' else False
    args.distill = True if args.distill == 'yes' else False
    args.disable_random = True if args.disable_random=='yes' else False
    args.pseudo= True if args.pseudo=='yes' else False
    args.print_log= True if args.print_log=='yes' else False
    args.log_difficulty=True if args.log_difficulty=='yes' else False


    if args.mini_batch_size==0:
        args.mini_batch_size=args.batch_size #no mini iterations

    # multimodal model has one extra layer
    if args.model == 'multimodal':
        args.n_layers -= 1

    # when set false, this is a deterministic algorithm, while being less efficient
    if args.disable_random:
        torch.backends.cudnn.enabled = False # if true, use non-determnistic algorithm
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.cuda:
        torch.cuda.set_device(args.gpu_id)
        if args.disable_random:
            torch.cuda.manual_seed_all(args.seed)

    x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)
    args.total_task=n_tasks
    args.n_class=n_outputs

    if x_tr[0][0]=='random permutation':
        args.class_per_task=1
    else:
        args.class_per_task=x_tr[0][0][1]-x_tr[0][0][0]


    continuum = Continuum(x_tr, args)
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    if args.cuda:
        model.cuda()

    if args.print_log:
        print('n_task: ',args.total_task)
        print('class_per_task: ',args.class_per_task)
        print('finetune: ',args.finetune)


    life_experience(model, continuum, x_te, args)