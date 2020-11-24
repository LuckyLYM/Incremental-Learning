import torch
from .common import MLP, ResNet18
import pretrained_cifar

class Net(torch.nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens

        if 'mnist' in args.data_file:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])
        else:
            if args.finetune=='None':
                self.net = ResNet18(n_outputs, bias=args.bias)
            else:
                self.net = pretrained_cifar.cifar_resnet20(pretrained=args.finetune)

        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)
        self.ce = torch.nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.n_iter=args.n_iter

    def forward(self, x, t=0):
        output = self.net(x)
        return output

    def observe(self, x, t, y, pretrained=None):
        self.train()
        for _ in range(self.n_iter):
            self.zero_grad()
            loss = self.ce(self.forward(x), y)
            loss.backward()
            self.opt.step()

    def feature_and_output(self,x):
        return self.net.feature_and_output(x)