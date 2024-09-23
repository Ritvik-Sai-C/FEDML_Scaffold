from torch.optim import Optimizer

class AlgoOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(AlgoOptimizer, self).__init__(params, defaults)

    def step(self, main_controls, client_controls, closure=None):

        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], main_controls.values(), client_controls.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group["lr"]

        return loss