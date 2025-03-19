import torch
import torch.optim as optim

class IPS(optim.Optimizer):
    def __init__(self, model_params, T, lower_bound):
        
        defaults = dict(T=T, lower_bound=lower_bound)
        
        super().__init__(model_params, defaults)
        
        self.best_loss = float('inf')
        self.best_params = None

        
        
    @torch.no_grad()
    def step(self, closure=None):
        

        if closure is not None:
            with torch.enable_grad():
                rtloss = closure()
                loss = rtloss.item()
                
        for group in self.param_groups:
            T = group['T']
            l_star = group['lower_bound']
            grad_norm_sq = self._compute_grad_norm(group['params'])
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad.data
                # grad_norm_sq = torch.sum(grad ** 2) + 1e-8
                
                inexact_step_size = (loss - l_star) / (grad_norm_sq * (T ** 0.5))
                
                # important, since we're using incremental stepsize without clamping gradient explodes
                # especially true in earlier steps
                # inexact_step_size = torch.clamp(inexact_step_size, min=0.0, max=1.0)
                param.data.add_(grad, alpha=-inexact_step_size)
                
        # if loss < self.best_loss:
        #     self.best_loss = loss
        #     self.best_params = [p.clone().detach() for p in self.param_groups[0]['params']]
                
        return rtloss
    
    def _compute_grad_norm(self, params):
        grads = []
        for param in params:
            grads.append(param.grad.view(-1))
            
        grads = torch.cat(grads)
            
        return torch.sum(grads ** 2) + 1e-8
    
    def load_best_params(self):
        if self.best_params:
            for param, best_param in zip(self.param_groups[0]['params'], self.best_params):
                param.data.copy_(best_param)
                