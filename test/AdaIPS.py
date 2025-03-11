import torch
import torch.optim as optim


class AdaIPS(optim.Optimizer):
    def __init__(self, model_params, T, lower_bound, lambda_t, beta_t):
        
        defaults = dict(lower_bound=lower_bound)
        
        super(AdaIPS, self).__init__(model_params, defaults)
        
        # save best parameters, haven't implemented yet
        self.best_loss = float('inf')
        self.best_params = None
        
        # initial value of T
        self.T0 = T
        
        """
            second moment estimate
            v_t = beta_t * v_t-1 + (1 - beta_t) * grad ** 2
            T = T0 + (lambda_t * v_t)
        """
        self.lambda_t = lambda_t
        self.beta_t = beta_t
        
        
        
    @torch.no_grad()
    def step(self, closure=None):
        
        if closure is not None:
            with torch.enable_grad():
                rtloss = closure()
                loss = rtloss.item()
                
        for group in self.param_groups:
            l_star = group['lower_bound']

            
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                if not hasattr(param, 'v_t'):
                    param.v_t = torch.zeros_like(param)
                
                grad = param.grad.data
                grad_norm_sq = torch.sum(grad ** 2) + 1e-8
                
                param.v_t = self.beta_t * self.v_t + (1 - self.beta_t) * grad_norm_sq

                # Compute adaptive T_t
                T_t = self.T0 + self.lambda_t * param.v_t.sum()
                
                inexact_step_size = (loss - l_star) / (grad_norm_sq * (T_t ** 0.5))
                
                # important, since we're using incremental stepsize without clamping gradient explodes
                # especially true in earlier steps
                inexact_step_size = torch.clamp(inexact_step_size, min=0.0, max=1.0)
                param.data.add_(grad, alpha=-inexact_step_size)
                
        # if loss < self.best_loss:
        #     self.best_loss = loss
        #     self.best_params = [p.clone().detach() for p in self.param_groups[0]['params']]
                
        return rtloss
    
    def load_best_params(self):
        if self.best_params:
            for param, best_param in zip(self.param_groups[0]['params'], self.best_params):
                param.data.copy_(best_param)
                