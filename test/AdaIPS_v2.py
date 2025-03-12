import torch
import torch.optim as optim


class AdaIPS_S(optim.Optimizer):
    def __init__(self, model_params, T, lower_bound, beta_1, beta_2, eps=1e-8):
        defaults = dict(lower_bound=lower_bound)
        super(AdaIPS_S, self).__init__(model_params, defaults)
        
        self.best_loss = float('inf')
        self.best_params = None
        self.T0 = T
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.t = 0  # Step counter for bias correction
        
    @torch.no_grad()
    def step(self, closure=None):
        self.t += 1
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                loss_value = loss.item()
        else:
            raise ValueError("AdaIPS requires closure for loss value")
                
        for group in self.param_groups:
            l_star = group['lower_bound']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                # Initialize momentum and second moment estimates if not present
                if not hasattr(param, 'v_t'):
                    param.v_t = torch.zeros_like(param)
                    
                if not hasattr(param, 'm_t'):
                    param.m_t = torch.zeros_like(param)
                
                grad = param.grad.data
                
                # Update first moment estimate (momentum)
                # first moment estimate holds information regarding gradient trends
                # ex, if gradient is a ball rolling down a hill then mt holds information like velocity, direction etc
                param.m_t = self.beta_1 * param.m_t + (1 - self.beta_1) * grad
                
                # Update second moment estimate
                param.v_t = self.beta_2 * param.v_t + (1 - self.beta_2) * grad ** 2
                
                # Bias correction for both estimates
                m_t_hat = param.m_t / (1 - self.beta_1 ** self.t)
                v_t_hat = param.v_t / (1 - self.beta_2 ** self.t)
                
                # Calculate parameter-wise adaptive T_t
                # high vt means gradient has been fluctuating, move slowly
                # v_t_hat shows gradient variance (low vt means high confidence, travelling in this direction reduces loss and vice versa)
                # high variance means unstable region
                # inverse relation, so to prevent overshooting, for large gradient variance small steps
                # ex, if gradient is a ball rolling down a hill then vt represents terrain difficulty
                
                param_T_t = self.T0 / (v_t_hat.sum().sqrt() + self.eps)
                
                grad_norm_sq = torch.sum(grad ** 2).clamp(min=self.eps)
                
                inexact_step_size = (loss_value - l_star) / (grad_norm_sq * (param_T_t ** 0.5))
                
                # Clamp step size to prevent instability in early iterations
                inexact_step_size = torch.clamp(torch.tensor(inexact_step_size), min=0.0, max=1.0).item()
                
                # Apply update using momentum direction instead of raw gradient
                param.data.add_(m_t_hat, alpha=-inexact_step_size)
                
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.best_params = [p.clone().detach() for p in self.param_groups[0]['params']]
                
        return loss
    
    def load_best_params(self):
        if self.best_params:
            for param, best_param in zip(self.param_groups[0]['params'], self.best_params):
                param.data.copy_(best_param)