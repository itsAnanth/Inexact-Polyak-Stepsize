import torch
import torch.optim as optim

class AdaIPS_S(optim.Optimizer):
    def __init__(self, model_params, T, lower_bound, beta_1=0.9, beta_2=0.999, eps=1e-8):
        defaults = dict(T0=T, lower_bound=lower_bound, beta_1=beta_1, beta_2=beta_2, eps=eps)
        super(AdaIPS_S, self).__init__(model_params, defaults)
        
        self.best_loss = float('inf')
        self.best_params = None  # List of lists for each param group
        self.t = 0

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
        
        current_params = []
        for group in self.param_groups:
            current_params.append([p.clone().detach() for p in group['params']])
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.best_params = current_params
                
        for group in self.param_groups:
            l_star = group['lower_bound']
            T0 = group['T0']
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            eps = group['eps']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad
                
                state = self.state[param]
                
                # Initialize state
                if len(state) == 0:
                    state['m_t'] = torch.zeros_like(param)
                    state['v_t'] = torch.zeros_like(param)
                
                
                m_t = state['m_t']
                v_t = state['v_t']
                
                # Update moments
                
                # Update first moment estimate (momentum)
                # first moment estimate holds information regarding gradient trends
                # ex, if gradient is a ball rolling down a hill then mt holds information like velocity, direction etc
                m_t.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                
                # Calculate parameter-wise adaptive T_t
                # high vt means gradient has been fluctuating, move slowly
                # v_t_hat shows gradient variance (low vt means high confidence, travelling in this direction reduces loss and vice versa)
                # high variance means unstable region
                # inverse relation, so to prevent overshooting, for large gradient variance small steps
                # ex, if gradient is a ball rolling down a hill then vt represents terrain difficulty
                v_t.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)
                
                
                # bias correction
                m_t_hat = m_t / (1 - beta_1 ** self.t)
                v_t_hat = v_t / (1 - beta_2 ** self.t)
                
                sum_v_t_hat = v_t_hat.sum()
                grad_norm_sq = grad.pow(2).sum().clamp(min=eps)
                
                param_t = (T0 ** 0.5) * (torch.sqrt(sum_v_t_hat) + eps)
                denominator = grad_norm_sq * torch.sqrt(param_t)
                denominator = denominator.clamp(min=eps)
                
                step_size = (loss_value - l_star) / denominator
                step_size = torch.clamp(step_size, min=0.0, max=1.0)
                
                param.data.add_(m_t_hat, alpha=-step_size)
        
        return loss
    
    def load_best_params(self):
        if self.best_params is not None:
            for group, best_group_params in zip(self.param_groups, self.best_params):
                for param, best_param in zip(group['params'], best_group_params):
                    param.data.copy_(best_param)