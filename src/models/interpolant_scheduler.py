import torch
import torch.nn as nn
from typing import Dict, Tuple, Union

class InterpolantScheduler(nn.Module):

    supported_schedule_types = ['cosine', 'linear', 'sqrt', 'log_diff']

    def __init__(self, canonical_feat_order: str, schedule_type: Union[str, Dict[str, str]] = 'cosine', params: dict = {}):
        super().__init__()

        self.feats = canonical_feat_order
        self.n_feats = len(self.feats)

        # check that schedule_type is a string or a dictionary
        if not isinstance(schedule_type, (str, dict)):
            raise ValueError('schedule_type must be a string or a dictionary')
        
        # if it is a string, assign the same schedule_type to all features
        if isinstance(schedule_type, str):
            if schedule_type not in self.supported_schedule_types:
                raise ValueError(f'unsupported schedule_type: {schedule_type}')
            self.schedule_dict = {
                feat: schedule_type for feat in self.feats
            }
        else:
            # schedule_type is a dictionary specifying the schedule_type for each feature
            for feat in self.feats:
                if feat not in schedule_type:
                    raise ValueError(f'must specify schedule_type for feature {feat}')

            self.schedule_dict = schedule_type 

        # if schedule_type == 'cosine':
        #     self.alpha_t = self.cosine_alpha_t
        #     self.alpha_t_prime = self.cosine_alpha_t_prime
        # elif schedule_type == 'linear':
        #     self.alpha_t = self.linear_alpha_t
        #     self.alpha_t_prime = self.linear_alpha_t_prime
        # else:
        #     raise NotImplementedError(f'unsupported schedule_type: {schedule_type}')
            

        # for features which have a cosine schedule, check that the parameter "nu" is provided
        for feat, schedule_type in self.schedule_dict.items():
            if schedule_type != 'linear' and feat not in params:
                raise ValueError(f'must specify params for feature {feat} with {schedule_type} schedule')
    
        # get a list of unique schedule types which are used
        self.schedule_types = list(set( self.schedule_dict.values() ))

        # if we are using a cosine schedule, convert all of the cosine_params to torch tensors
        
        self.params = params
        for feat in params:
            params[feat] = torch.tensor(params[feat]).unsqueeze(0)
        
        # save the params as an attribute
        self.params = params

        self.device = None

        self.clamp_t = True

        

    def update_device(self, t):
        if t.device != self.device:
            for key in self.params:
                self.params[key] = self.params[key].to(t.device)
            self.device = t.device

    def interpolant_weights(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the weights for x_0 and x_1 in the interpolation between x_0 and x_1.
        """
        # t has shape (n_timepoints,)
        # returns a tuple of 2 tensors of shape (n_timepoints, n_feats)
        # the tensor at index 0 is the weight for x_0
        # the tensor at index 1 is the weight for x_1

        self.update_device(t)

        alpha_t = self.alpha_t(t)
        weights = (1 - alpha_t, alpha_t)
        return weights
    
    def loss_weights(self, t: torch.Tensor):
        alpha_t = self.alpha_t(t)
        # alpha_t_prime = self.alpha_t_prime(t)
        # weights = alpha_t_prime/(1 - alpha_t + 1e-5)
        weights = alpha_t/(1 - alpha_t)

        # clamp the weights with a minimum of 0.05 and a maximum of 1.5
        weights = torch.clamp(weights, min=0.05, max=1.5)
        return weights
    
    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        self.update_device(t)

        per_feat_alpha = []
        for feat in self.feats:
            schedule_type = self.schedule_dict[feat]
            if schedule_type == 'cosine':
                alpha_t = self.cosine_alpha_t(t, nu=self.params[feat])
            elif schedule_type == 'linear':
                alpha_t = self.linear_alpha_t(t)
            elif schedule_type == 'sqrt':  # 新增：开方差值调度
                alpha_t = self.sqrt_alpha_t(t, power=self.params[feat])
            elif schedule_type == 'log_diff':  # 新增：对数差值调度
                alpha_t = self.log_diff_alpha_t(t, scale=self.params[feat])
            
            per_feat_alpha.append(alpha_t)

        alpha_t = torch.cat(per_feat_alpha, dim=1)
        return alpha_t

    def alpha_t_prime(self, t: torch.Tensor) -> torch.Tensor:
        self.update_device(t)

        per_feat_alpha_prime = []
        for feat in self.feats:
            schedule_type = self.schedule_dict[feat]
            if schedule_type == 'cosine':
                alpha_t_prime = self.cosine_alpha_t_prime(t, nu=self.params[feat])
            elif schedule_type == 'linear':
                alpha_t_prime = self.linear_alpha_t_prime(t)
            elif schedule_type == 'sqrt':  
                alpha_t_prime = self.sqrt_alpha_t_prime(t, power=self.params[feat])
            elif schedule_type == 'log_diff':  
                alpha_t_prime = self.log_diff_alpha_t_prime(t, scale=self.params[feat])
            
            per_feat_alpha_prime.append(alpha_t_prime)

        alpha_t_prime = torch.cat(per_feat_alpha_prime, dim=1)
        return alpha_t_prime


    def sqrt_alpha_t(self, t: torch.Tensor, power: float = 0.5) -> torch.Tensor:
        """
        开方差值调度: alpha_t = t^power
        power < 1: 早期快速变化，晚期缓慢 (适合坐标)
        power > 1: 早期缓慢，晚期快速变化 (适合分类)
        """
        t = t.unsqueeze(-1)
        alpha_t = torch.pow(t, power)
        return alpha_t

    def sqrt_alpha_t_prime(self, t: torch.Tensor, power: float = 0.5) -> torch.Tensor:
        if self.clamp_t:
            t = torch.clamp_(t, min=1e-9)
        
        t = t.unsqueeze(-1)
        alpha_t_prime = power * torch.pow(t, power - 1)
        return alpha_t_prime

    # 新增：对数差值调度  
    def log_diff_alpha_t(self, t: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        对数差值调度: alpha_t = log(1 + scale*t) / log(1 + scale)
        提供平滑的早期变化和稳定的晚期收敛
        """
        t = t.unsqueeze(-1)
        scale_tensor = torch.tensor(scale, device=t.device)
        alpha_t = torch.log(1 + scale_tensor * t) / torch.log(1 + scale_tensor)
        return alpha_t

    def log_diff_alpha_t_prime(self, t: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.clamp_t:
            t = torch.clamp_(t, min=1e-9)
        
        t = t.unsqueeze(-1)
        scale_tensor = torch.tensor(scale, device=t.device)
        denominator = torch.log(1 + scale_tensor)
        alpha_t_prime = (scale_tensor / (1 + scale_tensor * t)) / denominator
        return alpha_t_prime


    def cosine_alpha_t(self, t: torch.Tensor, nu: torch.Tensor) -> Dict[str, torch.Tensor]:
        # t has shape (n_timepoints,)
        # alpha_t has shape (n_timepoints, n_feats) containing the alpha_t for each feature
        t = t.unsqueeze(-1)
        alpha_t = 1 - torch.cos(torch.pi*0.5*torch.pow(t, nu)).square()
        return alpha_t
    
    def cosine_alpha_t_prime(self, t: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:

        if self.clamp_t:
            t = torch.clamp_(t, min=1e-9)

        t = t.unsqueeze(-1)
        sin_input = torch.pi*torch.pow(t, nu)
        alpha_t_prime = torch.pi*0.5*torch.sin(sin_input)*nu*torch.pow(t, nu-1)
        return alpha_t_prime
    
    def linear_alpha_t(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        alpha_t = t.unsqueeze(-1)
        return alpha_t
    
    def linear_alpha_t_prime(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        alpha_t_prime = torch.ones_like(t).unsqueeze(-1)
        return alpha_t_prime
    
    def sigma_t(self, t: torch.Tensor, eta: float) -> torch.Tensor:
        self.update_device(t)
        per_feat_sigma = []

        # 首先计算alpha_t
        alpha_t_values = self.alpha_t(t)
        
        for i, feat in enumerate(self.feats):
            if feat == 'x':
                # 获取该特征的alpha_t
                alpha_t_feat = alpha_t_values[:, i:i+1]
                
                # 基于alpha_t计算噪声，而不是原始t
                # alpha_t从0到1，我们希望在中间点噪声最大，两端为0
                base_sigma = torch.sqrt(torch.clamp(alpha_t_feat * (1. - alpha_t_feat), min=0))
                
                sigma = 0.05 * eta * base_sigma
                sigma = (sigma + 1e-5)
            else:
                sigma = torch.zeros_like(t).unsqueeze(-1)
            
            per_feat_sigma.append(sigma)

        sigma_t = torch.cat(per_feat_sigma, dim=1)
        return sigma_t

    def sigma_t_prime(self, t: torch.Tensor, eta: float) -> torch.Tensor:
        self.update_device(t)
        per_feat_sigma_prime = []

        # 需要alpha_t和alpha_t_prime
        alpha_t_values = self.alpha_t(t)
        alpha_t_prime_values = self.alpha_t_prime(t)
        
        for i, feat in enumerate(self.feats):
            if feat == 'x':
                alpha_t_feat = alpha_t_values[:, i:i+1]
                alpha_t_prime_feat = alpha_t_prime_values[:, i:i+1]
                
                # 使用链式法则：d(sigma)/dt = d(sigma)/d(alpha) * d(alpha)/dt
                inner_expr = torch.clamp(alpha_t_feat * (1. - alpha_t_feat), min=1e-8)
                
                # d(sigma)/d(alpha) = 0.05 * eta * (1 - 2*alpha) / (2 * sqrt(alpha*(1-alpha)))
                numerator = 1. - 2. * alpha_t_feat
                denominator = 2. * torch.sqrt(inner_expr)
                safe_denominator = torch.where(denominator > 1e-8, denominator, torch.ones_like(denominator))
                
                d_sigma_d_alpha = 0.05 * eta * numerator / safe_denominator
                
                # 链式法则：d(sigma)/dt = d(sigma)/d(alpha) * d(alpha)/dt
                sigma_prime = d_sigma_d_alpha * alpha_t_prime_feat
                
            else:
                sigma_prime = torch.zeros_like(t).unsqueeze(-1)
            
            per_feat_sigma_prime.append(sigma_prime)

        sigma_t_prime = torch.cat(per_feat_sigma_prime, dim=1)
        return sigma_t_prime