import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

class DenoisingDiffusionProbabilisticModel(torch.nn.Module):
    def __init__(self,
                 eps_model: torch.nn.Module,
                 T: int,
                 criterion: torch.nn.Module = torch.nn.MSELoss(),
                 context_dim: int = 512,
                 schedule_type: str = 'linear',
                 schedule_k: float = None,
                 schedule_beta_min: float = None,
                 schedule_beta_max: float = None,
                 adaptive_noise_budget_weight: float = 10.0,
                 adaptive_noise_var_weight: float = 1.0,
                 adaptive_noise_sin2_weight: float = 1.0,
                 adaptive_noise_sin2_target: float = 0.5,
                 adaptive_noise_hidden_dim: int = 128,
                 adaptive_noise_enabled: bool = True,
                 adaptive_noise_running_momentum: float = 0.01,
                 adaptive_noise_std_max: float = 0.80) -> None:

        super(DenoisingDiffusionProbabilisticModel, self).__init__()
        self.eps_model = eps_model

        betas = compute_beta_schedule(T, schedule_type,
                                      k=schedule_k, beta_min=schedule_beta_min, beta_max=schedule_beta_max)
        for k, v in precompute_schedule_constants(betas).items():
            self.register_buffer(k, v)

        self.T = T
        self.criterion = criterion
        self.schedule_type = schedule_type
        self.schedule_k = schedule_k

        self.adaptive_noise_enabled = adaptive_noise_enabled
        self.adaptive_noise_budget_weight = adaptive_noise_budget_weight
        self.adaptive_noise_var_weight = adaptive_noise_var_weight

        self.adaptive_noise_sin2_weight = adaptive_noise_sin2_weight
        self.adaptive_noise_sin2_target = adaptive_noise_sin2_target

        self.adaptive_noise_running_momentum = adaptive_noise_running_momentum

        self.adaptive_noise_std_max = adaptive_noise_std_max

        self.noise_adapter = torch.nn.Sequential(
            torch.nn.Linear(context_dim, adaptive_noise_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(adaptive_noise_hidden_dim, 1)
        )

        self.noise_adapter_log_temperature = torch.nn.Parameter(torch.tensor(0.0))

        self.noise_adapter_log_std = torch.nn.Parameter(torch.tensor(math.log(math.sqrt(1.0 / 12.0))))

        init_base_mean = (2.0 / math.pi) * math.acos(2.0 / math.pi)
        self.noise_adapter_logit_mean = torch.nn.Parameter(
            torch.tensor(math.log(init_base_mean / (1.0 - init_base_mean)))
        )

        self.noise_adapter_tanh_bias = torch.nn.Parameter(torch.tensor(0.0))

        self.register_buffer("noise_adapter_running_mean", torch.tensor(0.0))
        self.register_buffer("noise_adapter_running_var", torch.tensor(1.0))
        self.register_buffer("noise_adapter_running_inited", torch.tensor(0, dtype=torch.int32))

        self.last_noise_budget_loss = None
        self.last_noise_var_reg = None
        self.last_noise_sin2_loss = None

        self.last_noise_mean = None
        self.last_noise_std = None
        self.last_noise_base_mean = None
        self.last_noise_id_keep = None
        self.last_noise_sin2 = None

        self.alphas_prev = torch.tensor([1.0]).float()
        self.alphas_prev = torch.cat((self.alphas_prev, self.alpha_bars[:-1]), 0)
        self.alphas_next = torch.tensor([0.0]).float()
        self.alphas_next = torch.cat((self.alpha_bars[1:], self.alphas_next), 0)

    def _prepare_adapter_input(self, context: torch.Tensor) -> torch.Tensor:
        if context.dim() == 2:
            return context
        elif context.dim() == 3:
            return context.mean(dim=1)
        else:
            raise ValueError(f"Unsupported context shape for adaptive noise: {context.shape}")

    def _sample_random_slerp_target_batch(self, embedding: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        r = torch.randn_like(embedding, dtype=torch.float32, device=embedding.device)
        e = F.normalize(embedding.float(), dim=1, eps=eps)
        proj = (r * e).sum(dim=1, keepdim=True)
        r = r - proj * e
        t = F.normalize(r, dim=1, eps=eps).to(dtype=embedding.dtype)
        return t

    def _spherical_interpolation_batch(self, value: torch.Tensor, start: torch.Tensor, target: torch.Tensor):
        if value.dim() == 1:
            value = value.unsqueeze(1)

        start = F.normalize(start, dim=1)
        target = F.normalize(target, dim=1)

        dot = (start * target).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        omega = torch.acos(dot)
        so = torch.sin(omega)
        so = torch.where(so.abs() < 1e-8, torch.full_like(so, 1e-8), so)

        coeff_start = (torch.sin((1.0 - value.squeeze(1)) * omega) / so).unsqueeze(1)
        coeff_target = (torch.sin(value.squeeze(1) * omega) / so).unsqueeze(1)

        res = coeff_start * start + coeff_target * target
        return F.normalize(res, dim=1)

    def _update_running_stats(self, raw: torch.Tensor, eps_value: float = 1e-6):
        """
        raw: [B]
        Updates running mean/var (batch-independent) using EMA.
        """
        batch_mean = raw.mean().detach()
        batch_var = raw.var(unbiased=False).detach()

        if int(self.noise_adapter_running_inited.item()) == 0:
            self.noise_adapter_running_mean.copy_(batch_mean)
            self.noise_adapter_running_var.copy_(batch_var.clamp_min(eps_value))
            self.noise_adapter_running_inited.fill_(1)
            return

        m = self.adaptive_noise_running_momentum
        self.noise_adapter_running_mean.mul_(1.0 - m).add_(m * batch_mean)
        self.noise_adapter_running_var.mul_(1.0 - m).add_(m * batch_var.clamp_min(eps_value))

    def _make_adaptive_noisy_context(self, context: torch.Tensor):
        adapter_input = self._prepare_adapter_input(context)
        raw = self.noise_adapter(adapter_input).squeeze(1)

        temperature = torch.exp(self.noise_adapter_log_temperature).clamp(1e-4, 1e4)

        if self.training:
            self._update_running_stats(raw)

        eps_value = 1e-6
        run_mean = self.noise_adapter_running_mean
        run_std = torch.sqrt(self.noise_adapter_running_var + eps_value)

        delta = (raw - run_mean) / run_std

        tanh_bias = self.noise_adapter_tanh_bias
        z = torch.tanh(delta * temperature + tanh_bias)

        std = torch.exp(self.noise_adapter_log_std).clamp(1e-4, self.adaptive_noise_std_max)

        base_mean = torch.sigmoid(self.noise_adapter_logit_mean)

        base_mean = base_mean.clamp(eps_value, 1.0 - eps_value)

        a = base_mean + std * z
        a = a.clamp(eps_value, 1.0 - eps_value)

        pooled_context = adapter_input
        target = self._sample_random_slerp_target_batch(pooled_context)
        noisy_pooled_context = self._spherical_interpolation_batch(a, pooled_context, target)

        if context.dim() == 2:
            noisy_context = noisy_pooled_context
        else:
            noisy_context = noisy_pooled_context.unsqueeze(1).expand_as(context)

        return noisy_context, a, base_mean

    def forward(self, x0: torch.Tensor, context=None, dropout_mask=None) -> torch.Tensor:
        t = torch.randint(0, self.T, (x0.shape[0],)).to(x0.device)
        eps = torch.randn_like(x0)

        mean = self.sqrt_alpha_bars[t, None, None, None] * x0
        sd = self.sqrt_one_minus_alpha_bars[t, None, None, None]
        x_t = mean + sd * eps

        noise_budget_loss = torch.tensor(0.0, device=x0.device)
        noise_var_reg = torch.tensor(0.0, device=x0.device)
        noise_sin2_loss = torch.tensor(0.0, device=x0.device)

        if self.adaptive_noise_enabled and context is not None:
            context, noise_strength, base_mean = self._make_adaptive_noisy_context(context)

            theta = (0.5 * math.pi) * noise_strength

            id_keep = torch.cos(theta).mean()
            target_id_keep = 2.0 / math.pi
            noise_budget_loss = (id_keep - target_id_keep) ** 2

            sin2 = (torch.sin(theta) ** 2).mean()
            eps_value = 1e-6
            alpha = 8.0
            noise_sin2_loss = 1.0 / (1.0 + alpha * sin2)

            std_a = noise_strength.std(unbiased=False)
            min_std = math.sqrt(1.0 / 12.0)
            noise_var_reg = torch.relu(min_std - std_a) ** 2

            self.last_noise_budget_loss = noise_budget_loss.detach()
            self.last_noise_var_reg = noise_var_reg.detach()
            self.last_noise_sin2_loss = noise_sin2_loss.detach()
            self.last_noise_mean = noise_strength.mean().detach()
            self.last_noise_std = noise_strength.std(unbiased=False).detach()
            self.last_noise_base_mean = base_mean.detach()
            self.last_noise_id_keep = id_keep.detach()
            self.last_noise_sin2 = sin2.detach()
        else:
            z0 = torch.tensor(0.0, device=x0.device)
            self.last_noise_budget_loss = z0
            self.last_noise_var_reg = z0
            self.last_noise_sin2_loss = z0
            self.last_noise_mean = z0
            self.last_noise_std = z0
            self.last_noise_base_mean = z0
            self.last_noise_id_keep = z0
            self.last_noise_sin2 = z0

        pred = self.eps_model(x_t, t, context, dropout_mask)
        diffusion_loss = self.criterion(eps, pred)

        total_loss = diffusion_loss + \
                     self.adaptive_noise_budget_weight * noise_budget_loss + \
                     self.adaptive_noise_sin2_weight * noise_sin2_loss + \
                     self.adaptive_noise_var_weight * noise_var_reg

        return total_loss

    def sample(self, n_samples, size, x_T: torch.Tensor = None, context=None, dropout_mask=None) -> torch.Tensor:
        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()

        self.eval()
        with torch.no_grad():
            pbar = tqdm(reversed(range(0, self.T)), total=self.T)
            pbar.set_description("DDPM Sampling")

            for i in pbar:
                z = torch.randn(n_samples, *size).cuda() if i > 1 else 0
                t = torch.tensor(i).repeat(n_samples).cuda()

                eps = self.eps_model(x_t, t, context, dropout_mask)
                x_t = self.sqrt_alphas_inv[i] * (x_t - eps * self.one_minus_alphas_over_sqrt_one_minus_alpha_bars[i]) + \
                      self.sigmas[i] * z

        self.train()
        return x_t

    def sample_ddim(self, n_samples, size, x_T: torch.Tensor = None, context=None, dropout_mask=None, eta=0,
                    ddim_step=50) -> torch.Tensor:
        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()

        skip = self.T // ddim_step
        print("DDIM Sampling")
        print("skip: %d" % skip)
        self.eval()
        with torch.no_grad():
            for i in reversed(range(0, self.T, skip)):
                t = torch.tensor(i).repeat(n_samples).cuda()
                model_output = self.eps_model(x_t, t, context, dropout_mask)
                prev_timestep = i - skip
                alpha_prod_t = self.alpha_bars[i]
                alpha_prod_t_prev = self.alphas_prev[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0).cuda()
                beta_prod_t = 1 - alpha_prod_t
                beta_prod_t_prev = 1 - alpha_prod_t_prev
                pred_original_sample = (x_t - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

                variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
                std_dev_t = eta * variance ** (0.5)

                model_output = (x_t - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

                pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * model_output

                x_t = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

                if eta > 0:
                    device = model_output.device if torch.is_tensor(model_output) else "cpu"
                    noise = torch.randn(n_samples, *size).cuda()
                    variance = std_dev_t * noise
                    if not torch.is_tensor(model_output):
                        variance = variance.numpy()
                    x_t = x_t + variance

        self.train()

        return x_t

    def sample_classifier_free_guided(self, n_samples, size, context, guidance_scale,
                                      x_T: torch.Tensor = None) -> torch.Tensor:

        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()

        if type(guidance_scale) is float:
            guidance_scale = [guidance_scale] * self.T

        self.eval()
        with torch.no_grad():

            pbar = tqdm(reversed(range(0, self.T)), total=self.T)
            pbar.set_description(f"DDPM Classifier-Free Guided Sampling")

            for i in pbar:
                z = torch.randn(n_samples, *size).cuda() if i > 1 else 0
                t = torch.tensor(i).repeat(n_samples).cuda()

                eps_unconditional = self.eps_model(x_t, t, context=None, dropout_mask=None)
                eps_conditional = self.eps_model(x_t, t, context, dropout_mask=None)

                eps = (1 - guidance_scale[i]) * eps_unconditional + guidance_scale[i] * eps_conditional

                x_t = self.sqrt_alphas_inv[i] * (x_t - eps * self.one_minus_alphas_over_sqrt_one_minus_alpha_bars[i]) + \
                      self.sigmas[i] * z

        self.train()
        return x_t

    def sample_and_get_step_results(self, n_samples, size, x_T: torch.Tensor = None, context=None, dropout_mask=None,
                                    result_steps=None) -> torch.Tensor:
        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()

        result_xs = {0: x_t.clone()}

        self.eval()
        with torch.no_grad():

            pbar = tqdm(reversed(range(0, self.T)), total=self.T)
            pbar.set_description("DDPM Sampling")

            for i in pbar:
                z = torch.randn(n_samples, *size).cuda() if i > 1 else 0
                t = torch.tensor(i).repeat(n_samples).cuda()

                eps = self.eps_model(x_t, t, context, dropout_mask)
                x_t = self.sqrt_alphas_inv[i] * (x_t - eps * self.one_minus_alphas_over_sqrt_one_minus_alpha_bars[i]) + \
                      self.sigmas[i] * z

                if result_steps is not None and i + 1 in result_steps:
                    result_xs[i] = x_t.clone()

        result_xs[self.T] = x_t

        self.train()
        return result_xs

    @staticmethod
    def sample_prior(n_samples, size):
        return torch.randn(n_samples, *size)

def compute_beta_schedule(
        T: int, schedule_type: str = 'linear', k: float = 1.0,
        beta_min: float = None, beta_max: float = None) -> torch.Tensor:
    if schedule_type.lower() == 'linear':
        scale = 1000 / T
        beta_1 = scale * 0.0001
        beta_T = scale * 0.02
        return torch.linspace(beta_1, beta_T, T, dtype=torch.float32)

    elif schedule_type.lower() in ['cosine', 'cosine_warped']:

        s = 0.008
        beta_min = 0.0 if schedule_type.lower() == 'cosine' else beta_min
        k = 1 if schedule_type.lower() == 'cosine' else k

        return betas_for_alpha_bar(
            T, lambda t: math.cos(math.pi / 2 * (t + s) / (1 + s) ** k) ** 2,
            beta_min=beta_min, beta_max=beta_max
        )

    raise NotImplementedError

def betas_for_alpha_bar(T, alpha_bar, beta_min=0.0, beta_max=1.0):
    betas = []
    for i in range(T):
        t1 = i / T
        t2 = (i + 1) / T
        betas.append(min(max(1 - alpha_bar(t2) / alpha_bar(t1), beta_min), beta_max))
    return torch.tensor(betas).float()

def precompute_schedule_constants(betas: torch.Tensor):
    alphas = 1 - betas
    sqrt_alphas_inv = 1 / alphas.sqrt()

    sigmas = betas.sqrt()

    alpha_bars = torch.cumsum(torch.log(alphas), dim=0).exp()
    sqrt_alpha_bars = alpha_bars.sqrt()

    sqrt_one_minus_alpha_bars = (1 - alpha_bars).sqrt()
    one_minus_alphas_over_sqrt_one_minus_alpha_bars = (1 - alphas) / sqrt_one_minus_alpha_bars

    """
    import matplotlib.pyplot as plt
    plt.title("Variance Schedule")
    plt.plot(betas, label="betas")
    plt.plot(alpha_bars, label="alpha_bars")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    """

    return {
        "betas": betas,
        "alphas": alphas,
        "sigmas": sigmas,
        "sqrt_alphas_inv": sqrt_alphas_inv,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "one_minus_alphas_over_sqrt_one_minus_alpha_bars": one_minus_alphas_over_sqrt_one_minus_alpha_bars
    }
