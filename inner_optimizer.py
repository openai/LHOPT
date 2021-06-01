import numpy as np
import scipy.stats
import torch

from . import inner_features


def scale_grads(parameters, coef, grad_norms):
    """
    copied from torch.nn.utils.clip_grad_norm_
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    for p in parameters:
        p.grad.data.mul_(coef)

    return [
        grad_norm * coef if grad_norm is not None else None for grad_norm in grad_norms
    ]


def update_bias_correction(state, key, beta):
    prev = state[key]
    curr = prev * beta + 1 - beta
    state[key] = curr
    return curr


def initialize_ema(state, key, tensor_like=None):
    ema_key = key + "_ema"
    ema_bias_correction_key = key + "_ema_bias_correction"
    if tensor_like is None:
        state[ema_key] = 0.0
    else:
        state[ema_key] = torch.zeros_like(tensor_like)
    state[ema_bias_correction_key] = 0.0


def get_ema(state, key):
    ema_key = key + "_ema"
    ema_bias_correction_key = key + "_ema_bias_correction"
    return state[ema_key] / state[ema_bias_correction_key]


def update_ema(
    state, key, beta, value, is_tensor=False, should_return=False, square_value=False,
):
    ema_key = key + "_ema"
    ema_bias_correction_key = key + "_ema_bias_correction"
    if is_tensor:
        if not square_value:
            state[ema_key].mul_(beta).add_(1 - beta, value)
        else:
            # optimization
            state[ema_key].mul_(beta).addcmul_(1 - beta, value, value)
    else:
        assert not square_value
        prev = state[ema_key]
        state[ema_key] = beta * prev + (1 - beta) * value

    update_bias_correction(state, ema_bias_correction_key, beta)

    if should_return:
        return state[ema_key] / state[ema_bias_correction_key]


def initialize_ema_emvar(
    state, key, tensor_like=None,
):
    ema_key = key + "_ema"
    ema_bias_correction_key = key + "_ema_bias_correction"
    emvar_key = key + "_emvar"
    # NOTE: using same bias correction for emvar as ema
    if tensor_like is None:
        state[ema_key] = 0.0
        state[emvar_key] = 0.0
    else:
        state[ema_key] = torch.zeros_like(tensor_like)
        state[emvar_key] = torch.zeros_like(tensor_like)
    state[ema_bias_correction_key] = 0.0


def update_ema_emvar(state, key, beta, value, should_return=False):
    ema_key = key + "_ema"
    ema_bias_correction_key = key + "_ema_bias_correction"
    emvar_key = key + "_emvar"

    ombeta = 1 - beta
    if not torch.is_tensor(value):
        # scalar input
        state[ema_key] = beta * state[ema_key] + ombeta * value
    else:
        state[ema_key] = beta * state[ema_key] + ombeta * value.mean().item()

    bias_correction = update_bias_correction(state, ema_bias_correction_key, beta)
    new_ema_unbiased = state[ema_key] / bias_correction

    """
    https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
    problem:
      equation (142) uses previous mu (mu[n-1])
      this is undefined when we unbias mu
    want:
      represent mu[n-1] as a function of mu[n], x[n]
    given:
      mu[n] = mu[n-1] + alpha * (x[n] - mu[n-1])
      mu[n] = (1 - alpha) * mu[n-1] + alpha * x[n]

    x[n] - mu[n] = (alpha - 1) * mu[n-1] + (1 - alpha) * x[n]
    x[n] - mu[n] = (1 - alpha) * (x[n] - mu[n-1])

    thus:
    (x[n] - mu[n]) * (x[n] - mu[n-1]) = (x[n] - mu[n])^2 / (1 - alpha)

    this actually results in an incorrect estimate for alpha = 1, so we just remove the
    1 - alpha term
    """
    diff = value - new_ema_unbiased
    diff_sq = diff * diff
    if torch.is_tensor(value):
        diff_sq = diff_sq.mean().item()
    state[emvar_key] = beta * state[emvar_key] + ombeta * diff_sq

    if should_return:
        return new_ema_unbiased, (state[emvar_key] / bias_correction)


# NOTEL didn't see speedup jit-ing this, but did see speedup by re-ordering
# calculations and .item() calls
# @torch.jit.script
def calculate_features(
    m_hat: torch.Tensor,
    grad: torch.Tensor,
    sqrt_v_hat: torch.Tensor,
    epsilon: float,
    adam_pre_lr_update: torch.Tensor,
    p: torch.Tensor,
):
    m_hat_norm = m_hat.norm()
    m_hat_sub_grad_norm = (m_hat - grad).norm()
    sqrt_v_hat_gt_epsilon = (sqrt_v_hat > epsilon).to(sqrt_v_hat.dtype).mean()
    mean_log_abs_adam_pre_lr_update = torch.log(abs(adam_pre_lr_update) + 1e-8).mean()
    grad_dot_p = grad.flatten().dot(p.flatten())
    grad_dot_m_hat = grad.flatten().dot(m_hat.flatten())
    grad_dot_adam_pre_lr_update = grad.flatten().dot(adam_pre_lr_update.flatten())
    return {
        "m_hat_norm": m_hat_norm.item(),
        "m_hat_sub_grad_norm": m_hat_sub_grad_norm.item(),
        "sqrt_v_hat_gt_epsilon": sqrt_v_hat_gt_epsilon.item(),
        "mean_log_abs_adam_pre_lr_update": mean_log_abs_adam_pre_lr_update.item(),
        "grad_dot_p": grad_dot_p.item(),
        "grad_dot_m_hat": grad_dot_m_hat.item(),
        "grad_dot_adam_pre_lr_update": grad_dot_adam_pre_lr_update.item(),
    }


class CustomizableInnerAdaptiveOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        param_names=None,
        inner_step_features=None,
        # set defaults same as AdamW
        lr=1e-3,
        ombeta1=0.1,
        ombeta2=1e-3,
        epsilon=1e-8,
        weight_decay=1e-2,
        group_clip_grad_norm=None,
        scale_grad_norm_by_param_norm=False,
        scale_grad_norm_by_rolling_max=False,
        approximate_radam_warmup=False,
        lr_vector_multiplier=1,
        weight_decay_vector_multiplier=1,
        gradient_centralization=None,
        lamb_power=None,
        lamb_min_trust=0,
        lamb_max_trust=np.inf,
        lookahead_alpha=None,
        lookahead_k=5,
        qhadam_nu1=None,
        adaptive_scale_type=None,
        adaptive_scale_power=None,
        acclip_pre_lr_update_clip=None,
        nvlamb_gradient_pre_normalization=None,
        denominator_norm=None,
        inner_feature_frequency=None,
        adam_update_include_weight_decay=False,
        grad_clip_ombeta=None,
        adabelief_denominator=False,
        lamb_ignore_vectors=False,
        lamb_ignore_vectors_lr=None,
        lamb_ignore_vectors_weight_decay=None,
        tensor_geometric_mean_auto_balance=None,
        hypergradient_type=None,
        hypergradient_lr_lr=None,
        hypergradient_previous_update_ombeta=None,
        hypergradient_rms_ombeta=None,
        hypergradient_decay=None,
        hypergradient_soft_clip=None,
        hypergradient_update_type=None,
        lamb_update_norm_ombeta=None,
        auto_epsilon_z_score=None,
        auto_epsilon_ombeta=None,
        auto_epsilon_log_scale=False,
        group_clip_grad_z_score=None,
    ):
        assert 0.0 <= lr
        assert 0.0 <= epsilon
        assert 0 < ombeta1 <= 1.0
        assert 0 < ombeta2 <= 1.0
        assert group_clip_grad_norm is None or group_clip_grad_norm > 0
        assert group_clip_grad_norm is None or group_clip_grad_z_score is None

        if scale_grad_norm_by_param_norm:
            assert group_clip_grad_norm is not None
            assert not scale_grad_norm_by_rolling_max
            # assumes a single param group
            params = [p for p in params if p.requires_grad]
            with torch.no_grad():
                total_norm = 0
                norm_type = 2
                for p in params:
                    param_norm = p.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
                total_norm = total_norm ** (1.0 / norm_type)
            group_clip_grad_norm *= total_norm

        if scale_grad_norm_by_rolling_max:
            assert group_clip_grad_norm is not None
            assert not scale_grad_norm_by_param_norm

        defaults = dict(
            lr=lr,
            ombeta1=ombeta1,
            ombeta2=ombeta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            group_clip_grad_norm=group_clip_grad_norm,
            approximate_radam_warmup=approximate_radam_warmup,
            scale_grad_norm_by_rolling_max=scale_grad_norm_by_rolling_max,
            lr_vector_multiplier=lr_vector_multiplier,
            weight_decay_vector_multiplier=weight_decay_vector_multiplier,
            gradient_centralization=gradient_centralization,
            lamb_power=lamb_power,
            lamb_min_trust=lamb_min_trust,
            lamb_max_trust=lamb_max_trust,
            lookahead_alpha=lookahead_alpha,
            lookahead_k=lookahead_k,
            qhadam_nu1=qhadam_nu1,
            adaptive_scale_type=adaptive_scale_type,
            adaptive_scale_power=adaptive_scale_power,
            acclip_pre_lr_update_clip=acclip_pre_lr_update_clip,
            nvlamb_gradient_pre_normalization=nvlamb_gradient_pre_normalization,
            denominator_norm=denominator_norm,
            inner_feature_frequency=inner_feature_frequency,
            adam_update_include_weight_decay=adam_update_include_weight_decay,
            grad_clip_ombeta=grad_clip_ombeta,
            adabelief_denominator=adabelief_denominator,
            lamb_ignore_vectors=lamb_ignore_vectors,
            lamb_ignore_vectors_lr=lamb_ignore_vectors_lr,
            lamb_ignore_vectors_weight_decay=lamb_ignore_vectors_weight_decay,
            tensor_geometric_mean_auto_balance=tensor_geometric_mean_auto_balance,
            hypergradient_type=hypergradient_type,
            hypergradient_lr_lr=hypergradient_lr_lr,
            hypergradient_previous_update_ombeta=hypergradient_previous_update_ombeta,
            hypergradient_rms_ombeta=hypergradient_rms_ombeta,
            hypergradient_decay=hypergradient_decay,
            hypergradient_soft_clip=hypergradient_soft_clip,
            hypergradient_update_type=hypergradient_update_type,
            lamb_update_norm_ombeta=lamb_update_norm_ombeta,
            auto_epsilon_z_score=auto_epsilon_z_score,
            auto_epsilon_ombeta=auto_epsilon_ombeta,
            auto_epsilon_log_scale=auto_epsilon_log_scale,
            group_clip_grad_z_score=group_clip_grad_z_score,
        )
        super().__init__(params, defaults)

        if inner_step_features is None:
            inner_step_features = inner_features.InnerStepFeatures()
        self.inner_step_features = inner_step_features

        self.param_names = param_names
        if self.param_names is not None:
            self.param_to_name = {p: name for name, p in zip(self.param_names, params)}

        self.inner_step_count = 0

    def outer_step(self):
        self.inner_step_features.outer_step_aggregate()
        self.inner_step_count = 0

    def _init_param_state(self, p, group):
        state = self.state[p]
        # state initialization
        if len(state) == 0:
            state["step"] = 0
            # exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(p.data)
            # exponential moving average of squared gradient values
            state["exp_avg_sq"] = torch.zeros_like(p.data)
            # store bias correction terms because beta might change
            state["bias_correction1"] = 0.0
            state["bias_correction2"] = 0.0

            if group["lookahead_alpha"] is not None:
                state["lookahead_cache"] = torch.zeros_like(p.data)
                state["lookahead_cache"].copy_(p.data)

            hypergradient_type = group["hypergradient_type"]
            if hypergradient_type is not None:
                # keep previous state
                initialize_ema(state, "previous_update", tensor_like=p.data)

                if hypergradient_type == "parameter":
                    state["hypergradient_log_lr"] = torch.zeros_like(p.data)
                    if group["hypergradient_rms_ombeta"] is not None:
                        initialize_ema(state, "hypergradient_sq", tensor_like=p.data)
                else:
                    raise ValueError

            is_vector_or_scalar = p.dim() <= 1
            state["is_vector_or_scalar"] = is_vector_or_scalar
            should_apply_lamb = group["lamb_power"] is not None and not (
                group["lamb_ignore_vectors"] and is_vector_or_scalar
            )
            state["should_apply_lamb"] = should_apply_lamb
            if should_apply_lamb and group["lamb_update_norm_ombeta"] is not None:
                initialize_ema(state, "lamb_update_norm")

            if group["auto_epsilon_z_score"] is not None:
                initialize_ema_emvar(state, "auto_epsilon_stats")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                beta1 = 1 - group["ombeta1"]
                assert 0 <= beta1 < 1.0, beta1
                beta2 = 1 - group["ombeta2"]
                assert 0 <= beta2 < 1.0, beta2
                epsilon = group["epsilon"]
                assert epsilon > 0
                group_clip_grad_norm = group["group_clip_grad_norm"]
                assert group_clip_grad_norm is None or group_clip_grad_norm > 0
                gradient_centralization = group["gradient_centralization"]
                assert (
                    gradient_centralization is None or 0 <= gradient_centralization <= 1
                )
                lamb_power = group["lamb_power"]
                assert lamb_power is None or 0 <= lamb_power <= 1
                lamb_min_trust = group["lamb_min_trust"]
                lamb_max_trust = group["lamb_max_trust"]
                assert 0 <= lamb_min_trust <= lamb_max_trust
                lookahead_alpha = group["lookahead_alpha"]
                assert lookahead_alpha is None or 0 <= lookahead_alpha <= 1
                lookahead_k = group["lookahead_k"]
                assert isinstance(lookahead_k, int) and lookahead_k > 1
                qhadam_nu1 = group["qhadam_nu1"]
                assert qhadam_nu1 is None or 0 <= qhadam_nu1 <= 1
                adaptive_scale_type = group["adaptive_scale_type"]
                assert adaptive_scale_type in {None, "kaiming_init"}
                adaptive_scale_power = group["adaptive_scale_power"]
                assert adaptive_scale_power is None or 0 <= adaptive_scale_power <= 1
                acclip_pre_lr_update_clip = group["acclip_pre_lr_update_clip"]
                assert (
                    acclip_pre_lr_update_clip is None or acclip_pre_lr_update_clip > 0
                )
                nvlamb_gradient_pre_normalization = group[
                    "nvlamb_gradient_pre_normalization"
                ]
                # both can't be set
                assert (
                    group_clip_grad_norm is None
                    or nvlamb_gradient_pre_normalization is None
                )
                denominator_norm = group["denominator_norm"]
                assert (
                    denominator_norm is None
                    or denominator_norm == "inf"
                    or denominator_norm > 0
                )
                inner_feature_frequency = group["inner_feature_frequency"]
                adam_update_include_weight_decay = group[
                    "adam_update_include_weight_decay"
                ]
                grad_clip_ombeta = group["grad_clip_ombeta"]
                if grad_clip_ombeta is None:
                    grad_clip_beta = beta2
                else:
                    grad_clip_beta = 1 - grad_clip_ombeta
                assert 0 <= grad_clip_beta < 1.0, grad_clip_beta
                adabelief_denominator = group["adabelief_denominator"]
                lamb_ignore_vectors = group["lamb_ignore_vectors"]
                lamb_ignore_vectors_lr = group["lamb_ignore_vectors_lr"]
                lamb_ignore_vectors_weight_decay = group[
                    "lamb_ignore_vectors_weight_decay"
                ]
                tensor_geometric_mean_auto_balance = group[
                    "tensor_geometric_mean_auto_balance"
                ]
                assert (
                    tensor_geometric_mean_auto_balance is None
                    or 0 <= tensor_geometric_mean_auto_balance <= 1
                )
                lamb_update_norm_ombeta = group["lamb_update_norm_ombeta"]
                assert (
                    lamb_update_norm_ombeta is None or 0 <= lamb_update_norm_ombeta <= 1
                )
                auto_epsilon_z_score = group["auto_epsilon_z_score"]
                auto_epsilon_ombeta = group["auto_epsilon_ombeta"]
                auto_epsilon_log_scale = group["auto_epsilon_log_scale"]
                if auto_epsilon_z_score is not None:
                    assert auto_epsilon_ombeta is not None
                    assert 0 < auto_epsilon_ombeta <= 1
                group_clip_grad_z_score = group["group_clip_grad_z_score"]

                if inner_feature_frequency is None:
                    # NOTE: defaulting this to be true for backward compatibility
                    should_calculate_features = True
                elif inner_feature_frequency == 0:
                    should_calculate_features = False
                else:
                    should_calculate_features = (
                        self.inner_step_count % inner_feature_frequency
                    ) == 0

                # initialize param state
                for p in group["params"]:
                    if p.grad is not None:
                        self._init_param_state(p, group)

                grad_norms = [
                    p.grad.norm().item() if p.grad is not None else None
                    for p in group["params"]
                ]
                # gradient clipping
                if (
                    group_clip_grad_norm is not None
                    or nvlamb_gradient_pre_normalization is not None
                    or group_clip_grad_z_score is not None
                ):
                    grad_norm = np.sqrt(
                        sum(
                            [
                                grad_norm ** 2
                                for grad_norm in grad_norms
                                if grad_norm is not None
                            ]
                        )
                    )
                    if (
                        group_clip_grad_norm is not None
                        or group_clip_grad_z_score is not None
                    ):
                        if group_clip_grad_norm is not None:
                            clip = group_clip_grad_norm

                            if group["scale_grad_norm_by_rolling_max"]:
                                rolling_max = group.get("_grad_rolling_max", 0)
                                # reuse beta2 for rolling max
                                rolling_max = max(
                                    grad_norm, rolling_max * grad_clip_beta
                                )
                                group["_grad_rolling_max"] = rolling_max

                                # here clip is a ratio of the rolling max
                                clip = rolling_max * clip
                        elif group_clip_grad_z_score is not None:
                            # initialize state
                            if "_log_group_grad_norm_ema" not in group:
                                initialize_ema_emvar(group, "_log_group_grad_norm")
                                # make sure we are checking the right key
                                assert "_log_group_grad_norm_ema" in group

                            log_grad_norm_ema, log_grad_norm_emvar = update_ema_emvar(
                                state=group,
                                key="_log_group_grad_norm",
                                beta=grad_clip_beta,
                                value=np.log(grad_norm + 1e-15),
                                should_return=True,
                            )
                            clip = np.exp(
                                log_grad_norm_ema
                                + group_clip_grad_z_score * np.sqrt(log_grad_norm_emvar)
                            )
                        else:
                            raise NotImplementedError

                        clip_coef = float(clip) / (grad_norm + 1e-6)
                        if clip_coef < 1:
                            grad_norms = scale_grads(
                                group["params"], clip_coef, grad_norms
                            )

                        if should_calculate_features:
                            # NOTE: this is extremely cheap
                            self.inner_step_features.record_per_param_group(
                                "did_grad_clip", clip_coef < 1
                            )

                    elif nvlamb_gradient_pre_normalization is not None:
                        nvlamb_scale = 1 / (grad_norm + 1e-8)
                        grad_norms = scale_grads(
                            group["params"], nvlamb_scale, grad_norms
                        )
                    else:
                        raise NotImplementedError

                # hypergradient
                hypergradient_type = group["hypergradient_type"]
                hypergradient_lr_lr = group["hypergradient_lr_lr"]
                hypergradient_previous_update_ombeta = group[
                    "hypergradient_previous_update_ombeta"
                ]
                hypergradient_rms_ombeta = group["hypergradient_rms_ombeta"]
                hypergradient_decay = group["hypergradient_decay"]
                hypergradient_soft_clip = group["hypergradient_soft_clip"]
                hypergradient_update_type = group["hypergradient_update_type"]
                if hypergradient_type is None:
                    assert hypergradient_lr_lr is None
                    assert hypergradient_previous_update_ombeta is None
                    assert hypergradient_rms_ombeta is None
                    assert hypergradient_decay is None
                    assert hypergradient_soft_clip is None
                    assert hypergradient_update_type is None
                else:
                    assert hypergradient_type in ["parameter"]
                    assert hypergradient_lr_lr > 0
                    assert (
                        hypergradient_previous_update_ombeta is None
                        or 0 <= hypergradient_previous_update_ombeta < 1
                    )
                    assert (
                        hypergradient_rms_ombeta is None
                        or 0 < hypergradient_rms_ombeta < 1
                    )
                    assert hypergradient_decay is None or hypergradient_decay > 0
                    assert (
                        hypergradient_soft_clip is None or hypergradient_soft_clip > 0
                    )
                    assert hypergradient_update_type in [
                        "multiplicative",
                        "multiplicative_logscale",
                    ]

                    for p in group["params"]:
                        if p.grad is None:
                            continue

                        state = self.state[p]

                        if state["previous_update_ema_bias_correction"] == 0.0:
                            # if there's no state - don't do anything
                            break

                        hypergradient_log_lr = state["hypergradient_log_lr"]
                        previous_update = get_ema(state, "previous_update")

                        hypergradient = p.grad * previous_update

                        if hypergradient_type == "parameter":
                            # do nothing - hypergradient is already per-parameter
                            pass
                        else:
                            raise ValueError

                        if hypergradient_rms_ombeta is None:
                            normalized_hypergradient = hypergradient
                        else:
                            hypergradient_sq_ema = update_ema(
                                state=state,
                                key="hypergradient_sq",
                                beta=1 - hypergradient_rms_ombeta,
                                value=hypergradient,
                                is_tensor=True,
                                should_return=True,
                                square_value=True,
                            )
                            normalized_hypergradient = hypergradient / (
                                torch.sqrt(hypergradient_sq_ema) + epsilon
                            )

                        if hypergradient_update_type == "multiplicative":
                            hypergradient_log_lr.add_(
                                torch.log(
                                    1 + hypergradient_lr_lr * normalized_hypergradient
                                )
                            )
                        elif hypergradient_update_type == "multiplicative_logscale":
                            hypergradient_log_lr.add_(
                                hypergradient_lr_lr, normalized_hypergradient
                            )
                        else:
                            raise ValueError(hypergradient_update_type)

                        if hypergradient_decay is not None:
                            hypergradient_log_lr.mul_(
                                1 - hypergradient_lr_lr * hypergradient_decay
                            )

                        if hypergradient_soft_clip is None:
                            hypergradient_log_lr.copy_(
                                torch.tanh(
                                    hypergradient_log_lr / hypergradient_soft_clip
                                )
                                * hypergradient_soft_clip
                            )

                    else:
                        # hypergradient post-processing
                        pass

                for p, grad_norm in zip(group["params"], grad_norms):
                    if p.grad is None:
                        continue

                    if self.param_names is None:
                        param_name = None
                    else:
                        # for debugging
                        param_name = self.param_to_name[p]

                    grad = p.grad.data
                    assert not grad.is_sparse
                    state = self.state[p]

                    is_vector_or_scalar = state["is_vector_or_scalar"]
                    should_apply_lamb = state["should_apply_lamb"]

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                    state["step"] += 1
                    bias_correction1 = update_bias_correction(
                        state, "bias_correction1", beta1
                    )
                    bias_correction2 = update_bias_correction(
                        state, "bias_correction2", beta2
                    )

                    # gradient centralization
                    if gradient_centralization is not None and not is_vector_or_scalar:
                        grad.add_(
                            gradient_centralization
                            * -grad.mean(
                                dim=tuple(range(1, len(list(grad.size())))),
                                keepdim=True,
                            )
                        )

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)

                    denom_grad = grad

                    if adabelief_denominator:
                        # use grad residual instead
                        denom_grad = grad - exp_avg

                    if denominator_norm is None:
                        exp_avg_sq.mul_(beta2).addcmul_(
                            1 - beta2, denom_grad, denom_grad
                        )
                        sqrt_v_hat = exp_avg_sq.sqrt() / np.sqrt(bias_correction2)
                    elif denominator_norm == "inf":
                        # from AdaMax implementation
                        norm_buf = torch.cat(
                            [
                                exp_avg_sq.mul_(beta2).unsqueeze(0),
                                denom_grad.abs().unsqueeze_(0),
                            ],
                            0,
                        )
                        torch.max(
                            norm_buf,
                            0,
                            keepdim=False,
                            out=(exp_avg_sq, exp_avg_sq.new().long()),
                        )
                        sqrt_v_hat = exp_avg_sq
                    else:
                        # ACClip update for denominator_norm == 1
                        exp_avg_sq.mul_(beta2).add_(
                            1 - beta2, abs(denom_grad) ** denominator_norm
                        )
                        sqrt_v_hat = (exp_avg_sq / bias_correction2) ** (
                            1 / denominator_norm
                        )
                    # need to compute sqrt_v_hat term separately for
                    # inner feature computation
                    denom = sqrt_v_hat + epsilon

                    if auto_epsilon_z_score is not None:
                        auto_epsilon_stats = sqrt_v_hat
                        if auto_epsilon_log_scale:
                            auto_epsilon_stats = torch.log(auto_epsilon_stats + 1e-8)
                        # update ema and emvar
                        sqrt_v_hat_ema, sqrt_v_hat_emvar = update_ema_emvar(
                            state=state,
                            key="auto_epsilon_stats",
                            beta=1 - auto_epsilon_ombeta,
                            value=auto_epsilon_stats,
                            should_return=True,
                        )
                        auto_epsilon = sqrt_v_hat_ema + auto_epsilon_z_score * np.sqrt(
                            sqrt_v_hat_emvar
                        )
                        if auto_epsilon_log_scale:
                            auto_epsilon = np.exp(auto_epsilon)
                        denom += auto_epsilon

                    if qhadam_nu1 is None:
                        numerator = exp_avg
                    else:
                        numerator = qhadam_nu1 * exp_avg + (1 - qhadam_nu1) * grad

                    m_hat = exp_avg / bias_correction1
                    adam_pre_lr_update = numerator / denom / bias_correction1

                    if acclip_pre_lr_update_clip is not None:
                        adam_pre_lr_update.clamp_(
                            -acclip_pre_lr_update_clip, acclip_pre_lr_update_clip
                        )

                    if tensor_geometric_mean_auto_balance is not None:
                        geometric_mean = torch.exp(
                            torch.log(abs(adam_pre_lr_update)).mean()
                        )
                        adam_pre_lr_update.mul_(
                            geometric_mean ** tensor_geometric_mean_auto_balance
                        )

                    if hypergradient_type is not None:
                        # implementation note: our hypergradient doesn't take the effect
                        # of weight decay on the gradient into account, due to
                        # AdamW-style weight decay not directly relating to a gradient
                        # value, thus we take the previous update BEFORE weight decay to
                        # keep the hypergradient unbiased
                        if hypergradient_previous_update_ombeta is None:
                            state["previous_update_ema"].copy_(adam_pre_lr_update)
                            state["previous_update_ema_bias_correction"] = 1.0
                        else:
                            update_ema(
                                state=state,
                                key="previous_update",
                                beta=1 - hypergradient_previous_update_ombeta,
                                value=adam_pre_lr_update,
                                is_tensor=True,
                                should_return=False,
                            )

                    weight_decay = group["weight_decay"]

                    if is_vector_or_scalar:
                        weight_decay *= group["weight_decay_vector_multiplier"]

                        if (
                            lamb_ignore_vectors
                            and lamb_ignore_vectors_weight_decay is not None
                        ):
                            weight_decay = lamb_ignore_vectors_weight_decay

                    if adam_update_include_weight_decay:
                        adam_pre_lr_update.add_(weight_decay, p.data)

                    lr = group["lr"]

                    if is_vector_or_scalar:
                        lr *= group["lr_vector_multiplier"]

                        if lamb_ignore_vectors and lamb_ignore_vectors_lr is not None:
                            lr = lamb_ignore_vectors_lr

                    if adaptive_scale_type == "kaiming_init":
                        if not is_vector_or_scalar:
                            if "adaptive_scale" not in state:
                                # caching - this assumes that adaptive_scale_type
                                # doesn't change throughout training
                                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                                    p.data
                                )
                                state["adaptive_scale"] = 1 / np.sqrt(fan_in)
                            adaptive_scale = state["adaptive_scale"]
                            if adaptive_scale_power is not None:
                                adaptive_scale = adaptive_scale ** adaptive_scale_power
                            lr *= adaptive_scale

                    if group["approximate_radam_warmup"]:
                        warmup_lr = np.sqrt(2 / (2 - bias_correction2) - 1)
                        lr = warmup_lr * lr

                    p_norm = p.norm().item()
                    adam_pre_lr_update_norm = adam_pre_lr_update.norm().item()

                    if should_apply_lamb:
                        lamb_update_norm = adam_pre_lr_update_norm

                        if lamb_update_norm_ombeta is not None:
                            lamb_update_norm = update_ema(
                                state=state,
                                key="lamb_update_norm",
                                beta=1 - lamb_update_norm_ombeta,
                                value=lamb_update_norm,
                                should_return=True,
                            )

                        if p_norm >= 1e-8 and lamb_update_norm >= 1e-8:
                            # NOTE: LAMB v2 and onwards applies clipping on p_norm
                            # instead of on trust ratio
                            trust_ratio = p_norm / lamb_update_norm
                            trust_ratio = np.clip(
                                trust_ratio, lamb_min_trust, lamb_max_trust
                            )
                            trust_ratio = trust_ratio ** lamb_power
                            lr = trust_ratio * lr

                    adam_update = -lr * adam_pre_lr_update

                    if not adam_update_include_weight_decay:
                        # AdamW-style weight decay
                        p.data.mul_(1 - lr * weight_decay)

                    # perform Adam update
                    p.data.add_(adam_update)

                    # perform lookahead update
                    if lookahead_alpha is not None and state["step"] % lookahead_k == 0:
                        p.data.mul_(lookahead_alpha).add_(
                            1.0 - lookahead_alpha, state["lookahead_cache"]
                        )
                        state["lookahead_cache"].copy_(p.data)

                    # ------------------------
                    # feature calculation only
                    # ------------------------

                    if not should_calculate_features:
                        self.inner_step_features.param_aggregate()
                        continue

                    def _cosine_similarity(dot_prod, a_norm, b_norm, epsilon=1e-8):
                        denom = max(a_norm * b_norm, epsilon)
                        return dot_prod / denom

                    def _cdf_cosine_similarity(cos_sim, dim):
                        return scipy.stats.norm.cdf(cos_sim * np.sqrt(dim))

                    def _logit(p, eps=1e-8):
                        p = np.clip(p, eps, 1 - eps)
                        return np.log(p / (1 - p))

                    adam_update_norm = adam_pre_lr_update_norm * lr

                    _calc_feats = calculate_features(
                        m_hat=m_hat,
                        grad=grad,
                        sqrt_v_hat=sqrt_v_hat,
                        epsilon=epsilon,
                        adam_pre_lr_update=adam_pre_lr_update,
                        p=p.data,
                    )
                    m_hat_norm = _calc_feats["m_hat_norm"]
                    m_hat_sub_grad_norm = _calc_feats["m_hat_sub_grad_norm"]
                    sqrt_v_hat_gt_epsilon = _calc_feats["sqrt_v_hat_gt_epsilon"]
                    mean_log_abs_adam_pre_lr_update = _calc_feats[
                        "mean_log_abs_adam_pre_lr_update"
                    ]
                    grad_dot_p = _calc_feats["grad_dot_p"]
                    grad_dot_m_hat = _calc_feats["grad_dot_m_hat"]
                    grad_dot_adam_pre_lr_update = _calc_feats[
                        "grad_dot_adam_pre_lr_update"
                    ]

                    log_p_norm = np.log(p_norm + 1e-8)
                    log_grad_norm = np.log(grad_norm + 1e-8)
                    log_adam_pre_lr_update_norm = np.log(adam_pre_lr_update_norm + 1e-8)
                    log_m_hat_norm = np.log(m_hat_norm + 1e-8)
                    log_adam_update_norm = np.log(adam_update_norm + 1e-8)

                    self.inner_step_features.record_per_param(
                        "sqrt_v_hat_gt_epsilon", sqrt_v_hat_gt_epsilon
                    )

                    # TODO mean_abs_adam_pre_lr_update might make sense too
                    self.inner_step_features.record_per_param(
                        "mean_log_abs_adam_pre_lr_update",
                        mean_log_abs_adam_pre_lr_update,
                    )

                    self.inner_step_features.record_per_param(
                        "log_two_regimes_scale", log_m_hat_norm - log_grad_norm
                    )

                    self.inner_step_features.record_per_param(
                        "log_adam_update_norm_over_param_norm",
                        log_adam_update_norm - log_p_norm,
                    )

                    self.inner_step_features.record_per_param(
                        "log_noise_scale",
                        log_m_hat_norm - np.log(m_hat_sub_grad_norm + 1e-8),
                    )

                    cosine_similarity_grad_momentum = _cosine_similarity(
                        grad_dot_m_hat, grad_norm, m_hat_norm
                    )
                    self.inner_step_features.record_per_param(
                        "cosine_similarity_grad_momentum",
                        cosine_similarity_grad_momentum,
                    )

                    cosine_similarity_grad_param = _cosine_similarity(
                        grad_dot_p, grad_norm, p_norm
                    )
                    self.inner_step_features.record_per_param(
                        "cosine_similarity_grad_param", cosine_similarity_grad_param
                    )

                    self.inner_step_features.record_per_param(
                        "log_lamb", log_p_norm - log_adam_pre_lr_update_norm
                    )

                    cosine_similarity_grad_update = _cosine_similarity(
                        grad_dot_adam_pre_lr_update, grad_norm, adam_pre_lr_update_norm
                    )
                    self.inner_step_features.record_per_param(
                        "cosine_similarity_grad_update", cosine_similarity_grad_update
                    )

                    p_dim = np.prod(p.shape)
                    self.inner_step_features.record_per_param(
                        "logit_cdf_cosine_similarity_grad_momentum",
                        _logit(
                            _cdf_cosine_similarity(
                                cosine_similarity_grad_momentum, p_dim
                            )
                        ),
                    )

                    self.inner_step_features.record_per_param(
                        "logit_cdf_cosine_similarity_grad_update",
                        _logit(
                            _cdf_cosine_similarity(cosine_similarity_grad_update, p_dim)
                        ),
                    )

                    self.inner_step_features.record_per_param(
                        "cdf_cosine_similarity_grad_param",
                        _cdf_cosine_similarity(cosine_similarity_grad_param, p_dim),
                    )

                    self.inner_step_features.param_aggregate()

                self.inner_step_features.param_group_aggregate()

            self.inner_step_features.inner_step_aggregate()

        self.inner_step_count += 1
        return loss


def make_inner_optimizer(
    params, inner_hyperparameters, inner_step_features, param_names=None
):
    return CustomizableInnerAdaptiveOptimizer(
        params=params,
        param_names=param_names,
        inner_step_features=inner_step_features,
        **inner_hyperparameters
    )
