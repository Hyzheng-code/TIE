import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union, Dict
import pandas as pd
import math
from scipy import stats, integrate


class Calu_logt_score:
    """Score calculator for the log-t distribution (fixed ν=3.5).

    Censors the distribution at MAX_GENERATED_TOKENS to account for the
    max_tokens limit imposed by vLLM at inference time.
    """

    MAX_GENERATED_TOKENS = 2028.0
    CVAR_MAX_GENERATED_TOKENS = 2048.0
    NU = 3.5

    @staticmethod
    def quantile(mu: float, sigma: float, q: float) -> float:
        if q > 1:
            q = q / 100
        if not 0 < q < 1:
            raise ValueError(f"q must be in (0,1) or (0,100), got {q}")
        t_q = stats.t.ppf(q, df=Calu_logt_score.NU)
        return np.exp(mu + sigma * t_q)

    @staticmethod
    def mean(mu: float, sigma: float, n_samples: int = 10000) -> float:
        """Censored expectation E[min(X, x_max)] via Monte Carlo sampling."""
        t_samples = stats.t.rvs(df=Calu_logt_score.NU, size=n_samples)
        x_samples = np.exp(mu + sigma * t_samples)
        x_truncated = np.minimum(x_samples, Calu_logt_score.MAX_GENERATED_TOKENS)
        return np.mean(x_truncated)

    @staticmethod
    def var(mu: float, sigma: float, alpha: float) -> float:
        return Calu_logt_score.quantile(mu, sigma, alpha)

    @staticmethod
    def cvar(mu: float, sigma: float, alpha: float, n_samples: int = 10000) -> float:
        """Censored CVaR_α[X] via Monte Carlo sampling (paper Appendix B)."""
        if alpha > 1:
            alpha = alpha / 100
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1) or (0,100), got {alpha}")
        t_samples = stats.t.rvs(df=Calu_logt_score.NU, size=n_samples)
        x_samples = np.exp(mu + sigma * t_samples)
        x_truncated = np.minimum(x_samples, Calu_logt_score.CVAR_MAX_GENERATED_TOKENS)
        var = np.percentile(x_truncated, alpha * 100)
        exceedances = x_truncated[x_truncated > var]
        if len(exceedances) == 0:
            return var
        else:
            return np.mean(exceedances)

    def mean_add_n_pa(mu: float, sigma: float, a: float, n: float) -> float:
        mean = Calu_logt_score.mean(mu, sigma)
        pa = Calu_logt_score.quantile(mu, sigma, q=a)
        return mean + n * pa

    def mean_add_n_cvar(mu: float, sigma: float, a: float, n: float) -> float:
        """TIE score: E[X] + β·CVaR_α[X] (paper Eq. 11)."""
        mean = Calu_logt_score.mean(mu, sigma)
        cvar = Calu_logt_score.cvar(mu, sigma, alpha=a)
        return mean + n * cvar

    def pa_add_pb(mu: float, sigma: float, a: float, b: float) -> float:
        pa = Calu_logt_score.quantile(mu, sigma, q=a)
        pb = Calu_logt_score.quantile(mu, sigma, q=b)
        return pa + pb


class Calu_lognorm_score:
    """Score calculator for the log-normal distribution (baseline variant)."""

    @staticmethod
    def mean(mu: float, sigma: float) -> float:
        return math.exp(mu + (sigma ** 2) / 2)

    @staticmethod
    def quantile(mu: float, sigma: float, q: float) -> float:
        if q > 1:
            q = q / 100
        if not 0 < q < 1:
            raise ValueError(f"q must be in (0,1) or (0,100), got {q}")
        z_q = stats.norm.ppf(q)
        return math.exp(mu + sigma * z_q)

    @staticmethod
    def std(mu: float, sigma: float) -> float:
        variance = (math.exp(sigma ** 2) - 1) * math.exp(2 * mu + sigma ** 2)
        return math.sqrt(variance)

    @staticmethod
    def cv(mu: float, sigma: float) -> float:
        return math.sqrt(math.exp(sigma ** 2) - 1)

    @staticmethod
    def var(mu: float, sigma: float, alpha: float) -> float:
        return Calu_lognorm_score.quantile(mu, sigma, alpha)

    @staticmethod
    def cvar(mu: float, sigma: float, alpha: float) -> float:
        """Closed-form CVaR for the log-normal distribution."""
        if alpha > 1:
            alpha = alpha / 100
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1) or (0,100), got {alpha}")
        var_alpha = Calu_lognorm_score.quantile(mu, sigma, alpha)
        expectation = Calu_lognorm_score.mean(mu, sigma)
        # CVaR = E[X] * Phi((mu + sigma^2 - ln(VaR)) / sigma) / (1 - alpha)
        z = (mu + sigma ** 2 - math.log(var_alpha)) / sigma
        return expectation * stats.norm.cdf(z) / (1 - alpha)

    @staticmethod
    def mean_add_n_cvar(mu: float, sigma: float, a: float, n: float) -> float:
        mean = Calu_lognorm_score.mean(mu, sigma)
        cvar = Calu_lognorm_score.cvar(mu, sigma, alpha=a)
        return mean + n * cvar
