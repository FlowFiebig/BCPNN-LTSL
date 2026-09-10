"""
DeepVaR backtesting loss functions (Fatouros et al., Eqs. 11–14).
PnL is the realized log return; VaR is the left-tail quantile at confidence α.
"""
import numpy as np

DEEPVAR_LOSS_NAMES = (
    "quadratic_loss",
    "smooth_loss",
    "tick_loss",
    "firm_loss",
)

def _validate_pnl_var(pnl, var) :
    pnl = np.asarray(pnl, dtype=float).ravel()
    var = np.asarray(var, dtype=float).ravel()
    if len(pnl) != len(var) :
        raise ValueError(
            f"pnl and var must have the same length; got {len(pnl)} and {len(var)}.")
    mask = np.isfinite(pnl) & np.isfinite(var)
    if not mask.any() :
        raise ValueError("No finite (pnl, var) pairs to score.")
    return pnl[mask], var[mask]


def var_violation_indicator(pnl, var) :
    """
    VaR exceedance indicator I_t (DeepVaR Eq. 7): 1 if VaR_t > PnL_t, else 0.
    """
    pnl, var = _validate_pnl_var(pnl, var)
    return (var > pnl).astype(float)

def quadratic_loss(pnl, var) :
    """Quadratic loss l_QL (DeepVaR Eq. 11): sum of I_t * (1 + (PnL - VaR)^2).  Now changed to sample--wise loss"""
    pnl, var = _validate_pnl_var(pnl, var)
    i_t = var_violation_indicator(pnl, var)
    diff = pnl - var
    return float(np.mean(i_t * (1.0 + diff ** 2)))

def smooth_loss(pnl, var, alpha, d=25.0) :
    """Smooth Loss penalizes VaR-iolations more heavily (with weight alpha) Smaller loss indicates a better goodness of fit."""
    pnl, var = _validate_pnl_var(pnl, var)
    alpha = float(alpha)
    diff = pnl - var
    smooth_ind = 1.0 / (1.0 + np.exp(float(d) * diff))
    return float(np.mean((alpha - smooth_ind)*diff)) 

def tick_loss(pnl, var, alpha) :
    """Tick loss l_T (DeepVaR Eq. 13): sum of (α - I_t) * (PnL - VaR). BUT THIS IS NOT CORRECT, IT SHOULD BE (1-α) - I_t, so I changed it."""
    pnl, var = _validate_pnl_var(pnl, var)
    a = 1-float(alpha)
    i_t = var_violation_indicator(pnl, var)
    diff = pnl - var
    return float(np.mean((a - i_t) * diff))


def firm_loss(pnl, var, a=1.0) :
    """
    Firm loss l_F (DeepVaR Eq. 14): sum of (PnL-VaR)² on violations,
    else -a·VaR (paper uses a=1).

    Now changed to sample--wise loss
    """
    pnl, var = _validate_pnl_var(pnl, var)
    a = float(a)
    violation = pnl < var
    diff = pnl - var
    terms = np.where(violation, diff ** 2, -a * var)
    return float(np.mean(terms))


def aggregate_deepvar_losses(pnl, var, alpha, *, a=1.0, d=25.0) :
    """
    Compute all four DeepVaR losses plus violation counts on aligned arrays.
    """
    pnl, var = _validate_pnl_var(pnl, var)
    i_t = var_violation_indicator(pnl, var)
    n = len(pnl)
    n_violations = int(np.sum(i_t))
    return {
        "quadratic_loss"  : quadratic_loss(pnl, var),
        "smooth_loss"     : smooth_loss(pnl, var, alpha, d=d),
        "tick_loss"       : tick_loss(pnl, var, alpha),
        "firm_loss"       : firm_loss(pnl, var, a=a),
        "n_test_steps"    : n,
        "n_violations"    : n_violations,
        "violation_rate"  : float(n_violations / n) if n else float("nan"),
    }


# --------------------------------------------------------------------------
# Backtest coverage / independence tests (DeepVaR Tables 3, 5, 7, 9)
# --------------------------------------------------------------------------

def kupiec_uc(n_violations, n_steps, alpha) :
    """
    Kupiec (1995) unconditional-coverage LR test on the hit sequence.

    H0: P(violation) == 1 - alpha. The statistic is asymptotically chi2(1);
    large values mean the realized breach frequency is incompatible with the
    nominal level in *either* direction (too many *or* too few breaches).
    """
    from scipy import stats
    x, n, p = int(n_violations), int(n_steps), 1.0 - float(alpha)
    if n <= 0 :
        return float("nan"), float("nan")
    if x == 0 :
        lr = -2.0 * (n * np.log(1.0 - p))
    else :
        pi = x / n
        lr = -2.0 * ((n - x) * np.log(1.0 - p) + x * np.log(p)
                     - (n - x) * np.log(1.0 - pi) - x * np.log(pi))
    lr = float(max(lr, 0.0))
    return lr, float(1.0 - stats.chi2.cdf(lr, 1))


def christoffersen(indicator, alpha) :
    """
    Christoffersen (1998) independence and conditional-coverage LR tests.

    ``indicator`` is the 0/1 hit sequence in calendar order. LR_ind tests the
    first-order Markov transition probabilities against a common breach rate
    (i.e. it rejects *clustered* violations) and is chi2(1);
    LR_cc = LR_uc + LR_ind is the joint test and is chi2(2).
    """
    from scipy import stats
    I = np.asarray(indicator, dtype=int).ravel()
    n00 = int(np.sum((I[:-1] == 0) & (I[1:] == 0)))
    n01 = int(np.sum((I[:-1] == 0) & (I[1:] == 1)))
    n10 = int(np.sum((I[:-1] == 1) & (I[1:] == 0)))
    n11 = int(np.sum((I[:-1] == 1) & (I[1:] == 1)))
    p01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def _ll(k, q) :
        return k * np.log(q) if (k and q > 0) else 0.0

    ll_null = _ll(n00 + n10, 1.0 - p) + _ll(n01 + n11, p)
    ll_alt = (_ll(n00, 1.0 - p01) + _ll(n01, p01)
              + _ll(n10, 1.0 - p11) + _ll(n11, p11))
    lr_ind = float(max(-2.0 * (ll_null - ll_alt), 0.0))
    lr_uc, p_uc = kupiec_uc(int(I.sum()), int(I.size), alpha)
    lr_cc = lr_uc + lr_ind
    return {
        "LR_uc"  : lr_uc,
        "p_uc"   : p_uc,
        "LR_ind" : lr_ind,
        "p_ind"  : float(1.0 - stats.chi2.cdf(lr_ind, 1)),
        "LR_cc"  : float(lr_cc),
        "p_cc"   : float(1.0 - stats.chi2.cdf(lr_cc, 2)),
        "n00"    : n00, "n01" : n01, "n10" : n10, "n11" : n11,
    }


def coverage_tests(pnl, var, alpha) :
    """All four DeepVaR losses plus the Kupiec / Christoffersen statistics."""
    pnl, var = _validate_pnl_var(pnl, var)
    out = aggregate_deepvar_losses(pnl, var, alpha)
    out.update(christoffersen(var_violation_indicator(pnl, var), alpha))
    out["mean_var"] = float(np.mean(var))
    out["expected_violations"] = float((1.0 - float(alpha)) * len(pnl))
    return out
