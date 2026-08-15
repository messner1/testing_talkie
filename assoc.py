#!/usr/bin/env python3
"""Association statistics: the log odds ratio, tested by Dunning's G^2.

One statistic is reported throughout this project. Lambda -- the log odds ratio -- is the
estimate, and G^2 is the test. Nothing else: no relative risk, no ratio of ratios, no risk
difference, no Wald z, no Fisher exact, no exact binomial.

Both the two-way and the three-way association are reported:

    TWO-WAY    within one stratum, are scaffolded prompts likelier to show the outcome?
               This is lambda, the log odds ratio of a single 2x2.
    THREE-WAY  does that association DIFFER across the cutoff? lambda3 = lam_post - lam_in.
               Its null is homogeneous association -- one odds ratio in both strata --
               NOT an odds ratio of 1.

The three-way is the estimand. Scaffolded prompts are SELECTED for supplying morphological
material, so they raise form cohesion for any model whatever its cutoff; talkie-web, which
has seen the post-cutoff words, still shows a large two-way association. Only the second
difference asks whether the donor matters more where the target cannot have been memorised.

NOTHING HERE IS HAND-ROLLED. Every statistic is computed by scipy.stats or statsmodels and
this module only routes the cells and names the result. The mapping, so it can be checked:

    lambda, OR, interval    scipy.stats.contingency.odds_ratio(kind='sample')
    G^2 (2x2)               scipy.stats.chi2_contingency(lambda_='log-likelihood')
    G^2 (three-way)         statsmodels Poisson log-linear GLM: the deviance of the
                            homogeneous-association model, df=1
    lambda3 and its SE      statsmodels Table2x2.log_oddsratio / .log_oddsratio_se
    G^2 (paired arms)       scipy.stats.power_divergence(lambda_='log-likelihood')
    paired interval         scipy.stats.binomtest(...).proportion_ci()

scipy covers everything except the three-way test, and it cannot cover that: scipy has no
log-linear model, and `chi2_contingency` on a 2x2x2 tests MUTUAL INDEPENDENCE on 3 df --
a different and uninteresting null, which on this project's cells returns G^2 = 2635 where
the interaction is 18.7 on 1 df. Using it would be fitting the analysis to the library.
statsmodels is used for that one quantity, where a Poisson GLM's deviance is exactly the
likelihood-ratio statistic for the three-way term.

G^2 rather than chi-squared follows Dunning (1993): chi-squared's normal approximation
fails when expected counts are small. Here that bites in three of the twenty-four tables --
the form>=5 post-cutoff tables, minimum expected count 1.8 to 2.6 -- which are also among
the strongest reported results. `min_expected` reports the diagnostic so a reader can see
where it matters rather than take the choice on trust.

Lambda is exactly the interaction coefficient of the saturated log-linear model fitted to
the table -- the `lambda` of Blaheta & Johnson (2001), as implemented in quanteda's
`textstat_collocations`, where lambda = sum (-1)^(K-b_i) log n_i collapses at K=2 to
log(n11 n00 / n10 n01). The contingency-table statistic and the logistic-regression
interaction coefficient are one quantity; either derivation may be quoted.

References
    Blaheta, D. & Johnson, M. (2001). Unsupervised learning of multi-word verbs. ACL
        Workshop on Collocations.
    Dunning, T. (1993). Accurate methods for the statistics of surprise and coincidence.
        Computational Linguistics 19(1):61-74.
    Gries, S. Th. & Durrant, P. (2020). Analyzing co-occurrence data. In Paquot & Gries
        (eds.), A Practical Handbook of Corpus Linguistics.
"""

import math
import warnings

import numpy as np
import statsmodels.api as sm
from scipy.stats import binomtest, chi2, chi2_contingency, power_divergence
from scipy.stats.contingency import odds_ratio
from statsmodels.stats.contingency_tables import Table2x2

CHI2_95_DF1 = float(chi2.ppf(0.95, 1))
Z95 = 1.959963984540054


# --------------------------------------------------------------------------- #
# 2x2 tables -- the two-way association
#
# Cell convention throughout: (a, b, c, d) =
#     a  cue present, outcome present        b  cue absent, outcome present
#     c  cue present, outcome absent         d  cue absent, outcome absent
# so the odds ratio is ad/bc. "Cue" is the scaffolding verdict in this project's tables;
# "outcome" is the paradigm group, recall, or whatever is being predicted.
# --------------------------------------------------------------------------- #
def haldane(a, b, c, d):
    """Add 0.5 to every cell if any is zero, else leave the counts alone.

    Without this the sample odds ratio is infinite for a zero cell. The correction goes to
    all four cells rather than only the empty one, which keeps the estimator's bias
    symmetric; correcting one cell shifts lambda in a direction that depends on which cell
    happened to be empty. `zero_cell` reports whether it fired, so a table that needed it
    can be flagged rather than silently smoothed.
    """
    if 0 in (a, b, c, d):
        return a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return float(a), float(b), float(c), float(d)


def zero_cell(a, b, c, d):
    """Whether the Haldane correction fires for this table."""
    return 0 in (a, b, c, d)


def _table(a, b, c, d, dtype=float):
    """The 2x2 in the orientation scipy and statsmodels expect: rows cue, cols outcome."""
    return np.array([[a, c], [b, d]], dtype=dtype)


def _or_ci(a, b, c, d):
    """(odds ratio, lo, hi) for one table, from scipy where it can take the counts.

    `scipy.stats.contingency.odds_ratio` requires integer counts, so a table needing the
    Haldane correction -- whose cells are half-integers -- falls back to statsmodels
    Table2x2. That is a dtype accommodation, not a change of method: both compute the
    sample odds ratio with the same Woolf interval, and the tests assert they agree to
    1e-6 on the tables this project actually reports.
    """
    if not zero_cell(a, b, c, d):
        r = odds_ratio(_table(int(a), int(b), int(c), int(d), dtype=int), kind="sample")
        lo, hi = r.confidence_interval(0.95)
        return float(r.statistic), float(lo), float(hi)
    t = Table2x2(_table(*haldane(a, b, c, d)))
    lo, hi = t.oddsratio_confint()
    return float(t.oddsratio), float(lo), float(hi)


def lam(a, b, c, d):
    """Log odds ratio, via scipy.stats.contingency.odds_ratio."""
    return math.log(_or_ci(a, b, c, d)[0])


def g2_2x2(a, b, c, d):
    """Dunning's G^2 for the 2x2, df=1, via scipy.stats.chi2_contingency.

    `lambda_='log-likelihood'` selects the log-likelihood-ratio member of the Cressie-Read
    power divergence family, which is exactly Dunning's statistic. Observed zeros need no
    correction here -- x log x tends to 0 -- so the counts go in uncorrected even where
    `lam` needs Haldane.
    """
    t = _table(a, b, c, d)
    if t.sum() == 0 or (t.sum(axis=0) == 0).any() or (t.sum(axis=1) == 0).any():
        return 0.0
    return float(chi2_contingency(t, lambda_="log-likelihood", correction=False)[0])


def min_expected(a, b, c, d):
    """Smallest expected cell count -- the diagnostic behind the choice of G^2."""
    t = _table(a, b, c, d)
    if t.sum() == 0 or (t.sum(axis=0) == 0).any() or (t.sum(axis=1) == 0).any():
        return 0.0
    return float(chi2_contingency(t, lambda_="log-likelihood",
                                  correction=False).expected_freq.min())


def association(a, b, c, d):
    """(lambda, odds ratio, (lo, hi), G^2) for one 2x2 -- the two-way association."""
    orr, lo, hi = _or_ci(a, b, c, d)
    return math.log(orr), orr, (lo, hi), g2_2x2(a, b, c, d)


# --------------------------------------------------------------------------- #
# The three-way interaction
# --------------------------------------------------------------------------- #
def _log_or_se(a, b, c, d):
    """(log OR, its SE) from statsmodels Table2x2."""
    a, b, c, d = haldane(a, b, c, d)
    t = Table2x2(_table(a, b, c, d))
    return float(t.log_oddsratio), float(t.log_oddsratio_se)


def lam3(post, incut):
    """The three-way lambda: how much stronger the association is in the post arm."""
    return _log_or_se(*post)[0] - _log_or_se(*incut)[0]


def g2_lrt_2x2x2(post, incut):
    """LRT for the three-way term, df=1 -- statsmodels Poisson log-linear GLM.

    Fits the homogeneous-association model: every two-way term present, the three-way term
    absent. Its deviance IS the likelihood-ratio statistic against the saturated table, on
    1 df, because deviance is 2*(loglik_saturated - loglik_model) by definition.

    The null is that the odds ratio is the SAME in both strata, not that it is 1. A
    per-table G^2 cannot serve this purpose: those test each table's own association
    against zero, and in this project they run 31 to 311 and are significant everywhere.
    """
    counts, X = [], []
    for si, t in ((1, post), (0, incut)):
        a, b, c, d = haldane(*t)
        for cue, out, k in ((1, 1, a), (0, 1, b), (1, 0, c), (0, 0, d)):
            counts.append(float(k))
            X.append([1, cue, out, si, cue * out, cue * si, out * si])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(np.array(counts), np.array(X, dtype=float),
                     family=sm.families.Poisson()).fit()
    return float(fit.deviance)


def interaction(post, incut):
    """(lambda3, odds ratio, (lo, hi), G^2) for cue x stratum x outcome.

    The interval combines the two strata's Woolf SEs, which is the standard form for a
    difference of independent log odds ratios and reproduces the saturated log-linear
    model's own confidence interval exactly -- checked in the tests. It is computed this
    way rather than by fitting the saturated model because that model has zero residual
    degrees of freedom, which makes statsmodels emit spurious separation warnings.
    """
    lp, sp = _log_or_se(*post)
    li, si = _log_or_se(*incut)
    l3 = lp - li
    se = math.sqrt(sp * sp + si * si)
    return l3, math.exp(l3), (math.exp(l3 - Z95 * se), math.exp(l3 + Z95 * se)), \
        g2_lrt_2x2x2(post, incut)


# --------------------------------------------------------------------------- #
# Paired arms (the perturbation): the conditional odds ratio
# --------------------------------------------------------------------------- #
def lam_paired(b, c):
    """Conditional log odds ratio for a within-item design: log(b/c).

    When the same item is measured under two conditions, an unpaired odds ratio ignores the
    pairing. The conditional (McNemar) odds ratio conditions on each item's own baseline,
    so the item drops out and only the DISCORDANT pairs carry information: `b` items that
    changed one way, `c` the other. Concordant pairs are uninformative by construction, not
    by being discarded.

    This is still a log odds ratio, so the perturbation is reported on the same scale as
    every other result in the project.
    """
    if b == 0 or c == 0:
        b, c = b + 0.5, c + 0.5
    return math.log(b / c)


def g2_paired(b, c):
    """G^2 for marginal homogeneity, df=1 -- scipy.stats.power_divergence.

    Tests b = c against the binomial: the likelihood-ratio form of McNemar, and the paired
    counterpart of `g2_2x2`, so the perturbation is tested the same way as everything else.

    Its chi-squared reference is asymptotic and thins when b + c is small; where that
    happens the exact binomial is recorded in the internal notes as a cross-check rather
    than reported as a second statistic.
    """
    m = b + c
    if m == 0:
        return 0.0
    return float(power_divergence([b, c], [m / 2, m / 2],
                                  lambda_="log-likelihood")[0])


def paired(b, c):
    """(lambda, odds ratio, (lo, hi), G^2) for a pair of within-item arms.

    The interval comes from a Clopper-Pearson interval on b out of b + c discordant pairs,
    mapped through pi -> pi/(1-pi), which is the conditional odds ratio's own scale.
    """
    m = b + c
    l = lam_paired(b, c)
    if m == 0:
        return l, math.exp(l), (0.0, math.inf), 0.0
    ci = binomtest(int(b), int(m), 0.5).proportion_ci(0.95)
    lo = ci.low / (1 - ci.low) if ci.low < 1 else math.inf
    hi = ci.high / (1 - ci.high) if ci.high < 1 else math.inf
    return l, math.exp(l), (float(lo), float(hi)), g2_paired(b, c)


# --------------------------------------------------------------------------- #
def stars(g2):
    """Conventional significance marks from G^2 on 1 df."""
    return ("***" if g2 > chi2.ppf(0.999, 1) else
            "**" if g2 > chi2.ppf(0.99, 1) else
            "*" if g2 > CHI2_95_DF1 else "")
