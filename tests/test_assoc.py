#!/usr/bin/env python3
"""Check that `assoc.py` routes cells to the right library call and reads the right result.

Nothing in `assoc.py` implements a statistic; scipy and statsmodels do. So these tests are
not checking arithmetic, they are checking plumbing -- that the table is built in the
orientation each library expects, that the coefficient read off is the one intended, and
that independent derivations of the same quantity agree.

Three derivations of lambda are checked to coincide:
  * scipy.stats.contingency.odds_ratio           the contingency-table form
  * statsmodels Table2x2.log_oddsratio           the epidemiological form
  * a logistic regression's interaction term     the regression form

and two derivations of the three-way G^2:
  * statsmodels Poisson log-linear deviance      what `assoc` uses
  * a logistic nested-model likelihood ratio     an independent route
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import assoc  # noqa: E402

sm = pytest.importorskip("statsmodels.api")
from statsmodels.stats.contingency_tables import Table2x2  # noqa: E402
from scipy.stats.contingency import odds_ratio  # noqa: E402

# The real cells this project reports, so the tests exercise the actual regime rather than
# toy numbers. (a, b, c, d) = cue&outcome, nocue&outcome, cue&no-outcome, nocue&no-outcome.
TB_CERT_POST = (27, 96, 94, 2638)      # Talkie-Base, certified, post-cutoff
TB_CERT_IN = (591, 306, 1102, 1388)    # Talkie-Base, certified, in-cutoff
TB_F5_POST = (16, 27, 105, 2707)       # thinnest table in the analysis
TB_F5_IN = (355, 95, 1338, 1599)
ALL = (TB_CERT_POST, TB_CERT_IN, TB_F5_POST, TB_F5_IN)


def _expand(cells, stratum=None):
    """One row per item: (outcome, cue[, stratum]). For the regression cross-checks."""
    a, b, c, d = cells
    rows = [(1, 1)] * a + [(1, 0)] * b + [(0, 1)] * c + [(0, 0)] * d
    y = [r[0] for r in rows]
    if stratum is None:
        X = [[1.0, r[1]] for r in rows]
    else:
        X = [[1.0, r[1], stratum, r[1] * stratum] for r in rows]
    return y, X


# --------------------------------------------------------------------------- #
# the table is built in the orientation the libraries expect
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cells", ALL)
def test_orientation_gives_ad_over_bc(cells):
    """A transposed table would silently invert every odds ratio in the project."""
    a, b, c, d = cells
    assert assoc.lam(a, b, c, d) == pytest.approx(math.log(a * d / (b * c)), abs=1e-9)


@pytest.mark.parametrize("cells", ALL)
def test_scipy_and_statsmodels_agree_on_estimate_and_interval(cells):
    """Justifies the zero-cell fallback: scipy cannot take half-integer counts, so a
    Haldane-corrected table is routed to Table2x2. That is only legitimate if the two
    agree on tables where both apply -- same estimator, same Woolf interval."""
    a, b, c, d = cells
    Ti = np.array([[a, c], [b, d]], dtype=int)
    Tf = np.array([[a, c], [b, d]], dtype=float)
    s = odds_ratio(Ti, kind="sample")
    slo, shi = s.confidence_interval(0.95)
    t = Table2x2(Tf)
    tlo, thi = t.oddsratio_confint()
    assert math.log(s.statistic) == pytest.approx(t.log_oddsratio, abs=1e-9)
    assert (slo, shi) == pytest.approx((tlo, thi), abs=1e-6)
    assert assoc.lam(a, b, c, d) == pytest.approx(t.log_oddsratio, abs=1e-9)


def test_zero_cell_falls_back_without_changing_the_estimator():
    """A table with an empty cell still yields a finite lambda and a usable interval."""
    l, orr, (lo, hi), g2 = assoc.association(0, 12, 40, 900)
    assert math.isfinite(l) and lo < orr < hi
    a, b, c, d = assoc.haldane(0, 12, 40, 900)
    assert orr == pytest.approx(a * d / (b * c), abs=1e-9)


@pytest.mark.parametrize("cells", ALL)
def test_lambda_equals_logistic_slope(cells):
    """The contingency-table lambda IS the logistic coefficient."""
    y, X = _expand(cells)
    fit = sm.Logit(np.array(y, float), np.array(X, float)).fit(disp=0)
    assert assoc.lam(*cells) == pytest.approx(fit.params[1], abs=1e-6)


def test_lambda_is_sign_symmetric_under_outcome_polarity():
    """Relabelling "group present" as "group absent" must only flip lambda's sign.

    This is the property that motivated the odds ratio: a relative risk gives materially
    different answers to the two codings, and the outcome's polarity here is a labelling
    convention, so a measure that depends on it cannot carry the claim.
    """
    a, b, c, d = TB_CERT_POST
    assert assoc.lam(c, d, a, b) == pytest.approx(-assoc.lam(a, b, c, d), abs=1e-9)


# --------------------------------------------------------------------------- #
# G^2 is the log-likelihood-ratio member, not Pearson
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cells", ALL)
def test_g2_is_log_likelihood_not_pearson(cells):
    """`lambda_='log-likelihood'` must be selected; Pearson is a different statistic."""
    from scipy.stats import chi2_contingency
    T = np.array([[cells[0], cells[2]], [cells[1], cells[3]]], float)
    pearson = chi2_contingency(T, correction=False)[0]
    assert assoc.g2_2x2(*cells) != pytest.approx(pearson, rel=1e-6)
    exp = chi2_contingency(T, correction=False).expected_freq
    manual = 2 * sum(o * math.log(o / e)
                     for o, e in zip(T.ravel(), exp.ravel()) if o > 0)
    assert assoc.g2_2x2(*cells) == pytest.approx(manual, abs=1e-9)


def test_min_expected_flags_the_thin_tables():
    """form>=5 post-cutoff is the regime Dunning's argument is actually needed for."""
    assert assoc.min_expected(*TB_F5_POST) < 5
    assert assoc.min_expected(*TB_CERT_IN) > 5


# --------------------------------------------------------------------------- #
# the three-way interaction
# --------------------------------------------------------------------------- #
def test_lam3_equals_difference_of_table_lambdas():
    assert assoc.lam3(TB_CERT_POST, TB_CERT_IN) == pytest.approx(
        assoc.lam(*TB_CERT_POST) - assoc.lam(*TB_CERT_IN), abs=1e-9)


def test_lam3_equals_logistic_interaction_coefficient():
    """The three-way lambda IS the interaction term of `outcome ~ cue * stratum`."""
    ys, Xs = [], []
    for stratum, cells in ((1, TB_CERT_POST), (0, TB_CERT_IN)):
        y, X = _expand(cells, stratum)
        ys += y
        Xs += X
    fit = sm.Logit(np.array(ys, float), np.array(Xs, float)).fit(disp=0)
    assert assoc.lam3(TB_CERT_POST, TB_CERT_IN) == pytest.approx(fit.params[3], abs=1e-6)


def test_three_way_g2_matches_logistic_nested_likelihood_ratio():
    """Poisson log-linear deviance and a logistic LRT are the same model, two routes."""
    ys, Xs = [], []
    for stratum, cells in ((1, TB_CERT_POST), (0, TB_CERT_IN)):
        y, X = _expand(cells, stratum)
        ys += y
        Xs += X
    y, X = np.array(ys, float), np.array(Xs, float)
    full = sm.Logit(y, X).fit(disp=0)
    nested = sm.Logit(y, X[:, :3]).fit(disp=0)
    assert assoc.g2_lrt_2x2x2(TB_CERT_POST, TB_CERT_IN) == pytest.approx(
        2 * (full.llf - nested.llf), abs=1e-4)


def test_three_way_g2_is_one_degree_of_freedom_not_mutual_independence():
    """Guards against reaching for `chi2_contingency` on a 2x2x2, which tests a
    different null on 3 df and returns a wildly larger statistic."""
    from scipy.stats import chi2_contingency
    T3 = np.array([[[TB_CERT_POST[0], TB_CERT_POST[2]],
                    [TB_CERT_POST[1], TB_CERT_POST[3]]],
                   [[TB_CERT_IN[0], TB_CERT_IN[2]],
                    [TB_CERT_IN[1], TB_CERT_IN[3]]]], float)
    mutual = chi2_contingency(T3, lambda_="log-likelihood")
    assert mutual.dof > 1
    assert mutual.statistic > 100 * assoc.g2_lrt_2x2x2(TB_CERT_POST, TB_CERT_IN)


def test_interaction_interval_matches_saturated_loglinear_model():
    """The combined-Woolf interval must equal the saturated model's own conf_int.

    `interaction` avoids fitting the saturated model because it has zero residual degrees
    of freedom and statsmodels emits spurious separation warnings; this asserts the
    shortcut is exact rather than merely close.
    """
    import warnings
    counts, X = [], []
    for si, t in ((1, TB_CERT_POST), (0, TB_CERT_IN)):
        a, b, c, d = t
        for cue, out, k in ((1, 1, a), (0, 1, b), (1, 0, c), (0, 0, d)):
            counts.append(float(k))
            X.append([1, cue, out, si, cue * out, cue * si, out * si,
                      cue * out * si])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sat = sm.GLM(np.array(counts), np.array(X, float),
                     family=sm.families.Poisson()).fit()
    _, _, (lo, hi), _ = assoc.interaction(TB_CERT_POST, TB_CERT_IN)
    want_lo, want_hi = sat.conf_int()[-1]
    assert math.log(lo) == pytest.approx(want_lo, abs=1e-6)
    assert math.log(hi) == pytest.approx(want_hi, abs=1e-6)
    assert sat.params[-1] == pytest.approx(assoc.lam3(TB_CERT_POST, TB_CERT_IN), abs=1e-6)


def test_reported_interaction_values_are_stable():
    """Guards the figures quoted in results.md against silent drift."""
    l3, orr, (lo, hi), g2 = assoc.interaction(TB_CERT_POST, TB_CERT_IN)
    assert l3 == pytest.approx(1.177, abs=5e-3)
    assert orr == pytest.approx(3.24, abs=5e-3)
    assert g2 == pytest.approx(18.70, abs=0.05)
    assert lo > 1.0

    l3, orr, _, g2 = assoc.interaction(TB_F5_POST, TB_F5_IN)
    assert orr == pytest.approx(3.42, abs=5e-3)
    assert g2 == pytest.approx(11.06, abs=0.05)


def test_interval_and_test_agree_on_significance():
    for cells in ALL:
        _, _, (lo, hi), g2 = assoc.association(*cells)
        assert (lo > 1.0 or hi < 1.0) == (g2 > assoc.CHI2_95_DF1)


# --------------------------------------------------------------------------- #
# paired arms
# --------------------------------------------------------------------------- #
def test_paired_lambda_is_log_ratio_of_discordant_pairs():
    assert assoc.lam_paired(31, 7) == pytest.approx(math.log(31 / 7), abs=1e-12)


def test_g2_paired_is_the_binomial_likelihood_ratio():
    b, c = 31, 7
    n = b + c
    manual = 2 * ((b * math.log(b / n) + c * math.log(c / n)) - n * math.log(0.5))
    assert assoc.g2_paired(b, c) == pytest.approx(manual, abs=1e-9)


def test_g2_paired_tracks_mcnemar_where_counts_are_ample():
    from statsmodels.stats.contingency_tables import mcnemar
    b, c = 40, 18
    chi2_stat = mcnemar([[100, b], [c, 100]], exact=False, correction=False).statistic
    assert assoc.g2_paired(b, c) == pytest.approx(chi2_stat, rel=0.05)


def test_paired_interval_brackets_estimate_and_matches_its_test():
    _, orr, (lo, hi), g2 = assoc.paired(31, 7)
    assert lo < orr < hi
    assert (lo > 1.0 or hi < 1.0) == (g2 > assoc.CHI2_95_DF1)


# --------------------------------------------------------------------------- #
# zero cells
# --------------------------------------------------------------------------- #
def test_haldane_applies_to_all_cells_or_none():
    assert assoc.haldane(5, 0, 3, 9) == (5.5, 0.5, 3.5, 9.5)
    assert assoc.haldane(5, 2, 3, 9) == (5.0, 2.0, 3.0, 9.0)
    assert assoc.zero_cell(5, 0, 3, 9) and not assoc.zero_cell(5, 2, 3, 9)


def test_lambda_is_finite_with_a_zero_cell():
    assert math.isfinite(assoc.lam(0, 12, 40, 900))


def test_g2_ignores_empty_cells_without_correction():
    """x log x -> 0, so an observed zero contributes nothing and needs no Haldane."""
    assert math.isfinite(assoc.g2_2x2(0, 12, 40, 900))


def test_stars_track_g2_thresholds():
    assert assoc.stars(0.4) == ""
    assert assoc.stars(4.5) == "*"
    assert assoc.stars(7.0) == "**"
    assert assoc.stars(18.7) == "***"
