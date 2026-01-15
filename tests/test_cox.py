import numpy as np
import pytest
import scipy.linalg

import firthmodels.cox
from firthmodels.cox import (
    FirthCoxPH,
    _concordance_index,
    _CoxPrecomputed,
    _validate_survival_y,
    _Workspace,
    compute_cox_quantities,
)


def _structured_y(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    y = np.empty(len(time), dtype=[("event", bool), ("time", np.float64)])
    y["event"] = event
    y["time"] = time
    return y


class TestValidateSurvivalY:
    def test_accepts_structured_array(self):
        event = np.array([True, False, True])
        time = np.array([1.0, 2.0, 3.0])
        y = _structured_y(event, time)

        event_out, time_out = _validate_survival_y(y, n_samples=3)

        assert event_out.dtype == bool
        assert time_out.dtype == np.float64
        np.testing.assert_array_equal(event_out, event)
        np.testing.assert_allclose(time_out, time)

    def test_accepts_tuple_event_time(self):
        event = np.array([0, 1, 0], dtype=np.int64)
        time = np.array([5.0, 6.0, 7.0], dtype=np.float64)

        event_out, time_out = _validate_survival_y((event, time), n_samples=3)

        np.testing.assert_array_equal(event_out, np.array([False, True, False]))
        np.testing.assert_allclose(time_out, time)

    def test_rejects_event_values_not_binary(self):
        event = np.array([0, 2])
        time = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match=r"event must contain only 0/1"):
            _validate_survival_y((event, time), n_samples=2)


class TestCoxPrecomputed:
    def test_blocks_event_counts_and_sums(self):
        X = np.array(
            [
                [10.0, 1.0],  # time 2, event 1
                [20.0, 2.0],  # time 5, event 1
                [30.0, 3.0],  # time 2, event 0
                [40.0, 4.0],  # time 5, event 1
                [50.0, 5.0],  # time 3, event 1
                [60.0, 6.0],  # time 4, event 0
            ]
        )
        time = np.array([2.0, 5.0, 2.0, 5.0, 3.0, 4.0])
        event = np.array([1, 1, 0, 1, 1, 0], dtype=bool)

        pre = _CoxPrecomputed.from_data(X, time, event, backend="numpy")

        # Sorted times: 5,5,4,3,2,2 -> block ends at [2,3,4,6]
        np.testing.assert_array_equal(pre.block_ends, np.array([2, 3, 4, 6]))
        np.testing.assert_array_equal(pre.block_d, np.array([2, 0, 1, 1]))

        expected_block_s = np.array(
            [
                [60.0, 6.0],  # time 5: rows [20,2] and [40,4] had events
                [0.0, 0.0],  # time 4: no events
                [50.0, 5.0],  # time 3: row [50,5]
                [10.0, 1.0],  # time 2: row [10,1]
            ]
        )
        np.testing.assert_allclose(pre.block_s, expected_block_s)


class TestFirthCoxPH:
    def test_two_individual_example_matches_log3(self):
        # (Heinze and Schemper, 2001), Section 2: two individuals, one covariate.
        # The modified score has root exp(beta_hat) = 3.
        X = np.array([[1.0], [0.0]])
        time = np.array([1.0, 2.0])
        event = np.array([True, False])
        y = _structured_y(event, time)

        model = FirthCoxPH(backend="numpy")
        model.fit(X, y)

        assert model.converged_
        np.testing.assert_allclose(model.coef_[0], np.log(3.0), rtol=1e-6, atol=1e-6)

    def test_breslow_baseline_hazard_two_individual(self):
        # Test Breslow estimator for two-individual example.
        # With beta = log(3):
        #   At time=1: risk set = {1,2}, S0 = exp(log(3)) + exp(0) = 4, H = 1/4
        X = np.array([[1.0], [0.0]])
        time = np.array([1.0, 2.0])
        event = np.array([True, False])
        y = _structured_y(event, time)

        model = FirthCoxPH(backend="numpy").fit(X, y)

        np.testing.assert_array_equal(model.unique_times_, [1.0])
        np.testing.assert_allclose(model.cum_baseline_hazard_, [0.25], rtol=1e-6)
        np.testing.assert_allclose(
            model.baseline_survival_, np.exp(-model.cum_baseline_hazard_)
        )

    def test_matches_coxphf_with_monotone_likelihood(self, cox_separation_data):
        """Matches coxphf on data with monotone likelihood."""
        X, time, event = cox_separation_data
        y = _structured_y(event, time)

        model = FirthCoxPH(backend="numpy")
        model.fit(X, y)

        expected_coef = np.array(
            [3.8815800583, 0.6042066427, -0.4202139201, 1.0415063080]
        )
        expected_bse = np.array(
            [1.4430190269, 0.1599607483, 0.2541800107, 0.3128784410]
        )
        expected_lr = 57.6071

        assert model.converged_
        np.testing.assert_allclose(model.coef_, expected_coef, rtol=1e-6)
        np.testing.assert_allclose(model.bse_, expected_bse, rtol=1e-6)

        # Absolute penalized log-likelihoods differ with coxphf by an
        # additive constant. Compare likelihood ratios instead.
        pre = _CoxPrecomputed.from_data(X, time, event, backend="numpy")
        ws = _Workspace(pre.n_samples, pre.n_features)
        null_loglik = compute_cox_quantities(np.zeros(X.shape[1]), pre, ws).loglik
        lr_stat = 2.0 * (model.loglik_ - null_loglik)
        np.testing.assert_allclose(lr_stat, expected_lr, rtol=1e-6)

        # LRT p-values
        model.lrt()
        expected_lrt_pvalues = np.array(
            [
                7.318984974e-09,  # separator
                1.686334002e-04,  # x1
                9.295083722e-02,  # x2
                8.463066114e-04,  # x3
            ]
        )
        np.testing.assert_allclose(model.lrt_pvalues_, expected_lrt_pvalues, rtol=1e-6)

        ci = model.conf_int(method="pl")
        # Profile likelihood CIs from coxphf (in log scale)
        expected_ci = np.array(
            [
                [1.9343469272, 8.7215676015],  # separator
                [0.2921186188, 0.9164270978],  # x1
                [-0.9217200647, 0.0699415336],  # x2
                [0.4348431279, 1.6550461482],  # x3
            ]
        )

        np.testing.assert_allclose(ci, expected_ci, rtol=1e-6)

    def test_lrt_warm_start_matches(self, cox_separation_data):
        X, time, event = cox_separation_data
        y = _structured_y(event, time)

        model_warm = FirthCoxPH(backend="numpy").fit(X, y)
        model_warm.lrt(warm_start=True)

        model_cold = FirthCoxPH(backend="numpy").fit(X, y)
        model_cold.lrt(warm_start=False)

        np.testing.assert_allclose(
            model_cold.lrt_pvalues_, model_warm.lrt_pvalues_, rtol=1e-6
        )
        np.testing.assert_allclose(model_cold.lrt_bse_, model_warm.lrt_bse_, rtol=1e-6)

    def test_lrt_warm_start_false_uses_zero_init(
        self, cox_separation_data, monkeypatch
    ):
        X, time, event = cox_separation_data
        y = _structured_y(event, time)
        model = FirthCoxPH(backend="numpy").fit(X, y)
        captured = {}

        def fake_constrained_lrt_1df(*, beta_init_free, **kwargs):
            captured["beta_init_free"] = beta_init_free
            return firthmodels.cox.LRTResult(
                chi2=0.0, pvalue=1.0, bse_backcorrected=1.0
            )

        monkeypatch.setattr(
            "firthmodels.cox.constrained_lrt_1df", fake_constrained_lrt_1df
        )

        model.lrt(0, warm_start=False)
        assert captured["beta_init_free"] is None

    def test_dpstrf_fallback_in_compute_cox_quantities(
        self, cox_separation_data, monkeypatch
    ):
        X, time, event = cox_separation_data
        y = _structured_y(event, time)

        model_normal = FirthCoxPH(backend="numpy").fit(X, y)

        called = {"dpstrf": 0}
        orig_dpstrf = firthmodels.cox.dpstrf

        def wrapped_dpstrf(*args, **kwargs):
            called["dpstrf"] += 1
            return orig_dpstrf(*args, **kwargs)

        def fake_dpotrf(a, *args, **kwargs):
            return (a, 1)

        monkeypatch.setattr("firthmodels.cox.dpotrf", fake_dpotrf)
        monkeypatch.setattr("firthmodels.cox.dpstrf", wrapped_dpstrf)

        model_fallback = FirthCoxPH(backend="numpy").fit(X, y)
        assert called["dpstrf"] > 0
        np.testing.assert_allclose(model_fallback.coef_, model_normal.coef_, rtol=1e-6)

    def test_rank_deficient_raises(self, monkeypatch):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(8)
        X = np.column_stack([x, x])  # rank deficient
        time = rng.uniform(1, 10, 8)
        event = rng.choice([True, False], size=8)
        y = _structured_y(event, time)

        def fake_dpotrf(a, *args, **kwargs):
            return (a, 1)

        monkeypatch.setattr("firthmodels.cox.dpotrf", fake_dpotrf)

        with pytest.raises(scipy.linalg.LinAlgError, match="rank deficient"):
            FirthCoxPH(backend="numpy").fit(X, y)

    def test_penalty_weight_zero_matches_lifelines_coxph(self):
        rng = np.random.default_rng(42)
        n = 100

        x1 = rng.standard_normal(n)
        x2 = rng.uniform(-1, 1, n)
        X = np.column_stack([x1, x2])

        beta_true = np.array([0.5, -0.3])
        eta = X @ beta_true

        baseline_hazard = 0.1
        survival_time = rng.exponential(1 / (baseline_hazard * np.exp(eta)))
        censor_time = rng.exponential(scale=10.0, size=n)

        time = np.minimum(survival_time, censor_time)
        event = survival_time <= censor_time
        y = _structured_y(event, time)

        model = FirthCoxPH(penalty_weight=0.0, backend="numpy")
        model.fit(X, y)

        # reference values from lifelines CoxPHFitter
        expected_coef = np.array([0.524213441484833, -0.027371010108854982])
        expected_loglik = -200.1427020304977
        expected_bse = np.array([0.17439918364226892, 0.24924619900371428])

        np.testing.assert_allclose(model.coef_, expected_coef, rtol=1e-5)
        np.testing.assert_allclose(model.loglik_, expected_loglik, rtol=1e-5)
        np.testing.assert_allclose(model.bse_, expected_bse, rtol=1e-5)


class TestConcordanceIndex:
    def test_counts_concordant_discordant_pairs(self):
        # 2 concordant, 1 discordant -> C = 2/3
        event = np.array([True, True, True])
        time = np.array([1.0, 2.0, 3.0])
        risk = np.array([2.0, 3.0, 1.0])
        np.testing.assert_allclose(_concordance_index(event, time, risk), 2 / 3)

    def test_event_and_censor_at_same_time_are_comparable(self):
        # Event at t=1, censor at t=1: the event is observed, so comparable
        event = np.array([True, False])
        time = np.array([1.0, 1.0])
        risk = np.array([2.0, 1.0])
        assert _concordance_index(event, time, risk) == 1.0
