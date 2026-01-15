import numpy as np
import pytest
import scipy.linalg

from firthmodels import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    from firthmodels._numba.cox import (
        _STATUS_LINALG_FAIL,
        _STATUS_RANK_DEFICIENT,
        _STATUS_STEP_HALVING_FAILED,
        concordance_index,
        newton_raphson_cox,
        precompute_cox,
    )
    from firthmodels._numba.cox import (
        compute_cox_quantities as compute_cox_quantities_numba,
    )

from firthmodels.cox import (
    FirthCoxPH,
    _concordance_index,
    _CoxPrecomputed,
    _validate_survival_y,
    _Workspace,
    compute_cox_quantities,
)

pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")


def _structured_y(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    y = np.empty(len(time), dtype=[("event", bool), ("time", np.float64)])
    y["event"] = event
    y["time"] = time
    return y


class TestNumbaCoxPrecomputed:
    def test_blocks_event_counts_and_sums_numba(self):
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

        (_, _, _, block_ends, block_d, block_s) = precompute_cox(X, time, event)

        # Sorted times: 5,5,4,3,2,2 -> block ends at [2,3,4,6]
        np.testing.assert_array_equal(block_ends, np.array([2, 3, 4, 6]))
        np.testing.assert_array_equal(block_d, np.array([2, 0, 1, 1]))

        expected_block_s = np.array(
            [
                [60.0, 6.0],  # time 5: rows [20,2] and [40,4] had events
                [0.0, 0.0],  # time 4: no events
                [50.0, 5.0],  # time 3: row [50,5]
                [10.0, 1.0],  # time 2: row [10,1]
            ]
        )
        np.testing.assert_array_equal(block_s, expected_block_s)


class TestFirthCoxPH:
    def test_two_individual_example_matches_log3(self):
        # (Heinze and Schemper, 2001), Section 2: two individuals, one covariate.
        # The modified score has root exp(beta_hat) = 3.
        X = np.array([[1.0], [0.0]])
        time = np.array([1.0, 2.0])
        event = np.array([True, False])
        y = _structured_y(event, time)

        model = FirthCoxPH(backend="numba")
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

        model = FirthCoxPH(backend="numba").fit(X, y)

        np.testing.assert_array_equal(model.unique_times_, [1.0])
        np.testing.assert_allclose(model.cum_baseline_hazard_, [0.25], rtol=1e-6)
        np.testing.assert_allclose(
            model.baseline_survival_, np.exp(-model.cum_baseline_hazard_)
        )

    def test_matches_coxphf_with_monotone_likelihood(self, cox_separation_data):
        """Matches coxphf on data with monotone likelihood."""
        X, time, event = cox_separation_data
        y = _structured_y(event, time)

        model = FirthCoxPH(backend="numba")
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
        pre = _CoxPrecomputed.from_data(X, time, event, backend="numba")
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

        model = FirthCoxPH(penalty_weight=0.0, backend="numba")
        model.fit(X, y)

        # reference values from lifelines CoxPHFitter
        expected_coef = np.array([0.524213441484833, -0.027371010108854982])
        expected_loglik = -200.1427020304977
        expected_bse = np.array([0.17439918364226892, 0.24924619900371428])

        np.testing.assert_allclose(model.coef_, expected_coef, rtol=1e-5)
        np.testing.assert_allclose(model.loglik_, expected_loglik, rtol=1e-5)
        np.testing.assert_allclose(model.bse_, expected_bse, rtol=1e-5)

    def test_numba_rank_deficient_raises(self):
        """Numba backend detects rank deficiency via dpstrf fallback."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(8)
        X = np.column_stack([x, x])  # rank deficient
        time = rng.uniform(1, 10, 8)
        event = rng.choice([True, False], size=8)
        y = _structured_y(event, time)

        with pytest.raises(scipy.linalg.LinAlgError, match="rank deficient"):
            FirthCoxPH(backend="numba").fit(X, y)


class TestNewtonRaphsonCox:
    def test_step_halving_failure_returns_consistent_fisher_info(self):
        # Dataset chosen to deterministically hit the step-halving failure path.
        # seed=0 with these parameters triggers failure at iteration 12.
        np.random.seed(0)
        n, k = 10, 2
        X = np.random.randn(n, k) * 3
        time = np.abs(np.random.randn(n)) + 0.1
        event = np.random.randint(0, 2, n).astype(bool)
        event[0] = True  # ensure at least one event

        pre = _CoxPrecomputed.from_data(X, time, event, backend="numba")
        workspace = _Workspace(pre.n_samples, pre.n_features)

        beta, loglik, fisher_info, n_iter, status = newton_raphson_cox(
            X=pre.X,
            block_ends=pre.block_ends,
            block_d=pre.block_d,
            block_s=pre.block_s,
            max_iter=50,
            max_step=5.0,
            max_halfstep=2,
            gtol=1e-10,
            xtol=1e-10,
            workspace=workspace.numba_buffers(),
        )
        assert status == _STATUS_STEP_HALVING_FAILED
        fisher_info = fisher_info.copy()

        # Recompute quantities for the returned beta to verify the returned
        # fisher_info corresponds to the accepted (not rejected) beta.
        ref_workspace = _Workspace(pre.n_samples, pre.n_features)
        fisher_work = np.empty((k, k), dtype=np.float64, order="F")
        modified_score = np.empty(k, dtype=np.float64)
        x_bar = np.empty(k, dtype=np.float64)
        Ix = np.empty(k, dtype=np.float64)
        term1 = np.empty(k, dtype=np.float64)
        term23 = np.empty(k, dtype=np.float64)

        loglik_ref, status = compute_cox_quantities_numba(
            X=pre.X,
            block_ends=pre.block_ends,
            block_d=pre.block_d,
            block_s=pre.block_s,
            beta=beta,
            fisher_work=fisher_work,
            modified_score=modified_score,
            x_bar=x_bar,
            Ix=Ix,
            term1=term1,
            term23=term23,
            workspace=ref_workspace.numba_buffers(),
        )
        assert status == 0
        np.testing.assert_allclose(loglik_ref, loglik, rtol=1e-10)
        np.testing.assert_allclose(ref_workspace.fisher_info, fisher_info, rtol=1e-10)


class TestNumbaConcordanceIndex:
    def test_counts_concordant_discordant_pairs_numba(self):
        # 2 concordant, 1 discordant -> C = 2/3
        event = np.array([True, True, True])
        time = np.array([1.0, 2.0, 3.0])
        risk = np.array([2.0, 3.0, 1.0])
        np.testing.assert_allclose(concordance_index(event, time, risk), 2 / 3)

    def test_event_and_censor_at_same_time_are_comparable_numba(self):
        # Event at t=1, censor at t=1: the event is observed, so comparable
        event = np.array([True, False])
        time = np.array([1.0, 1.0])
        risk = np.array([2.0, 1.0])
        assert concordance_index(event, time, risk) == 1.0
