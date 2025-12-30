import numpy as np
import pytest

from firthmodels import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    from firthmodels._numba.cox import precompute_cox

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
        # np.testing.assert_allclose(model.bse_, expected_bse, rtol=1e-6)

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

        # ci = model.conf_int(method="pl")
        # # Profile likelihood CIs from coxphf (in log scale)
        # expected_ci = np.array(
        #     [
        #         [1.9343469272, 8.7215676015],  # separator
        #         [0.2921186188, 0.9164270978],  # x1
        #         [-0.9217200647, 0.0699415336],  # x2
        #         [0.4348431279, 1.6550461482],  # x3
        #     ]
        # )

        # np.testing.assert_allclose(ci, expected_ci, rtol=1e-6)
