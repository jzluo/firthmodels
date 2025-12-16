import numpy as np
import pytest

from firthmodels.cox import FirthCoxPH, _CoxPrecomputed, _validate_survival_y


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

        pre = _CoxPrecomputed.from_data(X, time, event)

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

        model = FirthCoxPH(max_iter=200, tol=1e-10)
        model.fit(X, y)

        assert model.converged_
        np.testing.assert_allclose(model.coef_[0], np.log(3.0), rtol=1e-6, atol=1e-6)
