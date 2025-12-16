import numpy as np
import pytest

from firthmodels.cox import _validate_survival_y


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
