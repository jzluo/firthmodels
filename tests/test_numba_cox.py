import numpy as np
import pytest

from firthmodels import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    from firthmodels._numba.logistic import precompute_cox


pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")


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
