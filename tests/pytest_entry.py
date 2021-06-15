import cProfile
import pstats

import pytest

with cProfile.Profile(builtins=False) as pr:
    pr.enable()
    pytest.main([])  # "tests/test_exp.py"
    pr.disable()

    pr.create_stats()
    pr.dump_stats("profile.pstat")

stat = pstats.Stats("profile.pstat")
stat.sort_stats(pstats.SortKey.TIME)
stat.print_stats("Desktop", 20)
