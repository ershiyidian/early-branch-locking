"""Dependency-light validation for the public research package."""
from early_branch_locking.core.countdown_shared import enumerate_solution_set
from early_branch_locking.core.entrance_detection import find_first_reasoning_entrance


def test_public_api() -> None:
    assert enumerate_solution_set([2, 3], 5)
    assert find_first_reasoning_entrance("<think>2 + 3 = 5</think>").found


def test_package_metadata() -> None:
    import early_branch_locking
    assert early_branch_locking.__package__ == "early_branch_locking"
    print("smoke tests passed")
