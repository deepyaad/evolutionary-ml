
import assignta as ata
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def cases():
    test1 = np.array(pd.read_csv('data/test1.csv', header=None))
    test2 = np.array(pd.read_csv('data/test2.csv', header=None))
    test3 = np.array(pd.read_csv('data/test3.csv', header=None))
    return test1, test2, test3

def test_overallocation(cases):
    test1, test2, test3 = cases
    assert ata.overallocation(test1) == 34, "Incorrect overallocation score for test1"
    assert ata.overallocation(test2) == 37, "Incorrect overallocation score for test2"
    assert ata.overallocation(test3) == 19, "Incorrect overallocation score for test3"

def test_conflicts(cases):
    test1, test2, test3 = cases
    assert ata.conflicts(test1) == 7, "Incorrect conflicts score for test1"
    assert ata.conflicts(test2) == 5, "Incorrect conflicts score for test2"
    assert ata.conflicts(test3) == 2, "Incorrect conflicts score for test3"

def test_undersupport(cases):
    test1, test2, test3 = cases
    assert ata.undersupport(test1) == 1, "Incorrect undersupport score for test1"
    assert ata.undersupport(test2) == 0, "Incorrect undersupport score for test2"
    assert ata.undersupport(test3) == 11, "Incorrect undersupport score for test3"

def test_unavailable(cases):
    test1, test2, test3 = cases
    assert ata.unavailable(test1) == 59, "Incorrect unwilling score for test1"
    assert ata.unavailable(test2) == 57, "Incorrect unwilling score for test2"
    assert ata.unavailable(test3) == 34, "Incorrect unwilling score for test3"

def test_unpreferred(cases):
    test1, test2, test3 = cases
    assert ata.unpreferred(test1) == 10, "Incorrect unpreferred score for test1"
    assert ata.unpreferred(test2) == 16, "Incorrect unpreferred score for test2"
    assert ata.unpreferred(test3) == 17, "Incorrect unpreferred score for test3"
