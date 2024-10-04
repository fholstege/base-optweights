

# import the add_up function from optweights
import pytest
from optweights.weights import add_up



def test_add_up():
    
    test_input = 5
    expected_output = 6

    assert add_up(test_input) == expected_output


# if main is run, run the tests
if __name__ == "__main__":
    test_add_up()
    