import os
from CheXpert2.Sampler import Sampler



def test_sampler():


    sampler = Sampler("data_test")
    samples = sampler.sampler()  # probably gonna break?


if __name__ == "__main__":

    test_sampler()
