TrafPy Distributions
====================

Network traffic patterns can be characterised by **distributions**. By
accurately describing a distribution, one can sample from it to generate
arbitrary amounts of realistic network traffic. This is fundamentally how
TrafPy generates traffic; by sampling from pre-defined discrete distributions.

TrafPy distributions are defined as **hashtables** (Python dictionaries).
These tables map each possible value taken by the random variable to some fractional
value between 0 and 1 (where this value could be e.g. probability, fraction of network requested, etc.).
The fractional values should sum to 1.0.

E.g. If you have integer random variable values 1-10 each with an equal probability of occurring,
this distribution would be represtented by the following hash table in TrafPy::

    dist = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1, 10: 0.1}

Similarly, a distribution where a random variable value of 5 always occurs would be::
    
    dist = {5: 1.0}

and so on.

TrafPy distributions are typically referred to as either **value distributions** which map
random variables such as flow size and inter-arrival times to 'probability of
occurring', or **node distributions** which map node pairs to 'fraction of the overall 
traffic load requested'.


