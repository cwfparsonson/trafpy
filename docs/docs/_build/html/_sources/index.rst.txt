Overview of TrafPy
==================
TrafPy is a Python package for the generation, management and standardisation of 
network traffic.

TrafPy contains the following key modules:

- ``trafpy.generator``: A package for generating custom and/or literature traffic
  trace data which can be exported into universally compatible file formats (e.g.
  CSV, Pickle, JSON, etc.) and imported into any simulation, emulation, or experimentation
  environment. It also comes with an interactive Jupyter Notebook tool for visually building
  and shaping distributions.
- ``trafpy.benchmarker``: A package for generating, reproducing, and establishing
  standard network traffic benchmarks.
- ``trafpy.manager``: A package for simulating a data centre network with various
  routing and scheduling protocols following the standard OpenAI Gym reinforcement
  learning interface.

TrafPy can be used to quickly and easily replicate traffic distributions from the
literature even in the absense of raw open-access data. Furthermore, it is hoped
that TrafPy will help towards standardising the traffic patterns used by networks
researchers to benchmark their management systems.



Getting Started
===============
Follow the :doc:`instructions <Install>` to install TrafPy, then have a look 
at the :doc:`tutorial <tutorial>` and the `examples <https://github.com/cwfparsonson/trafpy/tree/master/examples>`_ on the
GitHub page.



Free Software
=============
TrafPy is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. Contributions are welcome. Check out the `guidelines <https://trafpy.readthedocs.io/en/latest/Contribute.html>`_
on how to contribute. Contact cwfparsonson@gmail.com for questions.





Documentation
=============

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Install
   tutorial
   Contribute
   License
   Citing
   setup
   trafpy



Index
=====

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
