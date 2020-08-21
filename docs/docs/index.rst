Overview of TrafPy
==================
TrafPy is a Python package for the generation, management and standardisation of 
network traffic.

TrafPy provides:

- a pre-built **interactive Jupyter Notebook** tool for visually building distributions and 
  data which acccurately mimic traffic characteristics of real networks (e.g. data centres);
- a **generator** package for generating network traffic which can be flexibly integrated 
  into custom Python projects; and
- a **manager** package which can be used to simulate network management (scheduling,
  routing etc.) following the standard OpenAI Gym reinforcement learning framework.

TrafPy can be used to quickly and easily replicate traffic distributions from the
literature even in the absense of raw open-access data. Furthermore, it is hoped
that TrafPy will help towards standardising the traffic patterns used by networks
researchers to benchmark their management systems.



Getting Started
===============
Follow the :doc:`instructions <Install>` to install TrafPy, then have a look 
at the :doc:`tutorial <Tutorial>`.



Free Software
=============
TrafPy will one day be open-source. For now, it is a private repo found on
`GitHub <https://github.com/cwfparsonson/trafpy>`_. Contact cwfparsonson@gmail.com
to request access.





Documentation
=============

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Install
   Tutorial
   Contribute
   License
   Citing



Index
=====

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
