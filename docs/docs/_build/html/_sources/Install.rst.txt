Install
=======

Open Git Bash. Change the current working directory to the location where you want
to clone this `GitHub <https://github.com/cwfparsonson/trafpy>`_ project, and run::

    $ git clone https://github.com/cwfparsonson/trafpy

In the project's root directory, run::

    $ python setup.py install

Then, still in the root directory, install the required packages with either pip::

    $ pip install -r requirements/default.txt

or conda::

    $ conda install --file requirements/default.txt


You should then be able to import TrafPy into your Python script from any directory
on your machine::

    >>> import trafpy
