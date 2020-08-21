Contribute
==========

This guide will help you contribute to e.g. fix a bug or add a new feature for
TrafPy.

Development Workflow
--------------------

1. If you are a first-time contributor:

   * Go to `https://github.com/cwfparsonson/trafpy
     <https://github.com/cwfparsonson/trafpy>`_ and click the
     "fork" button to create your own copy of the project.

   * Clone the project to your local computer::

      git clone git@github.com:your-username/trafpy.git

   * Navigate to the folder trafpy and add the upstream repository::

      git remote add upstream git@github.com:cwfparsonson/trafpy.git

   * Now, you have remote repositories named:

     - ``upstream``, which refers to the ``trafpy`` repository
     - ``origin``, which refers to your personal fork

   * Next, you need to set up your build environment.
     Here are instructions for two popular environment managers:
   
     * ``venv`` (pip based)
     
       ::
     
         # Create a virtualenv named ``trafpy-dev`` that lives in the directory of
         # the same name
         python -m venv trafpy-dev
         # Activate it
         source trafpy-dev/bin/activate
         # Install main development and runtime dependencies of trafpy 
         pip install -r <(cat requirements/{default,docs}.txt)
         #
         # These packages require that you have your system properly configured
         # and what that involves differs on various systems.
         #
         # In the trafpy root directory folder, run
         python setup.py develop
         # Test your installation in a .py file
         import trafpy.generator as tpg
         from trafpy.manager import Demand, DCN, SRPT, RWA
         
     
     * ``conda`` (Anaconda or Miniconda)
    
       ::
 
         # Create a conda environment named ``trafpy-dev``
         conda create --name trafpy-dev
         # Activate it
         conda activate trafpy-dev
         # Install main development and runtime dependencies of trafpy 
         conda install -c conda-forge `for i in requirements/{default,doc}.txt; do echo -n " --file $i "; done`
         #
         # These packages require that you have your system properly configured
         # and what that involves differs on various systems.
         #
         # In the trafpy root directory folder, run
         python setup.py develop
         # Test your installation in a .py file
         import trafpy.generator as tpg
         from trafpy.manager import Demand, DCN, SRPT, RWA

   * Finally, it is recommended you use a pre-commit hook, which runs black when
     you type ``git commit``::

       pre-commit install

2. Develop your contribution:

   * Pull the latest changes from upstream::

      git checkout master
      git pull upstream master

   * Create a branch for the feature you want to work on. Since the
     branch name will appear in the merge message, use a sensible name
     such as 'bugfix-for-issue-1480'::

      git checkout -b bugfix-for-issue-1480

   * Commit locally as you progress (``git add`` and ``git commit``)

3. Submit your contribution:

   * Push your changes back to your fork on GitHub::

      git push origin bugfix-for-issue-1480

   * Go to GitHub. The new branch will show up with a green Pull Request
     button---click it.

   * If you want, email cwfparsonson@gmail.com to explain your changes or to ask 
     for review.

4. Review process:

   * Your pull request will be reviewed.

   * To update your pull request, make your changes on your local repository
     and commit. As soon as those changes are pushed up (to the same branch as
     before) the pull request will update automatically.

   .. note::

      If the PR closes an issue, make sure that GitHub knows to automatically
      close the issue when the PR is merged.  For example, if the PR closes
      issue number 1480, you could use the phrase "Fixes #1480" in the PR
      description or commit message.

5. Document changes

   If your change introduces any API modifications, please update
   ``doc/release/release_dev.rst``.

   If your change introduces a deprecation, add a reminder to
   ``doc/developer/deprecations.rst`` for the team to remove the
   deprecated functionality in the future.

   .. note::
   
      To reviewers: make sure the merge message has a brief description of the
      change(s) and if the PR closes an issue add, for example, "Closes #123"
      where 123 is the issue number.


Divergence from ``upstream master``
-----------------------------------

If GitHub indicates that the branch of your Pull Request can no longer
be merged automatically, merge the master branch into yours::

   git fetch upstream master
   git merge upstream/master

If any conflicts occur, they need to be fixed before continuing.  See
which files are in conflict using::

   git status

Which displays a message like::

   Unmerged paths:
     (use "git add <file>..." to mark resolution)

     both modified:   file_with_conflict.txt

Inside the conflicted file, you'll find sections like these::

   <<<<<<< HEAD
   The way the text looks in your branch
   =======
   The way the text looks in the master branch
   >>>>>>> master

Choose one version of the text that should be kept, and delete the
rest::

   The way the text looks in your branch

Now, add the fixed file::


   git add file_with_conflict.txt

Once you've fixed all merge conflicts, do::

   git commit








