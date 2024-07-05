Working from https://github.com/juliankappler/lunar-lander/

Three main training methods:

1. actor-critic
2. deep Q-learning
3. double deep Q-learning

Things to investigate:

1. Playing interactively as a human? Now feasible on Windows at https://github.com/mikerenfro/OpenAIGaming -- what makes the game difficult for a human to play?
2. Comparison of training times (across algorithms, across hardware types)
3. Inherent randomness in training scenarios (and learning)


## Making environment on HPC

```
spack load py-torch py-cloudpickle@2 py-h5py py-moviepy swig
spack load py-virtualenv
virtualenv lunarlander
source lunarlander/bin/activate
pip install --dry-run 'gymnasium[box2d]' # verify only Farama, box2d, gymnasium, pygame, swig packages to be installed.
deactivate # if needed to 'spack load' other packages
```
If Jupyter needed, add py-jupyter to the first spack load list above, then after activating virtual environment:
```
python -m ipykernel install --user --name lunarlander
# also, can do "jupyter kernelspec remove lunarlander"
```