# mysca

## About

## Setup

Create a project environment as follows:

```bash
mamba env create -p ./env -f environment.yml
conda activate env
```

Next install the project source code.
From the project directory, activate the environment and run:

```bash
conda activate env
python -m pip install -e '.[dev]'
```

Verify things have installed successfully by running:

```bash
pytest tests
```

We use a separate environment with an installation of [ClustalOmega](http://www.clustal.org/omega) in order to create Multiple Sequence Alginments (MSAs). Specifically, we download the [precompiled binary](http://www.clustal.org/omega/#Download), and link this to an empty conda environment as so:

```bash
# Download the binary to a specific directory
cd <software-directory>
wget http://www.clustal.org/omega/clustalo-1.2.4-Ubuntu-x86_64
# Initialize an empty conda environment
conda create -n clustalo-env
conda activate clustalo-env
ln -s <software-directory>/clustalo-1.2.4-Ubuntu-x86_64 $CONDA_PREFIX/bin/clustalo
```

## Directories

## Links

* [ClustalOmega](https://www.ebi.ac.uk/jdispatcher/msa/clustalo?stype=protein&outfmt=fa)
* [pySCA and tutorial](https://ranganathanlab.gitlab.io/pySCA/)
* [pySCA repo](https://github.com/ranganathanlab/pySCA)
* [pySCA data](https://github.com/ranganathanlab/pySCA-data)
* [pySCA S1A notebook](https://github.com/ranganathanlab/pySCA/blob/master/notebooks/SCA_S1A.ipynb)
* [About Protein Family Models](https://www.ncbi.nlm.nih.gov/genome/annotation_prok/evidence/)
* [nirB PFM](https://www.ncbi.nlm.nih.gov/genome/annotation_prok/evidence/TIGR02374/)

## References

[1] N. Halabi, O. Rivoire, S. Leibler, and R. Ranganathan, Protein Sectors: Evolutionary Units of Three-Dimensional Structure, Cell 138, 774 (2009).

[2] O. Rivoire, K. A. Reynolds, and R. Ranganathan, Evolution-Based Functional Decomposition of Proteins, PLoS Comput Biol 12, e1004817 (2016).
