#!/bin/bash -e

type wget 2>/dev/null || { echo "wget is not installed. Please install it using apt or yum." ; exit 1 ; }

CURRENTPATH=`pwd`
CYBERFOLDDIR="${CURRENTPATH}"

mkdir -p "${CYBERFOLDDIR}"
cd "${CYBERFOLDDIR}"
wget -q -P . https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash ./Mambaforge-Linux-x86_64.sh -b -p "${CYBERFOLDDIR}/conda"
rm Mambaforge-Linux-x86_64.sh

source "${CYBERFOLDDIR}/conda/etc/profile.d/conda.sh"
export PATH="${CYBERFOLDDIR}/conda/condabin:${PATH}"
conda update -n base conda -y
conda create -p "$CYBERFOLDDIR/cyberchipped-fold-conda" -c conda-forge -c bioconda \
    git python=3.10 openmm==7.7.0 pdbfixer \
    kalign2=2.04 hhsuite=3.3.0 mmseqs2=15.6f452 -y
conda activate "$CYBERFOLDDIR/cyberchipped-fold-conda"

# install Cyberchipped-fold and Jaxlib (AlphaFold2 only)
"$CYBERFOLDDIR/cyberchipped-fold-conda/bin/pip" install --no-warn-conflicts \
    "cyberchipped-fold[alphafold-minus-jax] @ git+https://github.com/truemagic-coder/cyberchipped-fold"
"$CYBERFOLDDIR/cyberchipped-fold-conda/bin/pip" install --upgrade "jax[cuda12]"==0.4.28
"$CYBERFOLDDIR/cyberchipped-fold-conda/bin/pip" install --upgrade tensorflow
"$CYBERFOLDDIR/cyberchipped-fold-conda/bin/pip" install silence_tensorflow

pushd "${CYBERFOLDDIR}/cyberchipped-fold-conda/lib/python3.10/site-packages/cyberchipped_fold"
# Use 'Agg' for non-GUI backend
sed -i -e "s#from matplotlib import pyplot as plt#import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt#g" plot.py
# modify the default params directory
sed -i -e "s#appdirs.user_cache_dir(__package__ or \"cyberchipped-fold\")#\"${CYBERFOLDDIR}/cyberchipped-fold\"#g" download.py
# suppress warnings related to tensorflow
sed -i -e "s#from io import StringIO#from io import StringIO\nfrom silence_tensorflow import silence_tensorflow\nsilence_tensorflow()#g" batch.py
# remove cache directory
rm -rf __pycache__
popd

# Download weights
"$CYBERFOLDDIR/cyberchipped-fold-conda/bin/python3" -m cyberchipped_fold.download
echo "Download of AlphaFold2 weights finished."
echo "-----------------------------s------------"
echo "Installation of Cyberchipped-Fold finished."
echo "Add ${CYBERFOLDDIR}/cyberchipped-fold-conda/bin to your PATH environment variable to run 'colabfold_batch' and 'colabfold_search'."
echo -e "i.e. for Bash:\n\texport PATH=\"${CYBERFOLDDIR}/cyberchipped-fold-conda/bin:\$PATH\""
echo "For more details, please run 'colabfold_batch --help' or 'colabfold_search --help'."
