#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

mkdir -p $base/venvs

python -m virtualenv -p ~/.pyenv/versions/3.8.6/bin/python $base/venvs/torch3

echo "To activate your environment:"
echo "    source $base/venvs/torch3/bin/activate"

