#!/bin/bash
#if [ ! -e /content/models ]; then
#        mkdir -p /root/.torch/models
#        mkdir -p /root/.fastai/data
#        ln -s /root/.torch/models /content
#        ln -s /root/.fastai/data /content
#        rm -rf /content/sample_data/
#fi

#echo Updating tensorflow to nightly...
#pip install tf-nightly --quiet > /dev/null
#echo Done.

echo Installing vi
sudo apt-get -qq install vim
echo Done

