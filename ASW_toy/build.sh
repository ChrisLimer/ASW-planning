#!/bin/bash

set -e  # stop on error

cd open_spiel_extensions
rm -rf build
mkdir build
cd build
cmake ..
cmake --build . -j
cd ../..
