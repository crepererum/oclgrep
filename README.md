# OCLGrep
This is a proof-of-concept OpenCL-driven regex search.

## Runtime Requirements
A OpenCL 1.1 device and a Linux PC.

## Build Instruction
To build this you'll need:
- C++14 compiler (tested with clang and GCC)
- a rather up-to-date boost library

Then building is kinda easy:

    mkdir build
    cd build
    cmake ..
    env LIBRARY_PATH=/opt/cuda-7.5/lib64 make
    cd ..

## Usage
Actually the current work dir doesn't matter because OpenCL files are build into the executable (keep that in mind in case you change them and remember to rebuild):

    ./build/oclgrep --help
    ./build/oclgrep foo test.txt
    ./build/oclgrep "[abcdefg]{1,3}[aijklmop]{1,5}[abcdefjijklmnop]{0,2}[qrstu]{4,10}[abc]{2}" big.1.txt --print-profile --max-chunk-size 33554432 --no-output

## License
Copyright © 2016 Marco Neumann

Distributed under the MIT License (MIT).
