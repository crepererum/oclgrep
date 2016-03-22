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

## Limitations
Because it's an proof-of-concept there are several things missing here:
- **Incomplete regex parser:** While the graph representation allows you do encode most (all?) regex inputs that do not rely on group capture, the regex parser is very incomplete. (e.g. no predefined character classes, no grouping, no escaping)
- **UTF32 overhead:** To simplify the OpenCL kernel, the input is currently converted into UTF32. For latin-based inputs, this results in 4 times larger input data compared to the original UTF8 text. While the conversion could be done by the kernel itself I'm not sure how efficient that would be. Also there would be other problems (like load balancing for texts with a huge amount of non-latin chars).
- **UI:** The output format is currently quite messy.
- **Local memory caching:** The caching mechanism used by the kernel is very inefficient.
- **Collector:** The collector scan implementation is bad. It could be way more efficient, but at the same time it gets more complex.
- **Tests:** There are currently no tests, not even simple ones.
- **Documentation:** non-existent, not even for the binary graph format

Be aware that this was only intended to be a prototype!

## License
Copyright © 2016 Marco Neumann

Distributed under the MIT License (MIT).
