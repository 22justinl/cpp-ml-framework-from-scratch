#!/usr/bin/env zsh
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ln -s build/compile_commands.json .
cmake --build build
