#pragma once

#include <cstddef>

class ProgressBar {
public:
    ProgressBar(size_t start, size_t end);
    void increment();
private:
    float curr_val;
    float end;
    float increment_val;
    size_t width = 70;
};
