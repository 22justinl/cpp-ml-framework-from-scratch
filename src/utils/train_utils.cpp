#include "train_utils.h"

#include <iomanip>
#include <iostream>

ProgressBar::ProgressBar(size_t start, size_t end): curr_val(start), end(end), increment_val(1) {}

void ProgressBar::increment() {
    curr_val += increment_val;
    std::cout << " [";
    float progress = curr_val/end;
    size_t count = width * progress;
    for (size_t i = 0; i < width; ++i) {
        if (i <= count) {
            std::cout << "=";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << std::setprecision(4) << std::left << std::setw(6) << 100*progress << "%\r";
    std::cout.flush();
    if (curr_val >= end) {
        std::cout << std::endl;
    }
}
