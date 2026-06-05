#include "nn_utils.h"
#include <random>

void initialize_tensor_normal(Tensor& t, float mean, float stddev) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> d{mean, stddev};

    std::vector<float>& data = t.data_raw();
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = d(gen);
    }
}
