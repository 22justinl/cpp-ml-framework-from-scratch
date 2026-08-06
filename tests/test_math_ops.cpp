#include "doctest.h"

#include "core/tensor.h"
#include "ops/math_ops.h"
#include "utils/tensor_utils.h"

#include <numbers>

TEST_CASE("Tensor add") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4});
    Tensor t2({7,6,5,4,3,2,1,0}, {2,4});
    Tensor t3({0,1,2,3,4,5,6,7}, {4,2});

    Tensor add1 = t1 + t2;
    Tensor add2 = t2 + t1;
    for (size_t i = 0; i < 8; ++i) {
        CHECK(add1.data_raw()[i] == 7);
        CHECK(add2.data_raw()[i] == 7);
    }
    CHECK_THROWS(t1+t3);
    CHECK_THROWS(t2+t3);
}

TEST_CASE("Tensor sub") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4});
    Tensor t2({0,1,2,3,4,5,6,7}, {2,4});
    Tensor t3({0,1,2,3,4,5,6,7}, {4,2});

    Tensor sub1 = t1 - t2;
    Tensor sub2 = t2 - t1;
    for (size_t i = 0; i < 8; ++i) {
        CHECK(sub1.data_raw()[i] == 0);
        CHECK(sub2.data_raw()[i] == 0);
    }
    CHECK_THROWS(t1-t3);
    CHECK_THROWS(t2-t3);
}

TEST_CASE("Tensor mul") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4});
    Tensor t2({2,2,2,2,2,2,2,2}, {2,4});
    Tensor t3({0,1,2,3,4,5,6,7}, {4,2});

    Tensor mul1 = t1 * t2;
    Tensor mul2 = t2 * t1;
    for (size_t i = 0; i < 8; ++i) {
        CHECK(mul1.data_raw()[i] == 2*i);
        CHECK(mul2.data_raw()[i] == 2*i);
    }
    CHECK_THROWS(t1*t3);
    CHECK_THROWS(t2*t3);
}

TEST_CASE("Tensor div") {
    Tensor t1({0,2,4,6,8,10,12,14}, {2,4});
    Tensor t2({2,2,2,2,2,2,2,2}, {2,4});
    Tensor t3({0,1,2,3,4,5,6,7}, {4,2});

    Tensor div1 = t1 / t2;
    for (size_t i = 0; i < 8; ++i) {
        CHECK(div1.data_raw()[i] == i);
    }
}

TEST_CASE("Tensor scalar_mul") {
    float f1 = 2;
    float f2 = -3;
    Tensor t1({1,2,3,4},{4});
    Tensor t2({1,2,3,4,5,6,7,8},{2,4});
    Tensor res1 = scalar_mul(f1, t1);
    Tensor res2 = scalar_mul(t1, f2);
    Tensor res3 = scalar_mul(f2, t2);
    Tensor res4 = scalar_mul(t2, f1);
    CHECK(check_tensor_equal(res1, Tensor({2,4,6,8},{4})));
    CHECK(check_tensor_equal(res2, Tensor({-3,-6,-9,-12},{4})));
    CHECK(check_tensor_equal(res3, Tensor({-3,-6,-9,-12,-15,-18,-21,-24},{2,4})));
    CHECK(check_tensor_equal(res4, Tensor({2,4,6,8,10,12,14,16},{2,4})));
}

TEST_CASE("Tensor matmul") {
    Tensor t1({0,1,2,3,4,5,6,7,8,9,10,11}, {3,4});
    Tensor t2({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {4,5});
    Tensor t3({1,1,1}, {1,3});
    Tensor res1 = matmul(t1,t2);
    Tensor res2 = matmul(t3,t1);
    Tensor expected1({70,76,82,88,94,190,212,234,256,278,310,348,386,424,462}, {3,5});
    Tensor expected2({12,15,18,21}, {1,4});
    CHECK(check_tensor_equal(res1, expected1));
    CHECK(check_tensor_equal(res2, expected2));
}
TEST_CASE("Tensor scaled matmul") {
    Tensor t1({0,1,2,3,4,5,6,7,8,9,10,11}, {3,4});
    Tensor t2({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {4,5});
    Tensor t3({1,1,1}, {1,3});
    Tensor res1 = scaled_matmul(2,t1,t2);
    Tensor res2 = scaled_matmul(3,t3,t1);
    Tensor expected1({70*2,76*2,82*2,88*2,94*2,190*2,212*2,234*2,256*2,278*2,310*2,348*2,386*2,424*2,462*2}, {3,5});
    Tensor expected2({12*3,15*3,18*3,21*3}, {1,4});
    CHECK(check_tensor_equal(res1, expected1));
    CHECK(check_tensor_equal(res2, expected2));
}
TEST_CASE("Tensor mmadd") {
    Tensor t1({0,1,2,3,4,5,6,7,8,9,10,11}, {3,4});
    Tensor t2({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {4,5});
    Tensor t3({1,1,1}, {1,3});
    Tensor t4(1,{3,5});
    Tensor t5(1, {1,4});
    Tensor res1 = mmadd(t1,t2,t4);
    Tensor res2 = mmadd(t3,t1,t5);
    Tensor expected1({70+1,76+1,82+1,88+1,94+1,190+1,212+1,234+1,256+1,278+1,310+1,348+1,386+1,424+1,462+1}, {3,5});
    Tensor expected2({12+1,15+1,18+1,21+1}, {1,4});
    CHECK(check_tensor_equal(res1, expected1));
    CHECK(check_tensor_equal(res2, expected2));
}
TEST_CASE("Tensor mmadd general (GEMM)") {
    Tensor t1({0,1,2,3,4,5,6,7,8,9,10,11}, {3,4});
    Tensor t2({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {4,5});
    Tensor t3({1,1,1}, {1,3});
    Tensor t4(1,{3,5});
    Tensor t5(1, {1,4});
    Tensor res1 = mmadd_general(2,t1,t2,3,t4);
    Tensor res2 = mmadd_general(5,t3,t1,-10,t5);
    Tensor expected1({70*2+3,76*2+3,82*2+3,88*2+3,94*2+3,190*2+3,212*2+3,234*2+3,256*2+3,278*2+3,310*2+3,348*2+3,386*2+3,424*2+3,462*2+3}, {3,5});
    Tensor expected2({12*5-10,15*5-10,18*5-10,21*5-10}, {1,4});
    CHECK(check_tensor_equal(res1, expected1));
    CHECK(check_tensor_equal(res2, expected2));
}

TEST_CASE("Tensor matmul weird shape") {
    Tensor t1(1, {931,583});
    Tensor t2(2, {583,285});
    Tensor res1 = matmul(t1,t2);
    Tensor expected1(1166, {931, 285});
    CHECK(check_tensor_equal(res1, expected1));
}
TEST_CASE("Tensor matmul large 1") {
    Tensor t1(1, {512,1024});
    Tensor t2(2, {1024,1024});
    Tensor res1 = matmul(t1,t2);
    Tensor expected1(2048, {512,1024});
    CHECK(check_tensor_equal(res1, expected1));
}
TEST_CASE("Tensor matmul large 2") {
    Tensor t1(0, {67,5});
    Tensor t2(0, {5,2});
    for (size_t i = 0; i < 67*5; ++i) {
        t1.impl()->storage->data[i] = i;
    }
    for (size_t i = 0; i < 5*2; ++i) {
        t2.impl()->storage->data[i] = i;
    }

    Tensor res1 = matmul(t1,t2);
    Tensor expected1({60,   70,  160,  195,  260,  320,  360,  445,  460,  570,  560,  695,
          660,  820,  760,  945,  860, 1070,  960, 1195, 1060, 1320, 1160, 1445,
         1260, 1570, 1360, 1695, 1460, 1820, 1560, 1945, 1660, 2070, 1760, 2195,
         1860, 2320, 1960, 2445, 2060, 2570, 2160, 2695, 2260, 2820, 2360, 2945,
         2460, 3070, 2560, 3195, 2660, 3320, 2760, 3445, 2860, 3570, 2960, 3695,
         3060, 3820, 3160, 3945, 3260, 4070, 3360, 4195, 3460, 4320, 3560, 4445,
         3660, 4570, 3760, 4695, 3860, 4820, 3960, 4945, 4060, 5070, 4160, 5195,
         4260, 5320, 4360, 5445, 4460, 5570, 4560, 5695, 4660, 5820, 4760, 5945,
         4860, 6070, 4960, 6195, 5060, 6320, 5160, 6445, 5260, 6570, 5360, 6695,
         5460, 6820, 5560, 6945, 5660, 7070, 5760, 7195, 5860, 7320, 5960, 7445,
         6060, 7570, 6160, 7695, 6260, 7820, 6360, 7945, 6460, 8070, 6560, 8195,
         6660, 8320}, {67,2});
    CHECK(check_tensor_equal(res1, expected1));
}
TEST_CASE("Tensor matmul large 3") {
    Tensor t1(0, {2,5});
    Tensor t2(0, {5,67});
    for (size_t i = 0; i < 2*5; ++i) {
        t1.impl()->storage->data[i] = i;
    }
    for (size_t i = 0; i < 5*67; ++i) {
        t2.impl()->storage->data[i] = i;
    }

    Tensor res1 = matmul(t1,t2);
    Tensor expected1({2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120,
         2130, 2140, 2150, 2160, 2170, 2180, 2190, 2200, 2210, 2220, 2230, 2240,
         2250, 2260, 2270, 2280, 2290, 2300, 2310, 2320, 2330, 2340, 2350, 2360,
         2370, 2380, 2390, 2400, 2410, 2420, 2430, 2440, 2450, 2460, 2470, 2480,
         2490, 2500, 2510, 2520, 2530, 2540, 2550, 2560, 2570, 2580, 2590, 2600,
         2610, 2620, 2630, 2640, 2650, 2660, 2670, 5360, 5395, 5430, 5465, 5500,
         5535, 5570, 5605, 5640, 5675, 5710, 5745, 5780, 5815, 5850, 5885, 5920,
         5955, 5990, 6025, 6060, 6095, 6130, 6165, 6200, 6235, 6270, 6305, 6340,
         6375, 6410, 6445, 6480, 6515, 6550, 6585, 6620, 6655, 6690, 6725, 6760,
         6795, 6830, 6865, 6900, 6935, 6970, 7005, 7040, 7075, 7110, 7145, 7180,
         7215, 7250, 7285, 7320, 7355, 7390, 7425, 7460, 7495, 7530, 7565, 7600,
         7635, 7670}, {2,67});
    CHECK(check_tensor_equal(res1, expected1));
}
TEST_CASE("Tensor mmadd general large") {
    Tensor t1(1, {1024, 4096});
    Tensor t2(2, {4096, 1024});
    Tensor t3(4, {1024, 1024});
    float beta = 0.5;
    Tensor res = mmadd_general(1, t1, t2, beta, t3);
    Tensor expected(8194, {1024,1024});
    CHECK(check_tensor_equal(res, expected));
}

TEST_CASE("Tensor matmul (error cases)") {
    Tensor t1(0.f, {10,20});
    Tensor t2(0.f, {10,30});
    Tensor t3(0.f, {10,20});
    CHECK_THROWS(matmul(t1, t2));
    CHECK_THROWS(matmul(t2, t1));
    CHECK_THROWS(matmul(t1, t3));
    CHECK_THROWS(matmul(t3, t1));
}

TEST_CASE("Tensor dot") {
    Tensor t1({1,2,3}, {3});
    Tensor t2({-1,-2,3}, {3});
    Tensor d1 = dot(t1, t2);
    CHECK(d1({0}) == 4);
}
TEST_CASE("Tensor dot (error cases)") {
    Tensor t1({1,2,3}, {3});
    Tensor t2({-1,-2,3}, {3});
    Tensor t3({1,2,3}, {3,1});
    Tensor t4({-1,-2,3}, {3,1});
    Tensor t5({1,2,3}, {1,3});
    Tensor t6({-1,-2,3}, {1,3});

    CHECK_THROWS(dot(t1, t3));
    CHECK_THROWS(dot(t1, t5));
    CHECK_THROWS(dot(t3, t2));
    CHECK_THROWS(dot(t3, t6));
    CHECK_THROWS(dot(t6, t1));
    CHECK_THROWS(dot(t6, t3));
}

TEST_CASE("Tensor negative") {
    Tensor t1({1,2,3,-1,-2,0},{2,3});
    Tensor expected1({-1,-2,-3,1,2,0},{2,3});
    Tensor res1 = -t1;
    CHECK(check_tensor_equal(res1, expected1));
}

TEST_CASE("Tensor inplace add") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    t1 += Tensor({1,-1,2,-2,3,-3}, {2,3});
    CHECK(check_tensor_equal(t1, Tensor({2,1,5,2,8,3}, {2,3})));
}

TEST_CASE("Tensor inplace sub") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    t1 -= Tensor({1,-1,2,-2,3,-3}, {2,3});
    CHECK(check_tensor_equal(t1, Tensor({0,3,1,6,2,9}, {2,3})));
}

TEST_CASE("Tensor inplace mul") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    t1 *= Tensor({1,-1,2,-2,3,-3}, {2,3});
    CHECK(check_tensor_equal(t1, Tensor({1,-2,6,-8,15,-18}, {2,3})));
}

TEST_CASE("Tensor inplace div") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    t1 /= Tensor({1,-1,2,-2,3,-3}, {2,3});
    CHECK(check_tensor_equal(t1, Tensor({1,-2,3.0/2,-2,5.0/3,-2}, {2,3})));
}

TEST_CASE("Tensor power") {
    Tensor t1({0,1,2,3,4,5}, {2,3});
    Tensor t2 = power(t1, 2);
    CHECK(check_tensor_equal(t2, Tensor({0,1,4,9,16,25}, {2,3})));
}

TEST_CASE("Tensor exp") {
    Tensor t1({0, 1, -1}, {3});
    Tensor t2 = exp(t1);
    CHECK(check_tensor_equal(t2, Tensor({1, std::numbers::e_v<float>, 1.0/std::numbers::e_v<float>}, {3})));
}

TEST_CASE("Tensor log") {
    Tensor t1({1, std::numbers::e_v<float>, 1.f/std::numbers::e_v<float>}, {3});
    Tensor t2 = log_e(t1);
    CHECK(check_tensor_equal(t2, Tensor({0, 1.0, -1.0}, {3})));
    // CHECK_THROWS(log(Tensor({0, -1},{2}))); NOTE: Should this throw?
}
