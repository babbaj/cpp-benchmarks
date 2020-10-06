#include <cstdint>
#include <type_traits>
#include <utility>
#include <array>
#include <cstring>
#include <algorithm>

#include <benchmark/benchmark.h>
#include <immintrin.h>


template<size_t N>
bool isEmptySimpleOr(const std::array<uint8_t, N> &array) {
    uint8_t acc = 0;
    for (int i = 0; i < array.size(); i++) {
        acc |= array[i];
    }
    return !acc;
}

template<typename T> requires (sizeof(T) >= sizeof(__m256i))
bool isEmptyOr(const T& input) {
    struct wrapper {
        const __m256i value;

        wrapper operator|(wrapper y) const {
            return wrapper{_mm256_or_si256(value, y.value)};
        }
    };

    constexpr auto arrSize = sizeof(T) / sizeof(__m256i);
    const auto* bytes = reinterpret_cast<const int8_t*>(&input);

    return [bytes]<std::size_t... I>(std::index_sequence<I...>) {
        const auto result = ([bytes] {
            __m256i n;
            memcpy(&n, &bytes[I * sizeof(__m256i)], sizeof(__m256i));
            return wrapper{n};
        }() | ...).value;

        return _mm256_testz_si256(result, result) == 0;
    }(std::make_index_sequence<arrSize>{});
}

template<typename T> requires (sizeof(T) >= sizeof(__m256i))
bool isEmptyLoopUnrolled(const T& input) {
    constexpr auto arrSize = sizeof(T) / sizeof(__m256i);
    const auto* bytes = reinterpret_cast<const int8_t*>(&input);

#pragma unroll
    for (int i = 0; i < arrSize; i++) {
        __m256i n;
        memcpy(&n, &bytes[i * sizeof(__m256i)], sizeof(__m256i));
        if (_mm256_testz_si256(n, n) == 0) return false;
    }
    return true;
}

template<typename T> requires (sizeof(T) >= sizeof(__m256i))
bool isEmptyLoop(const T& input) {
    constexpr auto arrSize = sizeof(T) / sizeof(__m256i);
    const auto* bytes = reinterpret_cast<const int8_t*>(&input);
#pragma clang loop unroll(disable)
    for (int i = 0; i < arrSize; i++) {
        __m256i n;
        memcpy(&n, &bytes[i * sizeof(__m256i)], sizeof(__m256i));
        if (_mm256_testz_si256(n, n) == 0) return false;
    }
    return true;
}

template<typename T> requires (sizeof(T) >= sizeof(int64_t))
bool isEmptyInt64Loop(const T& input) {
    constexpr auto arrSize = sizeof(T) / sizeof(int64_t);
    const auto* bytes = reinterpret_cast<const int8_t*>(&input);

#pragma clang loop unroll(disable)
    for (int i = 0; i < arrSize; i++) {
        int64_t n;
        memcpy(&n, &bytes[i * sizeof(int64_t)], sizeof(int64_t));
        if (n != 0) return false;
    }
    return true;
}

constexpr auto TEST_SIZE = 512;

constexpr auto testData() {
    std::array<uint8_t, TEST_SIZE> data{};
    for (int i = 128; i < TEST_SIZE; i++) {
        //data[i] = 0xFF;
    }
    return data;
}


static void BM_testOr(benchmark::State& state) {
    std::array<uint8_t, TEST_SIZE> data = testData();
    benchmark::DoNotOptimize(data);

    for (auto _ : state) {
        benchmark::DoNotOptimize(isEmptyOr(data));
    }
}

static void BM_testSimpleOr(benchmark::State& state) {
    std::array<uint8_t, TEST_SIZE> data = testData();
    benchmark::DoNotOptimize(data);

    for (auto _ : state) {
        benchmark::DoNotOptimize(isEmptySimpleOr(data));
    }
}

static void BM_testLoopUnrolled(benchmark::State& state) {
    std::array<uint8_t, TEST_SIZE> data = testData();
    benchmark::DoNotOptimize(data);

    for (auto _ : state) {
        benchmark::DoNotOptimize(isEmptyLoopUnrolled(data));
    }
}

static void BM_testInt64Loop(benchmark::State& state) {
    std::array<uint8_t, TEST_SIZE> data = testData();
    benchmark::DoNotOptimize(data);

    for (auto _ : state) {
        benchmark::DoNotOptimize(isEmptyInt64Loop(data));
    }
}

static void BM_testLoop(benchmark::State& state) {
    std::array<uint8_t, TEST_SIZE> data = testData();
    benchmark::DoNotOptimize(data);

    for (auto _ : state) {
        benchmark::DoNotOptimize(isEmptyLoop(data));
    }
}

static void BM_testCompareWithZero(benchmark::State& state) {
    std::array<uint8_t, TEST_SIZE> data = testData();
    benchmark::DoNotOptimize(data);

    const decltype(data) zero{};
    for (auto _ : state) {
        benchmark::DoNotOptimize(data == zero);
    }
}

static void BM_testMemcmp(benchmark::State& state) {
    std::array<uint8_t, TEST_SIZE> data = testData();
    benchmark::DoNotOptimize(data);

    const decltype(data) zero{};
    for (auto _ : state) {
        const bool result = memcmp(&data[0], &zero[0], sizeof(data));
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(BM_testOr);
BENCHMARK(BM_testSimpleOr);
BENCHMARK(BM_testLoop);
BENCHMARK(BM_testLoopUnrolled);
BENCHMARK(BM_testInt64Loop);
BENCHMARK(BM_testCompareWithZero);
BENCHMARK(BM_testMemcmp);

BENCHMARK_MAIN();