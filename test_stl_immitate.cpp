#include <iostream>
#include <string>

#include "stl_immitate.hpp"

static int failures = 0;

static void expectTrue(bool cond, const std::string& msg) {
    if (!cond) {
        ++failures;
        std::cerr << "FAIL: " << msg << "\n";
    }
}

template <class T>
static void expectEq(const T& got, const T& expected, const std::string& msg) {
    if (!(got == expected)) {
        ++failures;
        std::cerr << "FAIL: " << msg << " (got=" << got << ", expected=" << expected << ")\n";
    }
}

static int add3(int x) {
    return x + 3;
}

struct Counter {
    int* p;
    explicit Counter(int* out) : p(out) {}
    void operator()(int delta) {
        *p += delta;
    }
};

static void test_function() {
    stl_immitate::function<int(int)> f0;
    expectTrue(!f0, "function default empty");
    try {
        (void)f0(1);
        expectTrue(false, "empty function should throw");
    } catch (const stl_immitate::bad_function_call&) {
        expectTrue(true, "empty function throws bad_function_call");
    } catch (...) {
        expectTrue(false, "empty function throws wrong exception");
    }

    stl_immitate::function<int(int)> f1 = add3;
    expectTrue((bool)f1, "function from function pointer");
    expectEq(f1(7), 10, "function pointer invoke");

    int base = 5;
    stl_immitate::function<int(int)> f2 = [base](int x) { return base + x; };
    expectEq(f2(3), 8, "capturing lambda invoke");

    stl_immitate::function<int(int)> f3 = f2;
    expectEq(f3(10), 15, "copy ctor");

    stl_immitate::function<int(int)> f4 = stl_immitate::move(f3);
    expectEq(f4(2), 7, "move ctor");

    f4 = nullptr;
    expectTrue(!f4, "assign nullptr resets");

    int sum = 0;
    stl_immitate::function<void(int)> fv = Counter(&sum);
    fv(3);
    fv(4);
    expectEq(sum, 7, "void-return function");
}

int main() {
    test_function();

    if (failures == 0) {
        std::cout << "OKK\n";
        return 0;
    }
    return 1;
}

