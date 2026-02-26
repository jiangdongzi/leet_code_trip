#include <type_traits>
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

template <class A, class B>
static void expectSame(const std::string& msg) {
    expectTrue(std::is_same<A, B>::value, msg);
}

struct ForwardDemo {
    static int pick(int&) { return 10; }
    static int pick(int&&) { return 20; }

    template <class T>
    static int pass(T&& t) {
        return pick(stl_immitate::forward<T>(t));
    }
};

// --- remove_reference / move / forward / reference-collapsing -----------------
static void test_remove_reference_and_move_forward() {
    expectSame<typename std::remove_reference<int&>::type, int>("remove_reference<int&> -> int");
    expectSame<typename std::remove_reference<const int&>::type, const int>("remove_reference<const int&> keeps const");

    int x = 1;
    const int cx = 2;

    expectTrue(std::is_same<decltype(stl_immitate::move(x)), int&&>::value, "move(lvalue int) -> int&&");
    expectTrue(std::is_same<decltype(stl_immitate::move(cx)), const int&&>::value, "move(const int) -> const int&&");

    auto lref = [](int&) {};
    auto rref = [](int&&) {};

    auto sink = [&](int& a) { lref(a); };
    auto sink2 = [&](int&& a) { rref(stl_immitate::move(a)); };
    (void)sink;
    (void)sink2;

    auto overload = [](int&) -> int { return 1; };
    auto overload2 = [](int&&) -> int { return 2; };

    auto call = [&](int& v) { return overload(v); };
    auto call2 = [&](int&& v) { return overload2(stl_immitate::move(v)); };
    expectTrue(call(x) == 1, "lvalue binds to int& overload");
    expectTrue(call2(3) == 2, "rvalue binds to int&& overload");

    expectTrue(ForwardDemo::pass(x) == 10, "forward preserves lvalue");
    expectTrue(ForwardDemo::pass(7) == 20, "forward preserves rvalue");
}

// --- decay -------------------------------------------------------------------
static int fn(int) {
    return 0;
}

static void test_decay() {
    // decay removes references + cv, and also does array/function-to-pointer
    expectSame<typename std::decay<const int&>::type, int>("decay<const int&> -> int");
    expectSame<typename std::decay<int&&>::type, int>("decay<int&&> -> int");

    int arr[3] = {1, 2, 3};
    (void)arr;
    expectTrue(std::is_same<typename std::decay<decltype(arr)>::type, int*>::value, "decay<int[3]> -> int*");

    expectTrue(std::is_same<typename std::decay<decltype(fn)>::type, int (*)(int)>::value,
               "decay<function type> -> function pointer");

    // Useful contrast: remove_reference doesn't do array/function decay and doesn't drop const.
    expectSame<typename std::remove_reference<const int&>::type, const int>("remove_reference<const int&> -> const int");
    expectTrue(!std::is_same<typename std::remove_reference<const int&>::type, int>::value,
               "remove_reference != decay for const/ref");
}

// --- enable_if / SFINAE ------------------------------------------------------
template <class T>
typename std::enable_if<std::is_integral<T>::value, const char*>::type category(T) {
    return "integral";
}

template <class T>
typename std::enable_if<!std::is_integral<T>::value, const char*>::type category(T) {
    return "non-integral";
}

template <class T, class = typename std::enable_if<std::is_integral<T>::value>::type>
int only_integral(T x) {
    return (int)x + 1;
}

static void test_enable_if() {
    expectTrue(std::string(category(1)) == "integral", "enable_if return-type overload (int)");
    expectTrue(std::string(category(3.14)) == "non-integral", "enable_if return-type overload (double)");
    expectTrue(only_integral(10) == 11, "enable_if extra template-parameter gate");

    // You can't call only_integral(3.14) here: it would be a compile error by design.
}

// --- is_same / "disable the templated ctor" pattern --------------------------
struct demo_box {
    demo_box() {}
    demo_box(const demo_box&) {}

    template <class U,
              class D = typename std::decay<U>::type,
              class = typename std::enable_if<!std::is_same<D, demo_box>::value>::type>
    explicit demo_box(U&&) {}
};

static void test_disable_ctor_pattern() {
    demo_box a;
    demo_box b(a); // should use copy-ctor, not the templated ctor
    (void)b;
    demo_box c(123); // should use templated ctor
    (void)c;
    expectTrue(true, "disable templated ctor pattern compiles");
}

// --- is_void ---------------------------------------------------------------
static void test_is_void() {
    expectTrue(std::is_void<void>::value, "is_void<void> == true");
    expectTrue(!std::is_void<int>::value, "is_void<int> == false");
}

int main() {
    test_remove_reference_and_move_forward();
    test_decay();
    test_enable_if();
    test_disable_ctor_pattern();
    test_is_void();

    if (failures == 0) {
        std::cout << "OKK\n";
        return 0;
    }
    return 1;
}
