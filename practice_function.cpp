#include <exception>
#include <iostream>
#include <string>
#include <type_traits>
#include <typeinfo>

// 练习目标：
// 1) 自己实现一遍 `stl_immitate::function`（模仿 std::function 的最小子集）
// 2) 在实现前，先理解并练习本文件里用到的模板工具：
//    - std::remove_reference / 转发引用 / 引用折叠
//    - std::forward / std::move 的核心思想
//    - std::decay（为什么要“衰减”）
//    - std::enable_if + std::is_same（禁用某些重载：SFINAE）
//    - std::is_void（void 返回值的处理）
//
// 用法：
// - 先只编译：`make practice_function`（默认不会触发检查）
// - 写完后运行检查：`make practice_function_check`

namespace stl_immitate_practice {

// 一个“依赖模板参数的 false”，用于让 static_assert 在模板实例化时才报错。
template <class>
struct dependent_false : std::false_type {};

// =========================
// 1) 实现 move / forward / swap
// =========================

// 实现 move：把任意 T&& 转成“去引用后的右值引用”
template <class T>
constexpr typename std::remove_reference<T>::type&& move(T&& /*x*/) noexcept {
    static_assert(dependent_false<T>::value, "TODO: implement stl_immitate_practice::move");
}

// 实现 forward（左值版本）
template <class T>
constexpr T&& forward(typename std::remove_reference<T>::type& /*x*/) noexcept {
    static_assert(dependent_false<T>::value, "TODO: implement stl_immitate_practice::forward(lvalue)");
}

// 实现 forward（右值版本，注意需要 static_assert 防止把右值当左值转发）
template <class T>
constexpr T&& forward(typename std::remove_reference<T>::type&& /*x*/) noexcept {
    static_assert(dependent_false<T>::value, "TODO: implement stl_immitate_practice::forward(rvalue)");
}

// 实现 swap：用 move 进行三次移动
template <class T>
void swap(T& /*a*/, T& /*b*/) {
    static_assert(dependent_false<T>::value, "TODO: implement stl_immitate_practice::swap");
}

// =========================
// 2) 实现 bad_function_call
// =========================
struct bad_function_call : std::exception {
    const char* what() const noexcept override {
        return "bad_function_call";
    }
};

// =========================
// 3) 实现 function（重点练习）
// =========================
//
// 提示：
// - 这是一个“类型擦除”(type-erasure)：
//   - 用一个基类 base 定义虚接口：clone()/invoke()/target_type()
//   - 用 holder<F> 保存真正的可调用对象 F
// - 为什么要用 std::decay：
//   - 把 F 统一成可按值保存的类型（去引用/cv + 数组/函数衰减）
// - 为什么要用 enable_if + is_same：
//   - 禁止 `function` 的模板构造函数在传入 function 本身时参与重载

template <class>
class function; // 先声明

template <class R, class... Args>
class function<R(Args...)> {
public:
    function() noexcept = default;

    // TODO: 实现下面这些成员（按自己的节奏一步步来）
    // - function(std::nullptr_t)
    // - 拷贝构造 / 移动构造
    // - 模板构造：template<class F, class D=decay<F>::type, class = enable_if<!is_same<D,function>::value>::type>
    // - 析构
    // - operator=（nullptr / copy / move / 模板赋值）
    // - swap
    // - operator bool
    // - operator()(Args...)（注意 void/non-void 两种返回）
    // - target_type()

private:
    // TODO: 写 base/holder，并保存一个 base* 指针
};

} // namespace stl_immitate_practice

#ifdef PRACTICE_RUN
// 你写完后，用 `make practice_function_check` 来跑这些检查。
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

static void check_function_basic() {
    using stl_immitate_practice::bad_function_call;
    using stl_immitate_practice::function;

    function<int(int)> f0;
    expectTrue(!f0, "default function is empty");
    try {
        (void)f0(1);
        expectTrue(false, "empty function should throw");
    } catch (const bad_function_call&) {
        expectTrue(true, "throws bad_function_call");
    } catch (...) {
        expectTrue(false, "throws wrong exception");
    }

    function<int(int)> f1 = add3;
    expectTrue((bool)f1, "function from function pointer");
    expectEq(f1(7), 10, "invoke function pointer");

    int base = 5;
    function<int(int)> f2 = [base](int x) { return base + x; };
    expectEq(f2(3), 8, "invoke capturing lambda");

    function<int(int)> f3 = f2;
    expectEq(f3(10), 15, "copy");

    function<int(int)> f4 = stl_immitate_practice::move(f3);
    expectEq(f4(2), 7, "move");

    f4 = nullptr;
    expectTrue(!f4, "assign nullptr resets");

    int sum = 0;
    function<void(int)> fv = Counter(&sum);
    fv(3);
    fv(4);
    expectEq(sum, 7, "void-return function");
}

int main() {
    check_function_basic();
    if (failures == 0) {
        std::cout << "OKK\n";
        return 0;
    }
    return 1;
}
#else
int main() {
    std::cout << "practice_function built.\n"
                 "Run checks with: make practice_function_check\n";
    return 0;
}
#endif

