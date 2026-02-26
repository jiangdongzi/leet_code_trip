#pragma once

#include <exception>
#include <type_traits>
#include <typeinfo>

namespace stl_immitate {

template <class T>
constexpr typename std::remove_reference<T>::type&& move(T&& x) noexcept {
    return static_cast<typename std::remove_reference<T>::type&&>(x);
}

template <class T>
constexpr T&& forward(typename std::remove_reference<T>::type& x) noexcept {
    return static_cast<T&&>(x);
}

template <class T>
constexpr T&& forward(typename std::remove_reference<T>::type&& x) noexcept {
    static_assert(!std::is_lvalue_reference<T>::value, "bad forward of rvalue as lvalue");
    return static_cast<T&&>(x);
}

template <class T>
void swap(T& a, T& b) {
    T tmp = stl_immitate::move(a);
    a = stl_immitate::move(b);
    b = stl_immitate::move(tmp);
}

struct bad_function_call : std::exception {
    const char* what() const noexcept override {
        return "bad_function_call";
    }
};

template <class>
class function;

template <class R, class... Args>
class function<R(Args...)> {
private:
    struct base {
        virtual ~base() {}
        virtual base* clone() const = 0;
        virtual R invoke(Args... args) = 0;
        virtual const std::type_info& target_type() const noexcept = 0;
    };

    template <class F, bool IsVoid>
    struct holder_impl;

    template <class F>
    struct holder_impl<F, false> final : base {
        F f;
        explicit holder_impl(const F& fn) : f(fn) {}
        explicit holder_impl(F&& fn) : f(stl_immitate::move(fn)) {}

        base* clone() const override {
            return new holder_impl(f);
        }

        R invoke(Args... args) override {
            return f(stl_immitate::forward<Args>(args)...);
        }

        const std::type_info& target_type() const noexcept override {
            return typeid(F);
        }
    };

    template <class F>
    struct holder_impl<F, true> final : base {
        F f;
        explicit holder_impl(const F& fn) : f(fn) {}
        explicit holder_impl(F&& fn) : f(stl_immitate::move(fn)) {}

        base* clone() const override {
            return new holder_impl(f);
        }

        R invoke(Args... args) override {
            f(stl_immitate::forward<Args>(args)...);
        }

        const std::type_info& target_type() const noexcept override {
            return typeid(F);
        }
    };

    base* ptr_ = nullptr;

    void reset() noexcept {
        delete ptr_;
        ptr_ = nullptr;
    }

public:
    function() noexcept = default;
    function(std::nullptr_t) noexcept {}

    function(const function& other) : ptr_(other.ptr_ ? other.ptr_->clone() : nullptr) {}

    function(function&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    template <class F,
              class D = typename std::decay<F>::type,
              class = typename std::enable_if<!std::is_same<D, function>::value>::type>
    function(F&& f) : ptr_(new holder_impl<D, std::is_void<R>::value>(stl_immitate::forward<F>(f))) {}

    ~function() {
        reset();
    }

    function& operator=(std::nullptr_t) noexcept {
        reset();
        return *this;
    }

    function& operator=(const function& other) {
        if (this == &other) {
            return *this;
        }
        function tmp(other);
        swap(tmp);
        return *this;
    }

    function& operator=(function&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        ptr_ = other.ptr_;
        other.ptr_ = nullptr;
        return *this;
    }

    template <class F,
              class D = typename std::decay<F>::type,
              class = typename std::enable_if<!std::is_same<D, function>::value>::type>
    function& operator=(F&& f) {
        function tmp(stl_immitate::forward<F>(f));
        swap(tmp);
        return *this;
    }

    void swap(function& other) noexcept {
        stl_immitate::swap(ptr_, other.ptr_);
    }

    explicit operator bool() const noexcept {
        return ptr_ != nullptr;
    }

    template <class Ret = R>
    typename std::enable_if<!std::is_void<Ret>::value, Ret>::type operator()(Args... args) const {
        if (!ptr_) {
            throw bad_function_call();
        }
        return ptr_->invoke(stl_immitate::forward<Args>(args)...);
    }

    template <class Ret = R>
    typename std::enable_if<std::is_void<Ret>::value, void>::type operator()(Args... args) const {
        if (!ptr_) {
            throw bad_function_call();
        }
        ptr_->invoke(stl_immitate::forward<Args>(args)...);
    }

    const std::type_info& target_type() const noexcept {
        return ptr_ ? ptr_->target_type() : typeid(void);
    }
};

} // namespace stl_immitate

