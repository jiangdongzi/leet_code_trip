# 背景
我要使用c++练习leetcde

# 你的任务
为了方便我后面的调试, 你帮我创建一个完整的环境, 写好makefile, 编译要求以`Og`(完全不优化)编译, 并加`-g`, 方便我后面gdb调试, 另外你再帮我安装`fmtlib`, 我要用`fmt`输出调试, 比`std::cout`好用的多, 你创建个`hello world`吧, 我在这个文件中写力扣, 这个初始文件要使用`fmt`输出普通字符串, 并且要用`fmt`输出`stl`容器, 比如`vector`, `unordered_map`等.另外你要用普通的编译链接模式, 而不是`fmt`的`header-only`编译链接模式, 因为 `header-only`编译太慢了.

# 你的权限
你在docker测试容器中, 你是root用户, 随便安装环境, 随便折腾