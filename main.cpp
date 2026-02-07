#include <string>
#include <unordered_map>
#include <vector>

#include "lc.hpp"
using namespace std;
namespace A {
int numSquares(int n) {
    int dp[n + 1];
    for (int i = 0; i <= n; i++)
        dp[i] = i;
    for (int i = 2; i <= n; i++) {
        for (int j = 1; j * j <= i; j++) {
            dp[i] = std::min(dp[i], dp[i - j * j] + 1);
        }
    }
    return dp[n];
}

string removeDuplicateLetters(string s) {
    int cnt[128] {};
    for (const char c : s) cnt[c]++;
    bool inStk[128] {}
}
} // namespace A

int main() {
    fp("hello leetcode + fmt\n");
    return 0;
}
