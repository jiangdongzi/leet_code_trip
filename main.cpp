#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "lc.hpp"
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

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
    int cnt[128]{};
    for (const char c : s)
        cnt[c]++;
    bool inStk[128]{};
    std::string stk;
    for (const char c : s) {
        cnt[c]--;
        if (inStk[c])
            continue;
        inStk[c] = true;
        while (!stk.empty() && stk.back() > c && cnt[stk.back()] > 0) {
            inStk[stk.back()] = false;
            stk.pop_back();
        }
        stk.push_back(c);
    }
    return stk;
}

string simplifyPath(string path) {
    std::vector<std::string> stk;
    std::string tmp;
    for (const char c : path) {
        if (c == '/') {
            if (!tmp.empty()) {
                if (tmp == ".") {
                    // do nothing
                } else if (tmp == "..") {
                    if (!stk.empty())
                        stk.pop_back();
                } else {
                    stk.emplace_back(std::move(tmp));
                }

                tmp.clear();
            }
        } else {
            tmp.push_back(c);
        }
    }
    if (!tmp.empty()) {
        if (tmp == ".") {
            // do nothing
        } else if (tmp == "..") {
            if (!stk.empty())
                stk.pop_back();
        } else {
            stk.emplace_back(std::move(tmp));
        }

        tmp.clear();
    }
    std::string ret{"/"};
    for (const std::string &e : stk) {
        ret.append(e);
        ret.push_back('/');
    }
    if (ret.size() > 1)
        ret.pop_back();
    return ret;
}

void reverseString(vector<char> &s) {
    int l = 0, r = s.size() - 1;
    while (l < r) {
        std::swap(s[l++], s[r--]);
    }
}

vector<int> partitionLabels(string s) {
    const int sz = s.size();
    int farthestIdx[128];
    for (int i = 0; i < sz; i++) {
        farthestIdx[s[i]] = i;
    }
    int farthest = -1;
    std::vector<int> ret;
    int start = 0;
    for (int i = 0; i < sz; i++) {
        farthest = std::max(farthest, farthestIdx[s[i]]);
        if (farthest == i) {
            ret.emplace_back(i - start + 1);
            start = i + 1;
        }
    }
    return ret;
}

int integerBreak(int n) {
    if (n < 4)
        return n - 1;
    int dp[n + 1];
    std::memset(dp, 0, sizeof(dp));
    dp[1] = 1;
    dp[2] = 2;
    dp[3] = 3;
    for (int i = 4; i <= n; i++) {
        for (int j = 1; j < i; j++) {
            dp[i] = std::max(dp[i], j * dp[i - j]);
        }
    }
    return dp[n];
}

string intToRoman(int num) {
    int nums[] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    std::string strs[] = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
    std::string res;
    for (int i = 0; i < 13; i++) {
        while (num >= nums[i]) {
            res.append(strs[i]);
            num -= nums[i];
        }
    }
    return res;
}

int findTargetSumWays(vector<int> &nums, int target) {
    const int sm = std::accumulate(nums.begin(), nums.end(), 0);
    if (sm < std::abs(target))
        return 0;
    if ((sm + target) % 2 != 0)
        return 0;
    const int partSum = (sm + target) / 2;
    std::sort(nums.rbegin(), nums.rend());
    std::vector<int> suffix(nums.size(), 0);
    suffix[nums.size() - 1] = nums.back();
    for (int i = nums.size() - 2; i >= 0; i--) {
        suffix[i] = suffix[i + 1] + nums[i];
    }
    int ret = 0;
    std::function<void(const int, const int)> dfs = [&](const int start, const int k) {
        if (k == 0)
            ret++;
        if (start == nums.size()) {
            return;
        }
        for (int i = start; i < nums.size(); i++) {
            if (nums[i] > k)
                continue;
            if (suffix[i] < k)
                break;
            dfs(i + 1, k - nums[i]);
        }
    };
    dfs(0, partSum);
    return ret;
}

bool checkDynasty(vector<int> &places) {
    std::sort(places.begin(), places.end());
    int idx = 0;
    while (idx < 5 && places[idx] == 0)
        idx++;
    if (idx >= 4)
        return true;
    const int minVal = places[idx++];
    while (idx < 5 && places[idx - 1] != places[idx])
        idx++;
    if (idx != 5)
        return false;
    return places.back() - minVal <= 4;
}

ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    ListNode *a = headA, *b = headB;
    while (a != b) {
        if (a == nullptr) {
            a = headB;
        } else {
            a = a->next;
        }
        if (b == nullptr) {
            b = headA;
        } else {
            b = b->next;
        }
    }
    return a;
}

vector<vector<int>> fourSum(vector<int> &nums, int target) {
    const int nz = nums.size();
    auto twoSum = [&](int l, const long k) -> std::vector<std::pair<int, int>> {
        std::vector<std::pair<int, int>> ret;
        int r = nz - 1;
        while (l < r) {
            if ((long)nums[l] + (long)nums[r] < k) {
                l++;
            } else if ((long)nums[l] + (long)nums[r] > k) {
                r--;
            } else {
                ret.emplace_back(nums[l], nums[r]);
                const int lVal = nums[l++];
                while (l < r && lVal == nums[l])
                    l++;
            }
        }
        return ret;
    };
    auto threeSum = [&](const int start, const long k) -> std::vector<std::array<int, 3>> {
        std::vector<std::array<int, 3>> ret;
        for (int i = start; i <= nz - 3; i++) {
            if ((long)nums[i] + nums[i + 1] + nums[i + 2] > k)
                break;

            long long max1 = (long long)nums[i] + nums[nz - 1] + nums[nz - 2];
            if (max1 < k)
                continue;
            if (i > start && nums[i] == nums[i - 1])
                continue;
            const auto tSmRet = twoSum(i + 1, k - nums[i]);
            if (tSmRet.empty())
                continue;
            for (const auto &ele : tSmRet) {
                ret.push_back({nums[i], ele.first, ele.second});
            }
        }
        return ret;
    };
    std::vector<std::vector<int>> ret;
    std::sort(nums.begin(), nums.end());
    for (int i = 0; i <= nz - 4; i++) {
        if (nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target)
            break;
        long long max1 = (long long)nums[i] + nums[nz - 1] + nums[nz - 2] + nums[nz - 3];
        if (max1 < target)
            continue;
        if (i > 0 && nums[i] == nums[i - 1])
            continue;
        const auto thrSmRet = threeSum(i + 1, target - nums[i]);
        if (thrSmRet.empty())
            continue;
        for (const auto &ele : thrSmRet) {
            ret.push_back({nums[i], ele[0], ele[1], ele[2]});
        }
    }
    return ret;
}

} // namespace A

int main() {
    fp("hello leetcode + fmt\n");
    std::vector<int> nums{0, 0, 6, 7, 9};
    auto ret = A::checkDynasty(nums);
    fp("{}\n", ret);
    return 0;
}
