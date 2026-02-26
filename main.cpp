#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fmt/core.h>
#include <functional>
#include <list>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cmath>
#include <iostream>
#include <limits>

#include "lc.hpp"
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
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

int ways(vector<string> &pizza, int k) {
    const int m = pizza.size(), n = pizza[0].size();
    std::vector<std::vector<int>> matrixSum(m + 1, std::vector<int>(n + 1));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrixSum[i + 1][j + 1] =
                matrixSum[i + 1][j] + matrixSum[i][j + 1] - matrixSum[i][j] + (pizza[i][j] & 1);
        }
    }
    auto getApple = [&](const int i, const int j, const int ii, const int jj) -> int {
        return matrixSum[ii][jj] - matrixSum[i][jj] - matrixSum[ii][j] + matrixSum[i][j];
    };
    int cuts = k - 1;
    std::vector<std::vector<std::vector<int>>> cache(
        k, std::vector<std::vector<int>>(m, std::vector<int>(n, -1)));
    std::function<int(const int, const int, const int)> dfs = [&](const int c, const int i,
                                                                  const int j) -> int {
        if (cache[c][i][j] != -1)
            return cache[c][i][j];
        if (c == 0)
            return getApple(i, j, m, n) > 0 ? 1 : 0;
        int ret = 0;
        for (int i2 = i + 1; i2 < m; i2++) {
            if (getApple(i2, j, m, n) <= 0)
                continue;
            ret += dfs(c - 1, i2, j);
        }
        for (int j2 = j + 1; j2 < n; j2++) {
            if (getApple(i, j2, m, n) <= 0)
                continue;
            ret += dfs(c - 1, i, j2);
        }
        return ret;
    };
    return dfs(k - 1, 0, 0);
}

int fib(int n) {
    if (n < 2)
        return n;
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int c = a + b;
        a = b;
        b = c;
    }
    return b;
}

string convertToTitle(int columnNumber) {
    std::string ret;
    while (columnNumber > 0) {
        ret.push_back((columnNumber - 1) % 26 + 'A');
        columnNumber = (columnNumber - 1) / 26;
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
}

ListNode *partition(ListNode *head, int x) {
    ListNode dummy1(0), dummy2(0);
    ListNode *ptr1 = &dummy1, *ptr2 = &dummy2;
    while (head != nullptr) {
        if (head->val < x) {
            ptr1->next = head;
            ptr1 = head;
        } else {
            ptr2->next = head;
            ptr2 = head;
        }
        head = head->next;
    }
    ptr1->next = dummy2.next;
    ptr2->next = nullptr;
    return dummy1.next;
}

string removeDuplicates(string s) {
    std::string ret;
    for (const char c : s) {
        if (ret.empty() || ret.back() != c) {
            ret.push_back(c);
        } else {
            ret.pop_back();
        }
    }
    return ret;
}
vector<string> topKFrequent(vector<string> &words, int k) {
    std::unordered_map<std::string, int> um;
    for (const auto &word : words) {
        um[word]++;
    }
    typedef std::pair<std::string, int> psi;
    auto cmp = [](const psi &a, const psi &b) -> bool {
        if (a.second > b.second)
            return true;
        if (a.second < b.second)
            return false;
        return a.first < b.first;
    };
    std::priority_queue<psi, std::vector<psi>, decltype(cmp)> pq(cmp);
    for (const auto &ele : um) {
        pq.emplace(ele);
        if (pq.size() > k)
            pq.pop();
    }
    assert(pq.size() == k);
    std::vector<std::string> ret;
    while (!pq.empty()) {
        ret.emplace_back(pq.top().first);
        pq.pop();
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
}

int countSubstrings(string s) {
    std::string processedStr{"^#"};
    for (const char c : s) {
        processedStr.push_back(c);
        processedStr.push_back('#');
    }
    processedStr.push_back('$');
    int radius[processedStr.size()];
    radius[0] = radius[1] = 0;
    int maxCenterI = 1;
    int ret = 0;
    for (int i = 2; i < processedStr.size() - 2; i++) {
        int r = 1;
        if (maxCenterI + radius[maxCenterI] > i) {
            r = std::min(radius[2 * maxCenterI - i], maxCenterI + radius[maxCenterI] - i);
        }
        while (processedStr[i + r] == processedStr[i - r])
            r++;
        radius[i] = --r;
        if (i + radius[i] > maxCenterI + radius[maxCenterI]) {
            maxCenterI = i;
        }
        ret += (r + 1) / 2;
    }
    return ret;
}

vector<int> nextGreaterElements(vector<int> &nums) {
    auto it = std::max_element(nums.begin(), nums.end());
    const int beginIdx = it - nums.begin() + 1;
    std::vector<int> stk;
    std::vector<int> ret(nums.size(), -1);
    for (int i = 0; i < nums.size(); i++) {
        const int realIdx = (beginIdx + i) % nums.size();
        while (!stk.empty() && nums[stk.back()] < nums[realIdx]) {
            ret[stk.back()] = nums[realIdx];
            stk.pop_back();
        }
        stk.emplace_back(realIdx);
    }
    return ret;
}

string longestDupSubstring(string s) {
    const int sz = s.size();
    const int MOD = 1e9 + 7;
    long pw[sz];
    pw[0] = 1;
    for (int i = 1; i < sz; i++) {
        pw[i] = pw[i - 1] * 128;
        pw[i] %= MOD;
    }
    auto hasDup = [&](const int k) -> int {
        assert(k > 0 && k < sz);
        long sm = 0;
        for (int i = 0; i < k - 1; i++) {
            sm = sm * 128 + s[i];
            sm %= MOD;
        }
        std::unordered_map<int, std::vector<int>> um;
        for (int i = 0; i <= sz - k; i++) {
            sm = sm * 128 + s[i + k - 1];
            sm %= MOD;
            auto it = um.find(sm);
            if (it == um.end()) {
                um[sm].emplace_back(i);
            } else {
                for (const int j : it->second) {
                    int idx = 0;
                    while (idx < k && s[i + idx] == s[j + idx])
                        idx++;
                    if (idx == k)
                        return i;
                }
                it->second.emplace_back(i);
            }
            sm = ((sm - s[i] * pw[k - 1]) % MOD + MOD) % MOD;
        }
        return -1;
    };
    int l = 1, r = sz;
    int start = -1, len = 0;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        const int st = hasDup(mid);
        if (st == -1) {
            r = mid;
        } else {
            l = mid + 1;
            start = st;
            len = mid;
        }
    }
    if (start != -1)
        return s.substr(start, len);
    return "";
}

vector<vector<int>> pathSum(TreeNode *root, int targetSum) {
    if (root == nullptr)
        return {};
    std::vector<int> tmp;
    std::vector<std::vector<int>> ret;
    std::function<void(TreeNode *, const int)> dfs = [&](TreeNode *root, const int k) {
        if (root == nullptr) {
            if (k == 0) {
                ret.emplace_back(tmp);
            }
            return;
        }
        tmp.emplace_back(root->val);
        if (root->left == nullptr) {
            dfs(root->right, k - root->val);
            tmp.pop_back();
            return;
        }
        if (root->right == nullptr) {
            dfs(root->left, k - root->val);
            tmp.pop_back();
            return;
        }
        dfs(root->left, k - root->val);
        dfs(root->right, k - root->val);
        tmp.pop_back();
    };
    dfs(root, targetSum);
    return ret;
}

void solveSudoku(vector<vector<char>> &board) {
    bool rFlag[9][10]{};
    bool cFlag[9][10]{};
    bool boxFlag[9][10]{};
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (board[i][j] == '.')
                continue;
            const int x = board[i][j] - '0';
            rFlag[i][x] = true;
            cFlag[j][x] = true;
            boxFlag[i / 3 * 3 + j / 3][x] = true;
        }
    }
    std::function<bool(const int)> dfs = [&](const int start) -> bool {
        if (start == 81)
            return true;
        const int i = start / 9, j = start % 9;
        if (board[i][j] != '.')
            return dfs(start + 1);
        for (int x = 1; x <= 9; x++) {
            if (rFlag[i][x] || cFlag[j][x] || boxFlag[i / 3 * 3 + j / 3][x])
                continue;
            rFlag[i][x] = true;
            cFlag[j][x] = true;
            boxFlag[i / 3 * 3 + j / 3][x] = true;
            board[i][j] = x + '0';
            if (dfs(start + 1))
                return true;
            rFlag[i][x] = false;
            cFlag[j][x] = false;
            boxFlag[i / 3 * 3 + j / 3][x] = false;
        }
        board[i][j] = '.';
        return false;
    };
    dfs(0);
}

bool checkInclusion(string s1, string s2) {
    if (s1.size() > s2.size())
        return false;
    int cnt[128]{};
    for (const char c : s1)
        cnt[c]++;
    const int z1 = s1.size();
    int k = z1;
    int l = 0, r = 0;
    while (r < s2.size()) {
        if (cnt[s2[r++]]-- > 0)
            k--;
        if (k > 0)
            continue;
        while (cnt[s2[l++]]++ < 0)
            ;
        if (r - l + 1 == z1)
            return true;
        k = 1;
    }
    return false;
}

int nextGreaterElement(int n) {
    std::string nStr = std::to_string(n);
    const int nz = nStr.size();
    int idx = nz - 2;
    while (idx >= 0 && nStr[idx] >= nStr[idx + 1])
        idx--;
    if (idx == -1)
        return -1;
    int anchor = idx++;
    const char val = nStr[anchor];
    while (idx < nz && nStr[idx] > val)
        idx++;
    std::swap(nStr[anchor], nStr[idx - 1]);
    std::reverse(nStr.begin() + anchor + 1, nStr.end());
    const long ret = std::stol(nStr);
    if (ret > INT32_MAX)
        return -1;
    return ret;
}

int kthSmallest(vector<vector<int>> &matrix, int k) {
    const int m = matrix.size(), n = matrix[0].size();
    typedef std::pair<int, int> pii;
    auto cmp = [&](const pii &a, const pii &b) {
        return matrix[a.first][a.second] > matrix[b.first][b.second];
    };
    std::priority_queue<pii, std::vector<pii>, decltype(cmp)> pq(cmp);
    for (int i = 0; i < n && i < k; i++) {
        pq.push({0, i});
    }
    while (k-- > 1) {
        const auto curP = pq.top();
        pq.pop();
        if (curP.first + 1 < m) {
            pq.push({curP.first + 1, curP.second});
        }
    }
    const auto curP = pq.top();
    return matrix[curP.first][curP.second];
}
int findRepeatDocument(vector<int> &documents) {
    for (int i = 0; i < documents.size(); i++) {
        while (documents[i] != documents[documents[i]]) {
            std::swap(documents[i], documents[documents[i]]);
        }
    }
    for (int i = 0; i < documents.size(); i++) {
        if (i != documents[i]) {
            return documents[i];
        }
    }
    throw "no repeat data";
}

class RandomizedSet {
    std::vector<int> datas;
    std::unordered_map<int, int> valIdx;

  public:
    RandomizedSet() {}

    bool insert(int val) {
        auto it = valIdx.find(val);
        if (it != valIdx.end())
            return false;
        valIdx.emplace(val, datas.size());
        datas.emplace_back(val);
        return true;
    }

    bool remove(int val) {
        auto it = valIdx.find(val);
        if (it == valIdx.end())
            return false;
        datas[it->second] = datas.back();
        valIdx[datas.back()] = it->second;
        datas.pop_back();
        valIdx.erase(it);
        return true;
    }

    int getRandom() {
        const int idx = std::rand() % datas.size();
        return datas[idx];
    }
};

int numDistinct(string s, string t) {
    const int sz = s.size(), tz = t.size();
    if (sz < tz)
        return 0;
    unsigned long dp[tz + 1];
    std::memset(dp, 0, sizeof(dp));
    dp[0] = 1;
    for (const char c : s) {
        for (int i = tz - 1; i >= 0; i--) {
            if (c == t[i]) {
                dp[i + 1] += dp[i];
            }
        }
    }
    return dp[tz];
}

string convert(string s, int numRows) {
    if (numRows == 1)
        return s;
    std::vector<std::string> vec(numRows);
    int idx = 0;
    bool up2down = true;
    for (const char c : s) {
        vec[idx].push_back(c);
        if (up2down) {
            if (idx == numRows - 1) {
                idx = numRows - 2;
                up2down = false;
            } else {
                idx++;
            }
        } else {
            if (idx == 0) {
                idx = 1;
                up2down = true;
            } else {
                idx--;
            }
        }
    }
    std::string ret;
    for (const auto &str : vec) {
        ret.append(str);
    }
    return ret;
}

vector<string> letterCombinations(string digits) {
    static vector<string> phone_strs = {" ",   "",    "abc",  "def", "ghi",
                                        "jkl", "mno", "pqrs", "tuv", "wxyz"};

    const int dz = digits.size();
    std::string tmp;
    std::vector<std::string> ret;
    std::function<void(const int)> dfs = [&](const int start) {
        if (start == dz) {
            ret.emplace_back(tmp);
            return;
        }
        const std::string &curLetters = phone_strs[digits[start] - '0'];
        for (const char c : curLetters) {
            tmp.push_back(c);
            dfs(start + 1);
            tmp.pop_back();
        }
    };
    dfs(0);
    return ret;
}

int hammingWeight(int n) {
    int cnt = 0;
    while (n != 0) {
        cnt++;
        n &= (n - 1);
    }
    return cnt;
}

int splitArray(vector<int> &nums, int k) {
    auto getLeastGroupLEX = [&](const int x) -> int {
        int cnt = 1;
        int tmp = 0;
        for (const int i : nums) {
            if (tmp + i > x) {
                cnt++;
                tmp = i;
            } else {
                tmp += i;
            }
        }
        return cnt;
    };
    int l = *std::max_element(nums.begin(), nums.end());
    int r = std::accumulate(nums.begin(), nums.end(), 1);
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (getLeastGroupLEX(mid) > k) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

int findMin(vector<int> &nums) {
    int l = 0, r = nums.size() - 1;
    while (l < r && nums[l] == nums[r])
        l++;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (nums[mid] > nums[r]) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return nums[l];
}
typedef std::pair<int, int> pii;
int rob(TreeNode *root) {
    int ret = root->val;
    std::function<pii(TreeNode *)> dfs = [&](TreeNode *root) -> pii {
        if (root == nullptr) {
            return {0, 0};
        }
        pii l = dfs(root->left);
        pii r = dfs(root->right);
        const int maxOfRobRoot = l.second + r.second + root->val;
        const int maxOfNotRobRoot = l.first + r.first;
        const int maxOfTotal = std::max(maxOfRobRoot, maxOfNotRobRoot);
        ret = std::max(maxOfTotal, ret);
        return {maxOfTotal, maxOfNotRobRoot};
    };
    dfs(root);
    return ret;
}

string crackPassword(vector<int> &password) {
    std::vector<std::string> pStrs;
    for (const int i : password) {
        pStrs.emplace_back(std::to_string(i));
    }
    std::sort(pStrs.begin(), pStrs.end(),
              [](const std::string &a, const std::string &b) -> bool { return a + b < b + a; });
    return std::accumulate(pStrs.begin(), pStrs.end(), std::string(""));
}

vector<int> singleNumber(vector<int> &nums) {
    int z = 0;
    for (const int i : nums)
        z ^= i;
    z &= (-z);
    int a = 0, b = 0;
    for (const int i : nums) {
        if (i & z) {
            a ^= i;
        } else {
            b ^= i;
        }
    }
    return {a, b};
}

vector<int> smallestK(vector<int> &arr, int k) {
    if (k == 0)
        return {};
    auto partition = [&](const int left, const int right) {
        assert(left < right);
        int p = left - 1;
        const int pivot = arr[right];
        for (int i = left; i <= right; i++) {
            if (arr[i] <= pivot) {
                std::swap(arr[++p], arr[i]);
            }
        }
        return p;
    };
    int l = 0, r = arr.size() - 1;
    k--;
    while (l < r) {
        int p = partition(l, r);
        if (p < k) {
            l = p + 1;
        } else if (p > k) {
            r = p - 1;
        } else {
            l = p;
        }
    }
    arr.resize(l + 1);
    return arr;
}

class MyCircularQueue {
    int *arr;
    int length;
    int head, tail;

  public:
    MyCircularQueue(int k) {
        arr = new int[k + 1];
        length = k + 1;
        head = 0;
        tail = 0;
    }

    bool enQueue(int value) {
        if (isFull())
            return false;
        arr[tail] = value;
        tail = (tail + 1) % length;
        return true;
    }

    bool deQueue() {
        if (isEmpty())
            return false;
        head = (head + 1) % length;
        return true;
    }

    int Front() {
        if (isEmpty())
            return -1;
        return arr[head];
    }

    int Rear() {
        if (isEmpty())
            return -1;
        int idx = (tail - 1 + length) % length;
        return arr[idx];
    }

    bool isEmpty() { return head == tail; }

    bool isFull() { return (head + length - 1) % length == tail; }
};

int lengthOfLongestSubstring(string s) {
    if (s.empty())
        return 0;
    int left = 0, right = 0;
    int cnt[128]{};
    int ret = 1;
    while (right < s.size()) {
        if (cnt[s[right++]]++ == 0)
            continue;
        ret = std::max(ret, right - left - 1);
        while (cnt[s[left++]]-- == 1)
            ;
    }
    ret = std::max(ret, (int)s.size() - left);
    return ret;
}

vector<string> removeInvalidParentheses(string s) {
    std::vector<std::string> ret;
    std::function<void(string, const int, const int)> rmRight = [&](string str, const int startI,
                                                                    const int startJ) {
        int stk = 0;
        for (int j = startJ; j >= 0; j--) {
            if (str[j] == ')') {
                stk++;
            } else if (str[j] == '(') {
                stk--;
            }
            if (stk < 0) {
                for (int i = startI; i >= j; i--) {
                    if (str[i] != '(' || (i < startI && str[i + 1] == '('))
                        continue;
                    rmRight(str.substr(0, i) + str.substr(i + 1), i - 1, j - 1);
                }
                return;
            }
        }
        ret.emplace_back(str);
    };

    std::function<void(string, const int, const int)> rmLeft = [&](string str, const int startI,
                                                                   const int startJ) {
        int stk = 0;
        for (int j = startJ; j < str.size(); j++) {
            if (str[j] == '(') {
                stk++;
            } else if (str[j] == ')') {
                stk--;
            }
            if (stk < 0) {
                for (int i = startI; i <= j; i++) {
                    if (str[i] != ')' || (i > startI && str[i - 1] == ')'))
                        continue;
                    rmLeft(str.substr(0, i) + str.substr(i + 1), i, j);
                }
                return;
            }
        }
        rmRight(str, str.size() - 1, str.size() - 1);
    };
    rmLeft(s, 0, 0);
    return ret;
}

int countNodes(TreeNode *root) {
    static auto getFullTreeH = [](TreeNode *const root) -> int {
        int cnt = 0;
        TreeNode *cur = root;
        while (cur != nullptr) {
            cnt++;
            cur = cur->left;
        }
        return cnt;
    };
    if (root == nullptr)
        return 0;
    int lh = getFullTreeH(root->left);
    int rh = getFullTreeH(root->right);
    if (lh == rh) {
        return std::pow(2, lh) + countNodes(root->right);
    } else {
        return countNodes(root->left) + std::pow(2, rh);
    }
}

vector<int> searchRange(vector<int> &nums, int target) {
    int l = 0, r = nums.size();
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (nums[mid] < target) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    if (l == nums.size() || nums[l] != target) {
        return {-1, -1};
    }
    r = nums.size();
    int first = l;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (nums[mid] == target) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return {first, r - 1};
}

class Solution {
    std::vector<int> pSm;
    const int sz;

  public:
    Solution(vector<int> &w) : sz(w.size()) {
        pSm.resize(sz);
        std::partial_sum(w.begin(), w.end(), pSm.begin());
    }

    int pickIndex() {
        const int target = std::rand() % pSm.back();
        int l = 0, r = pSm.size();
        while (l < r) {
            const int mid = l + (r - l) / 2;
            if (pSm[mid] <= target) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return l;
    }
};

TreeNode *sortedArrayToBST(vector<int> &nums) {
    std::function<TreeNode *(const int, const int)> dfs = [&](const int l,
                                                              const int r) -> TreeNode * {
        if (l < r)
            return nullptr;
        const int mid = l + (r - l) / 2;
        TreeNode *root = new TreeNode(nums[mid]);
        root->left = dfs(l, mid - 1);
        root->right = dfs(mid + 1, r);
        return root;
    };
    return dfs(0, nums.size() - 1);
}

int maxProfit2(int k, vector<int> &prices) {
    const int pz = prices.size();
    int dp[k + 1][pz];
    std::memset(dp, 0, sizeof(dp));
    for (int i = 1; i <= k; i++) {
        int maxHold = -prices[0];
        for (int j = 1; j < pz; j++) {
            dp[i][j] = std::max(dp[i][j - 1], maxHold + prices[j]);
            maxHold = std::max(maxHold, dp[i - 1][j - 1] - prices[j]);
        }
    }
    return dp[k][pz - 1];
}

constexpr static int dirs[4][2]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
void solve(vector<vector<char>> &board) {
    const int m = board.size(), n = board[0].size();
    auto isValid = [&](const int i, const int j) { return i >= 0 && i < m && j >= 0 && j < n; };
    std::function<void(const int, const int)> dfs = [&](const int i, const int j) {
        if (!isValid(i, j))
            return;
        if (board[i][j] != 'O')
            return;
        board[i][j] = '#';
        for (const auto &dir : dirs) {
            dfs(i + dir[0], j + dir[1]);
        }
    };
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i != 0 && i != m - 1 && j != 0 && j != n - 1)
                continue;
            dfs(i, j);
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j] == '#') {
                board[i][j] = 'O';
            } else if (board[i][j] == 'O') {
                board[i][j] = 'X';
            }
        }
    }
}

bool verifyTreeOrder(vector<int> &postorder) {
    std::vector<int> stk;
    int curMax = 1000000;
    for (int i = postorder.size() - 1; i >= 0; i--) {
        if (postorder[i] >= curMax)
            return false;
        while (!stk.empty() && postorder[i] < stk.back()) {
            curMax = stk.back();
            stk.pop_back();
        }
        stk.emplace_back(postorder[i]);
    }
    return true;
}

std::vector<int> ans;

void getChildreeDistK(TreeNode *root, const int k) {
    if (root == nullptr || k < 0)
        return;
    if (k == 0)
        ans.emplace_back(root->val);
    getChildreeDistK(root->left, k - 1);
    getChildreeDistK(root->right, k - 1);
}

int distDfs(TreeNode *root, TreeNode *target, const int k) {
    if (root == nullptr)
        return -1;
    if (root == target)
        return 0;
    const int l = distDfs(root->left, target, k);
    if (l != -1) {
        if (l + 1 == k) {
            ans.emplace_back(root->val);
        } else if (l + 1 < k) {
            getChildreeDistK(root->right, k - l - 2);
        }
        return l + 1;
    }
    const int r = distDfs(root->right, target, k);
    if (r != -1) {
        if (r + 1 == k) {
            ans.emplace_back(root->val);
        } else if (r + 1 < k) {
            getChildreeDistK(root->left, k - r - 2);
        }
        return r + 1;
    }
    return -1;
}

vector<int> distanceK(TreeNode *root, TreeNode *target, int k) {
    getChildreeDistK(target, k);
    distDfs(root, target, k);
    return ans;
}

int removeDuplicates(vector<int> &nums) {
    if (nums.size() < 3)
        return nums.size();
    int p = 1;
    for (int i = 2; i < nums.size(); i++) {
        if (nums[i] == nums[p - 1])
            continue;
        nums[++p] = nums[i];
    }
    return p + 1;
}

int findLengthOfLCIS(vector<int> &nums) {
    if (nums.size() < 2)
        return nums.size();
    nums.emplace_back(INT32_MIN);
    int ret = 1;
    int start = 0;
    for (int i = 1; i < nums.size(); i++) {
        if (nums[i] > nums[i - 1])
            continue;
        ret = std::max(ret, i - start);
        start = i;
    }
    return ret;
}

vector<int> twoSum(vector<int> &numbers, int target) {
    int l = 0, r = numbers.size() - 1;
    while (l < r) {
        const int sm = numbers[l] + numbers[r];
        if (sm < target) {
            l++;
        } else if (sm > target) {
            r--;
        } else {
            return {l + 1, r + 1};
        }
    }
    return {};
}

bool isSameTree(TreeNode *p, TreeNode *q) {
    if (p == nullptr)
        return q == nullptr;
    if (q == nullptr)
        return false;
    return p->val == q->val && isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

bool isAnagram(string s, string t) {
    if (s.size() != t.size())
        return false;
    int cnt[128]{};
    for (const char c : s)
        cnt[c]++;
    int k = s.size();
    int l = 0, r = 0;
    while (r < s.size()) {
        if (cnt[t[r++]]-- > 0)
            k--;
    }
    return k == 0;
}

string reverseWords(string s) {
    int idx = 0, sz = s.size();
    while (idx < sz) {
        while (idx < sz && std::isspace(s[idx]))
            idx++;
        if (idx == sz)
            break;
        int start = idx++;
        while (idx < sz && !std::isspace(s[idx]))
            idx++;
        std::reverse(s.begin() + start, s.begin() + idx);
    }
    return s;
}

vector<int> sortedSquares(vector<int> &nums) {
    int l = 0, r = nums.size() - 1;
    std::vector<int> ret(nums.size());
    int p = nums.size();
    while (l <= r) {
        const int lVal = nums[l];
        const int rVal = nums[r];
        if (lVal * lVal > rVal * rVal) {
            ret[--p] = lVal * lVal;
            l++;
        } else {
            ret[--p] = rVal * rVal;
            r--;
        }
    }
    return ret;
}

vector<vector<int>> combine(int n, int k) {
    std::vector<int> tmp;
    std::vector<std::vector<int>> ret;
    std::function<void(const int)> dfs = [&](const int start) {
        if (tmp.size() + n - start + 1 < k)
            return;
        if (tmp.size() == k) {
            ret.emplace_back(tmp);
            return;
        }
        for (int i = start; i <= n; i++) {
            tmp.emplace_back(i);
            dfs(i + 1);
            tmp.pop_back();
        }
    };
    dfs(1);
    return ret;
}

vector<string> binaryTreePaths(TreeNode *root) {
    std::string tmp;
    std::vector<std::string> ret;
    std::function<void(TreeNode *)> dfs = [&](TreeNode *root) {
        if (root == nullptr) {
            ret.emplace_back(tmp);
            ret.back().pop_back();
            ret.back().pop_back();
            return;
        }
        const std::string valStr = std::to_string(root->val);
        tmp.append(valStr);
        tmp.append("->");
        if (root->left == nullptr) {
            dfs(root->right);
            tmp.resize(tmp.size() - valStr.size() - 2);
            return;
        }
        if (root->right == nullptr) {
            dfs(root->left);
            tmp.resize(tmp.size() - valStr.size() - 2);
            return;
        }
        dfs(root->left);
        dfs(root->right);
        tmp.resize(tmp.size() - valStr.size() - 2);
    };
    dfs(root);
    return ret;
}

int missingNumber(vector<int> &nums) {
    return (1 + nums.size()) * nums.size() / 2 - std::accumulate(nums.begin(), nums.end(), 0);
}

vector<int> getMaxMatrix(vector<vector<int>> &matrix) {
    const int m = matrix.size();
    const int n = matrix[0].size();
    int maxSoFar = matrix[0][0];
    int sm[n];
    int c1 = 0, c2 = 0, d1 = 0, d2 = 0;
    for (int i = 0; i < m; i++) {
        std::memset(sm, 0, sizeof(sm));
        for (int j = i; j < m; j++) {
            int maxEndingHere = -1;
            int start = 0;
            for (int k = 0; k < n; k++) {
                sm[k] += matrix[j][k];
                if (maxEndingHere > 0) {
                    maxEndingHere += sm[k];
                } else {
                    maxEndingHere = sm[k];
                    start = k;
                }
                if (maxEndingHere > maxSoFar) {
                    maxSoFar = maxEndingHere;
                    c1 = i;
                    c2 = start;
                    d1 = j;
                    d2 = k;
                }
            }
        }
    }
    return {c1, c2, d1, d2};
}

int maxSumDivThree(vector<int> &nums) {
    const int nz = nums.size();
    int dp[nz + 1][3];
    dp[0][1] = dp[0][2] = -100000;
    dp[0][0] = 0;
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < 3; j++) {
            dp[i + 1][j] = std::max(dp[i][j], dp[i][(3 + (j - nums[i]) % 3) % 3] + nums[i]);
        }
    }
    return dp[nz][0];
}

int romanToInt(string s) {
    // unordered_map<char, int> symbolValues = {
    //     {'I', 1}, {'V', 5}, {'X', 10}, {'L', 50}, {'C', 100}, {'D', 500}, {'M', 1000},
    // };
    short symbolValues[128]{};
    symbolValues['I'] = 1;
    symbolValues['V'] = 5;
    symbolValues['X'] = 10;
    symbolValues['L'] = 50;
    symbolValues['C'] = 100;
    symbolValues['D'] = 500;
    symbolValues['M'] = 1000;
    int ret = 0;
    int mp[s.size()];
    for (int i = 0; i < s.size(); i++) {
        mp[i] = symbolValues[s[i]];
    }
    for (int i = 0; i < s.size() - 1; i++) {
        if (mp[i] < mp[i + 1]) {
            ret -= mp[i];
        } else {
            ret += mp[i];
        }
    }
    return ret + mp[s.size() - 1];
}

TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
    if (root == nullptr || root == p || root == q)
        return root;
    TreeNode *l = lowestCommonAncestor(root->left, p, q);
    TreeNode *r = lowestCommonAncestor(root->right, p, q);
    if (l == nullptr)
        return r;
    if (r == nullptr)
        return l;
    return root;
}

vector<int> findAnagrams(string s, string p) {
    const int sz = s.size(), pz = p.size();
    int cnt[128]{};
    for (const char c : p)
        cnt[c]++;
    int k = pz;
    int l = 0, r = 0;
    std::vector<int> ret;
    while (r < sz) {
        if (cnt[s[r++]]-- > 0)
            k--;
        if (k > 0)
            continue;
        while (cnt[s[l++]]++ < 0)
            ;
        k = 1;
        if (r + 1 - l == pz) {
            ret.emplace_back(l - 1);
        }
    }
    return ret;
}

int shortestSubarray(vector<int> &nums, int k) {
    std::vector<long> psm(nums.size() + 1);
    for (int i = 0; i < nums.size(); ++i) {
        psm[i + 1] = psm[i] + nums[i];
    }
    std::deque<long> idxQ{0};
    long ret = nums.size() + 1;
    for (int i = 1; i < psm.size(); i++) {
        while (!idxQ.empty() && psm[i] - psm[idxQ.front()] >= k) {
            ret = std::min(ret, i - idxQ.front());
            idxQ.pop_front();
        }
        while (!idxQ.empty() && psm[i] <= psm[idxQ.back()]) {
            idxQ.pop_back();
        }
        idxQ.emplace_back(i);
    }
    if (ret == nums.size() + 1)
        return -1;
    return ret;
}

vector<vector<int>> decorateRecord(TreeNode *root) {
    if (root == nullptr)
        return {};
    std::queue<TreeNode *> q;
    q.emplace(root);
    bool l2r = true;
    std::vector<std::vector<int>> ret;
    while (!q.empty()) {
        const int qz = q.size();
        std::vector<int> tmp(qz);
        for (int i = 0; i < qz; i++) {
            const TreeNode *cur = q.front();
            q.pop();
            const int idx = l2r ? i : qz - 1 - i;
            tmp[idx] = cur->val;
            if (cur->left) {
                q.emplace(cur->left);
            }
            if (cur->right) {
                q.emplace(cur->right);
            }
        }
        l2r = !l2r;
        ret.emplace_back(std::move(tmp));
    }
    return ret;
}

ListNode *middleNode(ListNode *head) {
    ListNode *fast = head, *slow = head;
    while (fast != nullptr && fast->next != nullptr) {
        fast = fast->next->next;
        slow = slow->next;
    }
    return slow;
}

int leastBricks(vector<vector<int>> &wall) {
    std::unordered_map<long, long> um;
    for (const auto &v : wall) {
        std::vector<long> psm(v.size());
        psm[0] = v[0];
        for (int i = 1; i < v.size(); ++i) {
            psm[i] = psm[i - 1] + v[i];
        }
        for (int i = 0; i < psm.size() - 1; i++) {
            um[psm[i]]++;
        }
    }
    long maxEqSm = 0;
    for (const auto &ele : um) {
        maxEqSm = std::max(maxEqSm, ele.second);
    }
    return wall.size() - maxEqSm;
}

vector<vector<string>> groupAnagrams(vector<string> &strs) {
    std::unordered_map<std::string, std::vector<std::string>> um;
    for (const auto &str : strs) {
        auto tmp = str;
        std::sort(tmp.begin(), tmp.end());
        um[tmp].emplace_back(str);
    }
    std::vector<std::vector<std::string>> ret;
    for (auto &ele : um) {
        ret.emplace_back(std::move(ele.second));
    }
    return ret;
}

TreeNode *sortedListToBST(ListNode *head) {
    if (head == nullptr)
        return nullptr;
    if (head->next == nullptr)
        return new TreeNode(head->val);
    ListNode *fast = head->next->next, *slow = head;
    while (fast != nullptr && fast->next != nullptr) {
        fast = fast->next->next;
        slow = slow->next;
    }
    TreeNode *root = new TreeNode(slow->next->val);
    root->right = sortedListToBST(slow->next->next);
    slow->next = nullptr;
    root->left = sortedListToBST(head);
    return root;
}

int countPrimes(int n) {
    if (n < 2)
        return 0;
    bool isCompositeNum[n];
    std::memset(isCompositeNum, 0, sizeof(isCompositeNum));
    int ret = 0;
    for (int i = 2; i < n; i++) {
        if (isCompositeNum[i])
            continue;
        ret++;
        for (int j = i + i; j < n; j += i) {
            isCompositeNum[j] = true;
        }
    }
    return ret;
}

TreeNode *mergeTrees(TreeNode *root1, TreeNode *root2) {
    if (root1 == nullptr)
        return root2;
    if (root2 == nullptr)
        return root1;
    root1->val += root2->val;
    root1->left = mergeTrees(root1->left, root2->left);
    root1->right = mergeTrees(root1->right, root2->right);
    return root1;
}

ListNode *trainningPlan(ListNode *head) {
    ListNode *ptr = nullptr;
    while (head != nullptr) {
        ListNode *tmp = head->next;
        head->next = ptr;
        ptr = head;
        head = tmp;
    }
    return ptr;
}

bool validateStackSequences(vector<int> &pushed, vector<int> &popped) {
    std::vector<int> stk;
    int idx = 0;
    for (const int i : pushed) {
        stk.push_back(i);
        while (!stk.empty() && popped[idx] == stk.back()) {
            stk.pop_back();
            idx++;
        }
    }
    return stk.empty();
}

int search(vector<int> &arr, int target) {
    int l = 0, r = arr.size() - 1;
    while (arr[l] == arr[r])
        r--;
    arr.resize(r + 1);
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (arr[mid] > arr[r]) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    const int minIdx = l;
    if (target > arr.back()) {
        int l = 0, r = minIdx;
        while (l < r) {
            const int mid = l + (r - l) / 2;
            if (arr[mid] < target) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        if (arr[l] == target)
            return l;
        return -1;
    } else {
        int l = minIdx, r = arr.size();
        while (l < r) {
            const int mid = l + (r - l) / 2;
            if (arr[mid] < target) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        if (l == arr.size())
            return -1;
        if (arr[l] == target)
            return l;
        return -1;
    }
}

void hanota(vector<int> &A, vector<int> &B, vector<int> &C) {
    std::function<void(const int n, vector<int> &, vector<int> &, vector<int> &)> dfs =
        [&](const int n, vector<int> &A, vector<int> &B, vector<int> &C) {
            if (n == 1) {
                C.push_back(A.back());
                A.pop_back();
                return;
            }
            dfs(n - 1, A, C, B);
            C.push_back(A.back());
            A.pop_back();
            dfs(n - 1, B, A, C);
        };
}

int findMaxLength(vector<int> &nums) {
    const int nz = nums.size();
    std::vector<int> mp(2 * nz + 1, -2);
    int *mpp = mp.data() + nz;
    mpp[0] = -1;
    int sm = 0;
    int ret = 0;
    for (int i = 0; i < nz; i++) {
        if (nums[i] == 1) {
            sm++;
        } else {
            sm--;
        }
        if (mpp[sm] == -2) {
            mpp[sm] = i;
        } else {
            ret = std::max(ret, i - mpp[sm]);
        }
    }
    return ret;
}

class MyHashMap {

    struct Node {
        int key{};
        int val{};
        Node *next{};
    };

    Node bucket[4024]{};

    uint32_t intHash(uint32_t x) {
        x ^= x >> 16;
        x *= 0x85ebca6b;
        x ^= x >> 13;
        x *= 0xc2b2ae35;
        x ^= x >> 16;
        return x;
    }

    Node &getB(const int key) {
        const int hs = intHash(key);
        std::cout << hs << std::endl;
        return bucket[hs % 4024];
    }

    Node *getLastNode(Node &curB) {
        Node *cur = &curB;
        while (cur->next != &curB) {
            cur = cur->next;
        }
        return cur;
    }

    Node *getPreNode(Node &curB, const int key) {
        Node *cur = &curB;
        while (cur->next != &curB) {
            if (cur->next->key == key)
                return cur;
            cur = cur->next;
        }
        return nullptr;
    }

  public:
    MyHashMap() {}

    void put(int key, int value) {
        auto &curB = getB(key);
        if (curB.next == nullptr) {
            curB.key = key;
            curB.val = value;
            curB.next = &curB;
            return;
        }
        if (curB.key == key) {
            curB.val = value;
            return;
        }
        const Node *head = &curB;
        Node *ptr = curB.next;
        while (ptr != head) {
            if (ptr->key == key) {
                ptr->val = value;
                return;
            }
            ptr = ptr->next;
        }
        Node *curNode = new Node;
        curNode->key = key;
        curNode->val = value;
        curNode->next = head->next;
        curB.next = curNode;
    }

    int get(int key) {
        auto &curB = getB(key);
        if (curB.next == nullptr)
            return -1;
        if (curB.key == key)
            return curB.val;
        Node *head = &curB;
        Node *ptr = head->next;
        while (ptr != head) {
            if (ptr->key == key)
                return ptr->val;
            ptr = ptr->next;
        }
        return -1;
    }

    void remove(int key) {
        auto &curB = getB(key);
        if (curB.next == nullptr)
            return;
        if (curB.key == key) {
            if (curB.next == &curB) {
                curB.next = nullptr;
                return;
            }
            Node *lastNode = getLastNode(curB);
            lastNode->next = &curB;
            Node *nextNode = curB.next;
            curB = *nextNode;
            delete nextNode;
        } else {
            Node *preNode = getPreNode(curB, key);
            if (preNode == nullptr)
                return;
            Node *curNode = preNode->next;
            preNode->next = curNode->next;
            delete curNode;
        }
    }
};

class MountainArray {
  public:
    int get(int index) { return 1; };
    int length() { return 2; }
};

int findInMountainArray(int target, MountainArray &mountainArr) {
    const int length = mountainArr.length();
    int l = 0, r = length - 1;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (mountainArr.get(mid) < mountainArr.get(mid + 1)) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    const int peakIdx = l;
    int ll = 0, rr = peakIdx;
    while (ll < rr) {
        const int mid = ll + (rr - ll) / 2;
        if (mountainArr.get(mid) < target) {
            ll = mid + 1;
        } else {
            rr = mid;
        }
    }
    if (mountainArr.get(ll) == target)
        return ll;
    l = peakIdx + 1, r = length;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (mountainArr.get(mid) > target) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    if (l == length || mountainArr.get(l) != target)
        return -1;
    return l;
}

class Node {
  public:
    int val;
    Node *next;
    Node *random;

    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};

Node *copyRandomList(Node *head) {
    Node *cur = head;
    while (cur != nullptr) {
        Node *tmp = cur->next;
        Node *newNode = new Node(cur->val);
        newNode->next = tmp;
        cur->next = newNode;
        cur = tmp;
    }
    cur = head;
    while (cur != nullptr) {
        Node *tmp = cur->next->next;
        if (cur->random != nullptr) {
            Node *newNode = cur->next;
            newNode->random = cur->random->next;
        }
        cur = tmp;
    }
    Node origin(0), *pO = &origin;
    Node newNode(0), *pN = &newNode;
    cur = head;
    while (cur != nullptr) {
        Node *tmp = cur->next->next;
        Node *newNode = cur->next;
        pO->next = cur;
        pO = cur;
        pN->next = newNode;
        pN = newNode;
        cur = tmp;
    }
    pO->next = nullptr;
    return newNode.next;
}
typedef std::pair<int, int> pii;
int orangesRotting(vector<vector<int>> &grid) {
    const int m = grid.size(), n = grid[0].size();
    constexpr static int dirs[4][2]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    auto isValid = [&](const int i, const int j) { return i >= 0 && i < m && j >= 0 && j < n; };
    std::queue<pii> q;
    int freshCnt = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == 2) {
                q.emplace(i, j);
            } else if (grid[i][j] == 1) {
                freshCnt++;
            }
        }
    }
    int step = 0;
    if (q.empty() && freshCnt == 0)
        return 0;
    while (!q.empty()) {
        const int qz = q.size();
        for (int i = 0; i < qz; i++) {
            const pii cur = q.front();
            q.pop();
            for (const auto &d : dirs) {
                const int nextI = cur.first + d[0];
                const int nextJ = cur.second + d[1];
                if (!isValid(nextI, nextJ))
                    continue;
                if (grid[nextI][nextJ] != 1)
                    continue;
                grid[nextI][nextJ] = 2;
                freshCnt--;
                ;
                q.emplace(nextI, nextJ);
            }
        }
        step++;
    }
    if (freshCnt > 0)
        return -1;
    return step - 1;
}

int maxCoins(vector<int> &nums) {
    nums.emplace(nums.begin(), 1);
    nums.emplace_back(1);
    const int nz = nums.size();
    int dp[nz][nz];
    std::memset(dp, 0, sizeof(dp));
    for (int i = 2; i < nz; i++) {
        for (int j = i - 2; j >= 0; j--) {
            for (int k = i - 1; k > j; k--) {
                dp[j][i] = std::max(nums[k] * nums[i] * nums[j] + dp[j][k] + dp[k][i], dp[j][i]);
            }
        }
    }
    return dp[0][nz - 1];
}

/*
边界正好与中间用一样的逻辑就是正确的
*/
int countDigitOne(int n) {
    int tens = 1;
    int ret = 0;
    while (n >= tens) {
        const int curDig = n / tens % 10;
        const int left = n / tens / 10;
        const int right = n % tens;
        if (curDig > 1) {
            ret += (left + 1) * tens; //[0, left] * tens;
        } else if (curDig == 0) {
            ret += left * tens; //[0, left) * tens;
        } else {
            ret += left * tens + right + 1; // [0, left) * tens + right + 1
        }
        tens *= 10;
    }
    return ret;
}

bool increasingTriplet(vector<int> &nums) {
    int a = INT32_MAX, b = a;
    for (int i : nums) {
        if (i > b)
            return true;
        if (i <= a) {
            a = i;
        } else {
            b = i;
        }
    }
    return false;
}

string thousandSeparator(int n) {
    std::string nStr = std::to_string(n);
    const int nz = nStr.size();
    std::string ret;
    for (int i = 0; i < nz; i++) {
        ret.push_back(nStr[nz - 1 - i]);
        if (i % 3 == 2) {
            ret.push_back('.');
        }
    }
    if (ret.back() == '.')
        ret.pop_back();
    std::reverse(ret.begin(), ret.end());
    return ret;
}

int trailingZeroes(int n) {
    int ret = 0;
    while (n > 0) {
        n /= 5;
        ret += n;
    }
    return ret;
}

string compressString(string s) {
    const int sz = s.size();
    int idx = 0;
    std::string ret;
    while (idx < sz) {
        const int start = idx++;
        const char c = s[start];
        while (idx < sz && s[idx] == c)
            idx++;
        const int cnt = idx - start;
        ret.push_back(c);
        ret.append(std::to_string(cnt));
    }
    if (ret.size() < s.size())
        return ret;
    return s;
}

int singleNonDuplicate(vector<int> &nums) {
    assert(nums.size() % 2 == 1);
    int l = 0, r = nums.size() - 1;
    while (l < r) {
        const int mid = ((l + (r - l) / 2) & (~1));
        if (nums[mid] == nums[mid + 1]) {
            l = mid + 2;
        } else {
            r = mid;
        }
    }
    return nums[l];
}

vector<vector<int>> levelOrderBottom(TreeNode *root) {
    if (root == nullptr)
        return {};
    std::queue<TreeNode *> q;
    q.emplace(root);
    std::vector<std::vector<int>> ret;
    std::vector<int> tmp;
    while (!q.empty()) {
        const int qz = q.size();
        for (int i = 0; i < qz; i++) {
            const TreeNode *cur = q.front();
            q.pop();
            tmp.emplace_back(cur->val);
            if (cur->left) {
                q.emplace(cur->left);
            }
            if (cur->right) {
                q.emplace(cur->right);
            }
        }
        ret.emplace_back(std::move(tmp));
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
}

int evalRPN(vector<string> &tokens) {
    std::vector<int> stk;
    for (const auto &s : tokens) {
        if (s.size() > 1 || std::isdigit(s.front())) {
            stk.emplace_back(std::stoi(s));
            continue;
        }
        const char sign = s.front();
        const int a = stk.back();
        stk.pop_back();
        const int b = stk.back();
        stk.pop_back();
        int newNum;
        switch (sign) {
        case '+': {
            newNum = a + b;
            break;
        }
        case '-': {
            newNum = b - a;
            break;
        }
        case '*': {
            newNum = a * b;
            break;
        }
        case '/': {
            newNum = b / a;
            break;
        }
        }
        stk.push_back(newNum);
    }
    return stk.back();
}

const static int tens[4]{1, 10, 100, 1000};
inline int lockAdd1(const short a, const int i) {
    const short dig = a / tens[i] % 10;
    if (dig == 9) {
        return a - 9 * tens[i];
    } else {
        return a + tens[i];
    }
}

inline int lockDec1(const short a, const int i) {
    const short dig = a / tens[i] % 10;
    if (dig == 0) {
        return a + 9 * tens[i];
    } else {
        return a - tens[i];
    }
}

int openLock(vector<string> &deadends, string target) {
    bool visited[10000]{};
    for (const auto &s : deadends) {
        visited[std::stoi(s)] = true;
    }
    if (visited[0])
        return -1;
    std::queue<short> q;
    q.emplace(0);
    int step = 0;
    const short tt = std::stoi(target);
    while (!q.empty()) {
        const int qz = q.size();
        for (int i = 0; i < qz; i++) {
            const short cur = q.front();
            q.pop();
            if (cur == tt)
                return step;
            for (int i = 0; i < 4; i++) {
                const short newNum = lockAdd1(cur, i);
                if (visited[newNum])
                    continue;
                visited[newNum] = true;
                q.emplace(newNum);
            }
            for (int i = 0; i < 4; i++) {
                const short newNum = lockDec1(cur, i);
                if (visited[newNum])
                    continue;
                visited[newNum] = true;
                q.emplace(newNum);
            }
        }
        step++;
    }
    return -1;
}

} // namespace A

namespace B {
#pragma GCC optimize("O3,inline,unroll-loops,fast-math,no-exceptions")
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using namespace std;

class Solution {
  public:
    // 预计算 10 的幂，避免重复计算
    static constexpr int tens[4] = {1, 10, 100, 1000};

    // 使用 int8_t 减小 visited 数组大小，提高 CPU 缓存命中率
    // 0: 未访问, 1: 从起点访问, 2: 从终点访问, 3: 死锁/已处理
    int8_t visited[10000];

    // 手写队列，避免 STL 动态分配开销
    // 10000 个状态足够覆盖所有可能性
    int q_fwd[10000];
    int q_bwd[10000];

    // 手写字符串转整数，比 std::stoi 快
    inline int str2int(const string &s) {
        return (s[0] - '0') * 1000 + (s[1] - '0') * 100 + (s[2] - '0') * 10 + (s[3] - '0');
    }

    int openLock(vector<string> &deadends, string target) {
        // 1. 初始化
        memset(visited, 0, sizeof(visited));

        for (const auto &d : deadends) {
            visited[str2int(d)] = 3; // 标记死锁
        }

        int start = 0;
        int end = str2int(target);

        if (visited[start] == 3)
            return -1;
        if (start == end)
            return 0;

        // 2. 设置双向 BFS 的起点和终点
        int head_fwd = 0, tail_fwd = 0;
        int head_bwd = 0, tail_bwd = 0;

        q_fwd[tail_fwd++] = start;
        visited[start] = 1; // 1 代表正向

        q_bwd[tail_bwd++] = end;
        visited[end] = 2; // 2 代表反向

        int step = 0;

        // 3. 开始双向搜索
        while (head_fwd < tail_fwd && head_bwd < tail_bwd) {
            step++;

            // 总是扩展队列较小的一端，保持搜索平衡
            bool processing_fwd = (tail_fwd - head_fwd) <= (tail_bwd - head_bwd);

            int *q_curr = processing_fwd ? q_fwd : q_bwd;
            int &head = processing_fwd ? head_fwd : head_bwd;
            int &tail = processing_fwd ? tail_fwd : tail_bwd;
            int my_mark = processing_fwd ? 1 : 2;
            int target_mark = processing_fwd ? 2 : 1;

            int size = tail - head;
            while (size-- > 0) {
                int cur = q_curr[head++];

                // 尝试拨动 4 个转盘
                for (int i = 0; i < 4; ++i) {
                    int ten = tens[i];
                    int digit = (cur / ten) % 10;

                    // 两个方向：+1 和 -1
                    // 预先计算变化后的数字，避免多次除法
                    int next_nums[2];

                    // 逻辑：(digit + 1) % 10
                    if (digit == 9)
                        next_nums[0] = cur - 9 * ten;
                    else
                        next_nums[0] = cur + ten;

                    // 逻辑：(digit - 1 + 10) % 10
                    if (digit == 0)
                        next_nums[1] = cur + 9 * ten;
                    else
                        next_nums[1] = cur - ten;

                    for (int next_val : next_nums) {
                        int status = visited[next_val];
                        if (status == 0) {
                            // 未访问过，加入当前队列
                            visited[next_val] = my_mark;
                            q_curr[tail++] = next_val;
                        } else if (status == target_mark) {
                            // 遇到了对面搜索过来的节点，说明路径连通！
                            return step;
                        }
                        // 如果 status == my_mark (已访问) 或 3 (死锁)，则忽略
                    }
                }
            }
        }

        return -1;
    }
};

int findCircleNum(vector<vector<int>> &isConnected) {
    struct UnionF {
        std::vector<int> pr;
        int cnt;
        UnionF(const int n) {
            pr.resize(n);
            cnt = n;
            for (int i = 0; i < n; i++) {
                pr[i] = i;
            }
        }
        int findP(const int a) {
            if (a == pr[a]) {
                return a;
            }
            return pr[a] = findP(pr[a]);
        }
        void merge(const int a, const int b) {
            const int pa = findP(a);
            const int pb = findP(b);
            if (pa != pb) {
                cnt--;
                pr[pa] = pb;
            }
        }
        int getDisjointSet() { return cnt; }
    };

    const int m = isConnected.size(), n = isConnected[0].size();
    UnionF uf(m);
    for (int i = 1; i < m; i++) {
        for (int j = 0; j < i; j++) {
            if (isConnected[i][j]) {
                uf.merge(i, j);
            }
        }
    }
    return uf.getDisjointSet();
}

int maxEnvelopes(vector<vector<int>> &envelopes) {
    typedef std::vector<int> vi;
    typedef std::vector<std::vector<int>> vvi;
    std::sort(envelopes.begin(), envelopes.end(), [](const vi &a, const vi &b) -> bool {
        if (a[0] == b[0]) {
            return a[1] > b[1];
        }
        return a[0] < b[0];
    });
    std::vector<int> lisV;
    for (const auto &v : envelopes) {
        const int i = v[1];
        int l = 0, r = lisV.size();
        while (l < r) {
            const int mid = l + (r - l) / 2;
            if (lisV[mid] < i) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        if (l == lisV.size()) {
            lisV.emplace_back(i);
        } else {
            lisV[l] = i;
        }
    }
    return lisV.size();
}

bool validPalindrome(string s) {
    int l = 0, r = s.size() - 1;
    while (l < r && s[l] == s[r]) {
        l++;
        r--;
    }
    if (l >= r)
        return true;
    int ll = l, rr = r;
    l++;
    while (l < r && s[l] == s[r]) {
        l++;
        r--;
    }
    if (l >= r)
        return true;
    l = ll;
    r = rr - 1;
    while (l < r && s[l] == s[r]) {
        l++;
        r--;
    }
    return l >= r;
}

int strStr(string haystack, string needle) {
    const int MOD = 1313131;
    const int base = 128;
    const int hz = haystack.size();
    const int nz = needle.size();
    std::function<int(const int)> myP = [&](const int n) -> int {
        if (n == 0)
            return 1;
        const long a = myP(n / 2);
        if (n & 1) {
            return (a * a * base) % MOD;
        } else {
            return (a * a) % MOD;
        }
    };
    const int pnz_1 = myP(nz - 1);
    int nsm = 0;
    for (const char c : needle) {
        nsm = nsm * base + c;
        nsm %= MOD;
    }
    int hsm = 0;
    for (int i = 0; i < nz - 1; i++) {
        hsm = hsm * base + haystack[i];
        hsm %= MOD;
    }
    for (int i = 0; i <= hz - nz; i++) {
        hsm = hsm * base + haystack[i + nz - 1];
        hsm %= MOD;
        if (hsm == nsm) {
            int idx = 0;
            while (idx < nz && haystack[i + idx] == needle[idx]) {
                idx++;
            }
            if (idx == nz)
                return i;
        }
        hsm = ((hsm - haystack[i] * pnz_1) % MOD + MOD) % MOD;
    }
    return -1;
}

bool validateBookSequences(vector<int> &putIn, vector<int> &takeOut) {
    std::vector<int> stk;
    int idx = 0;
    for (const int i : putIn) {
        stk.push_back(i);
        while (!stk.empty() && takeOut[idx] == stk.back()) {
            idx++;
            stk.pop_back();
        }
    }
    return stk.empty();
}

vector<int> lexicalOrder(int n) {
    std::vector<int> ret;
    std::function<void(int)> dfs = [&](int cur) {
        ret.emplace_back(cur);
        cur *= 10;
        for (int i = 0; i < 10; i++) {
            if (cur + i > n)
                break;
            dfs(cur + i);
        }
    };
    for (int i = 1; i < 10; i++) {
        if (i > n)
            break;
        dfs(i);
    }
    return ret;
}

TreeNode *deduceTree(vector<int> &preorder, vector<int> &inorder) {
    std::unordered_map<int, int> inValIdx;
    const int z = preorder.size();
    if (z == 0)
        return nullptr;
    for (int i = 0; i < z; i++) {
        inValIdx[inorder[i]] = i;
    }
    std::function<TreeNode *(const int, const int, const int)> dfs =
        [&](const int pIdx, const int iIdx, const int length) -> TreeNode * {
        if (length <= 0)
            return nullptr;
        assert(pIdx < z);
        TreeNode *root = new TreeNode(preorder[pIdx]);
        const int inorderAnchorIdx = inValIdx[preorder[pIdx]];
        const int lPreOrderIdx = pIdx + 1;
        const int lInorderIdx = iIdx;
        const int lLength = inorderAnchorIdx - iIdx;
        assert(lLength >= 0);
        root->left = dfs(lPreOrderIdx, lInorderIdx, lLength);
        const int rPreorderIdx = pIdx + lLength + 1;
        const int rInorderIdx = inorderAnchorIdx + 1;
        const int rLength = length - lLength - 1;
        assert(rLength >= 0);
        root->right = dfs(rPreorderIdx, rInorderIdx, rLength);
        return root;
    };
    return dfs(0, 0, z);
}

int atMostNGivenDigitSet(vector<string> &digits, int n) {
    const int dz = digits.size();
    const std::string nStr = std::to_string(n);
    const int nz = nStr.size();
    std::vector<int> pw(nz);
    pw[0] = 1;
    for (int i = 1; i < nz; i++) {
        pw[i] = dz * pw[i - 1];
    }
    int ret = 0;
    for (int i = 1; i < nz; i++) {
        ret += pw[i];
    }
    for (int i = 0; i < nz; i++) {
        int idx = 0;
        const char c = nStr[i];
        while (idx < dz && digits[idx].front() < c)
            idx++;
        ret += idx * pw[nz - 1 - i];
        if (idx == dz || digits[idx].front() > c)
            return ret;
    }
    return ret + 1;
}

bool predictTheWinner(vector<int> &nums) {
    const int nz = nums.size();
    int dp[nz][nz];
    std::memset(dp, 0, sizeof(dp));
    for (int i = 0; i < nz; i++)
        dp[i][i] = nums[i];
    for (int i = 1; i < nz; i++) {
        for (int j = i - 1; j >= 0; j--) {
            dp[j][i] = std::max(nums[j] - dp[j + 1][i], nums[i] - dp[j][i - 1]);
        }
    }
    return dp[0][nz - 1] >= 0;
}

vector<int> asteroidCollision(vector<int> &asteroids) {
    std::vector<int> stk;
    std::vector<int> ret;
    for (const int i : asteroids) {
        if (i > 0) {
            stk.push_back(i);
            continue;
        }
        while (!stk.empty() && stk.back() < -i) {
            stk.pop_back();
        }
        if (stk.empty()) {
            ret.emplace_back(i);
        } else if (stk.back() == -i) {
            stk.pop_back();
        }
    }
    ret.insert(ret.end(), stk.begin(), stk.end());
    return ret;
}

string getPermutation(int n, int k) {
    int permu[9];
    permu[0] = 1;
    for (int i = 1; i <= 8; i++) {
        permu[i] = i * permu[i - 1];
    }
    std::string s{"123456789"};
    s.resize(n);
    k--;
    std::string ret;
    while (k > 0) {
        const int pCnt = permu[s.size() - 1];
        const int idx = k / pCnt;
        ret.push_back(s[idx]);
        s.erase(s.begin() + idx);
        k %= pCnt;
    }
    ret.append(s);
    return ret;
}

int maximumProduct(vector<int> &nums) {
    std::sort(nums.begin(), nums.end());
    const int nz = nums.size();
    return std::max(nums[0] * nums[1] * nums.back(), nums[nz - 1] * nums[nz - 2] * nums[nz - 3]);
}

vector<int> productExceptSelf(vector<int> &nums) {
    const int nz = nums.size();
    std::vector<int> ret(nz);
    std::partial_sum(nums.begin(), nums.end(), ret.begin(), std::multiplies<int>());
    ret.back() = ret[nz - 2];
    for (int i = nz - 2; i >= 1; i--) {
        nums[i] *= nums[i + 1];
        ret[i] = ret[i - 1] * nums[i + 1];
    }
    ret[0] = nums[1];
    return ret;
}

bool isAdditiveNumber(string num) {
    const int nz = num.size();
    auto isValid = [&](const int start, const int end) -> bool {
        // start是第二个数的开头, 也是第一个数的"右开", end是第二个数的开
        if (num[start] == '0' && (end - start > 1))
            return false;
        if (std::max(start, end - start) > nz - end)
            return false;
        long a = std::stol(num.substr(0, start));
        long b = std::stol(num.substr(start, end - start));
        int idx = end;
        while (idx < nz) {
            long c = a + b;
            std::string cStr = std::to_string(c);
            const int cz = cStr.size();
            if (cz > nz - idx)
                return false;
            int i = 0;
            while (i < cz && cStr[i] == num[idx + i])
                i++;
            if (i != cz)
                return false;
            idx += cz;
            a = b;
            b = c;
        }
        return true;
    };
    for (int i = 1; i <= nz - 2; i++) {
        if (i > 1 && num[0] == '0')
            break;
        for (int j = i + 1; j < nz; j++) {
            if (isValid(i, j)) {
                return true;
            }
        }
    }
    return false;
}

string fractionToDecimal(int numerator, int denominator) {
    long a = denominator, b = numerator;
    std::string ret;
    if (a * b < 0)
        ret.push_back('-');
    a = std::abs(a);
    b = std::abs(b);
    long integerPart = b / a;
    ret.append(std::to_string(integerPart));
    int remainder = b % a;
    if (remainder == 0)
        return ret;
    ret.push_back('.');
    std::unordered_map<int, int> rIdx;
    while (remainder > 0 && rIdx.count(remainder) == 0) {
        rIdx.emplace(remainder, ret.size());
        remainder *= 10;
        ret.push_back(remainder / a + '0');
        remainder %= a;
    }
    if (remainder == 0)
        return ret;
    const int idx = rIdx[remainder];
    ret.insert(ret.begin() + idx, '(');
    ret.push_back(')');
    return ret;
}

vector<int> findClosestElements(vector<int> &arr, int k, int x) {
    const int z = arr.size();
    int l = 0, r = z - k;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (x - arr[mid] > arr[mid + k] - x) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    std::vector<int> ret(arr.begin() + l, arr.begin() + l + k);
    return ret;
}

int eraseOverlapIntervals(vector<vector<int>> &intervals) {
    typedef const std::vector<int> cvi_t;
    std::sort(intervals.begin(), intervals.end(),
              [](cvi_t &a, cvi_t &b) -> bool { return a[1] < b[1]; });
    int ret = 1;
    int rightMost = intervals.front().back();
    for (cvi_t &ele : intervals) {
        if (ele.front() < rightMost)
            continue;
        rightMost = ele.back();
        ret++;
    }
    return intervals.size() - ret;
}

bool checkSubarraySum(vector<int> &nums, int k) {
    const int nz = nums.size();
    std::vector<int> psm(nz + 1);
    std::partial_sum(nums.begin(), nums.end(), psm.begin() + 1);
    std::unordered_map<int, int> um;
    um.reserve(2 * nz);
    auto r = um.emplace(0, 0);
    auto pre_it = r.first;
    for (int i = 1; i < psm.size(); i++) {
        auto ret = um.emplace(psm[i] % k, 0);
        if (ret.first->second == 1)
            return true;
        pre_it->second = 1;
        pre_it = ret.first;
    }
    return false;
}

int subarraysDivByK(vector<int> &nums, int k) {
    int um[k + 1];
    std::memset(um, 0, sizeof(um));
    um[0] = 1;
    long sum = 0;
    int ret = 0;
    for (const int i : nums) {
        sum += i;
        ret += um[(k + sum % k) % k]++;
    }
    return ret;
}

bool isHappy(int n) {
    auto next = [](int n) -> int {
        int ret = 0;
        while (n > 0) {
            const int d = n % 10;
            ret += d * d;
            n /= 10;
        }
        return ret;
    };
    if (n == 1)
        return true;
    std::unordered_set<int> us{n};
    while (true) {
        n = next(n);
        if (n == 1)
            return true;
        if (!us.emplace(n).second)
            return false;
    }
}

vector<int> countSmaller(vector<int> &nums) {
    const int nz = nums.size();
    auto getHigh = [](const long i) -> long { return i >> 32; };
    auto getLow = [](const long i) -> long { return int(i); };
    auto buildL = [](const long h, const uint32_t l) -> long { return (h << 32) | l; };
    std::vector<long> compositeNums(nz);
    for (int i = 0; i < nz; i++) {
        compositeNums[i] = buildL(i, nums[i]);
    }
    std::vector<int> ret(nz, 0);
    auto merge = [&](const int l, const int mid, const int r) {
        int i = l, j = mid;
        long tmp[r - l];
        int p = -1;
        while (i < mid && j < r) {
            const long realI = getLow(compositeNums[i]);
            const long realJ = getLow(compositeNums[j]);
            if (realI > realJ) {
                const int idx = getHigh(compositeNums[i]);
                ret[idx] += r - j;
                tmp[++p] = compositeNums[i++];
            } else {
                tmp[++p] = compositeNums[j++];
            }
        }
        if (l < mid) {
            std::memcpy(tmp + p + 1, compositeNums.data() + i, (mid - i) * 8);
        } else {
            std::memcpy(tmp + p + 1, compositeNums.data() + j, (r - j) * 8);
        }
        std::memcpy(compositeNums.data() + l, tmp, sizeof(tmp));
    };
    std::function<void(const int, const int)> mergeSort = [&](const int l, const int r) {
        if (l >= r)
            return;
        const int mid = l + (r - l) / 2;
        mergeSort(l, mid);
        mergeSort(mid + 1, r);
        merge(l, mid + 1, r + 1);
    };
    mergeSort(0, nz - 1);
    return ret;
}

vector<int> dailyTemperatures(vector<int> &temperatures) {
    std::vector<int> stk;
    std::vector<int> ret(temperatures.size());
    for (int i = 0; i < temperatures.size(); i++) {
        while (!stk.empty() && temperatures[stk.back()] < temperatures[i]) {
            ret[stk.back()] = i - stk.back();
            stk.pop_back();
        }
        stk.emplace_back(i);
    }
    return ret;
}

string validIPAddress(string queryIP) {
    auto split = [&](const char spliter) -> std::vector<std::string> {
        std::vector<std::string> ret;
        std::string tmp;
        for (const char c : queryIP) {
            if (c == spliter) {
                if (tmp.empty())
                    return {};
                ret.emplace_back(std::move(tmp));
            } else {
                tmp.push_back(c);
            }
        }
        if (tmp.empty())
            return {};
        ret.emplace_back(tmp);
        return ret;
    };

    std::vector<std::string> strs;
    auto isValidIpV4 = [&]() -> bool {
        if (strs.size() != 4)
            return false;
        auto isIpV4Ele = [](const std::string &str) -> bool {
            if (str.size() > 3)
                return false;
            for (const char c : str) {
                if (!std::isdigit(c))
                    return false;
            }
            if (str.size() > 1 && str.front() == '0')
                return false;
            const int num = std::stoi(str);
            return num <= 255;
        };
        for (const std::string &str : strs) {
            if (!isIpV4Ele(str))
                return false;
        }
        return true;
    };

    auto isValidIpV6 = [&]() -> bool {
        if (strs.size() != 8)
            return false;
        auto isIpV6Ele = [](const std::string &str) -> bool {
            if (str.size() > 4)
                return false;
            for (const char c : str) {
                if (!isalnum(c))
                    return false;
                if (!std::isdigit(c)) {
                    const char cc = std::tolower(c);
                    if (cc > 'f')
                        return false;
                }
            }
            return true;
        };
        for (const std::string &str : strs) {
            if (!isIpV6Ele(str))
                return false;
        }
        return true;
    };
    if (queryIP.find('.') == std::string::npos) {
        strs = split(':');
        if (isValidIpV6())
            return "IPv6";
    } else {
        strs = split('.');
        if (isValidIpV4())
            return "IPv4";
    }
    return "Neither";
}

bool exist(vector<vector<char>> &board, string word) {
    constexpr static int dirs[4][2]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    const int m = board.size(), n = board[0].size();
    auto isValid = [&](const int i, const int j) { return i >= 0 && i < m && j >= 0 && j < n; };
    std::vector<std::vector<bool>> visited(m, std::vector<bool>(n));
    std::function<bool(const int, const int, const int)> dfs = [&](const int i, const int j,
                                                                   const int idx) -> bool {
        if (board[i][j] != word[idx])
            return false;
        if (idx + 1 == word.size())
            return true;
        for (const auto &dir : dirs) {
            const int ii = i + dir[0], jj = j + dir[1];
            if (!isValid(ii, jj))
                continue;
            if (visited[ii][jj])
                continue;
            visited[ii][jj] = true;
            if (dfs(i + dir[0], j + dir[1], idx + 1))
                return true;
            visited[ii][jj] = false;
        }
        return false;
    };
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            visited[i][j] = true;
            if (dfs(i, j, 0))
                return true;
            visited[i][j] = false;
        }
    }
    return false;
}

int calculate(string s) {
    int idx = 0;
    const int sz = s.size();
    std::function<int()> dfs = [&]() -> int {
        long ret = 0;
        int presign = 1;
        while (idx < sz) {
            const char c = s[idx++];
            if (std::isspace(c))
                continue;
            if (c == '+') {
                presign = 1;
            } else if (c == '-') {
                presign = -1;
            } else if (std::isdigit(c)) {
                int start = idx - 1;
                while (idx < sz && std::isdigit(s[idx]))
                    idx++;
                const int num = std::stoi(s.substr(start, idx - start));
                ret += presign * num;
            } else if (c == '(') {
                const int num = dfs();
                ret += presign * num;
            } else if (c == ')') {
                return ret;
            }
        }
        return ret;
    };
    return dfs();
}

class Codec {
  public:
    // Encodes a tree to a single string.
    string serialize(TreeNode *root) {
        std::string ret;
        std::function<void(TreeNode *)> dfs = [&](TreeNode *root) {
            if (root == nullptr) {
                ret.append("# ");
                return;
            }
            ret.append(std::to_string(root->val) + ' ');
            dfs(root->left);
            dfs(root->right);
        };
        return ret;
    }

    // Decodes your encoded data to tree.
    TreeNode *deserialize(string data) {
        std::string tmp;
        std::vector<std::string> strs;
        for (const char c : data) {
            if (c == ' ') {
                strs.emplace_back(std::move(tmp));
            } else {
                tmp.push_back(c);
            }
        }
        int idx = 0;
        std::function<TreeNode *()> dfs = [&]() -> TreeNode * {
            const std::string str = strs[idx++];
            if (str == "#") {
                return nullptr;
            }
            TreeNode *root = new TreeNode(std::stoi(str));
            root->left = dfs();
            root->right = dfs();
            return root;
        };
        return dfs();
    }

    vector<vector<int>> permuteUnique(vector<int> &nums) {
        std::sort(nums.begin(), nums.end());
        std::vector<int> tmp;
        std::vector<std::vector<int>> res;
        std::function<void()> dfs = [&]() {
            if (tmp.size() == nums.size()) {
                res.emplace_back(tmp);
                return;
            }
            for (int i = 0; i < nums.size(); i++) {
                if (i > 0 && nums[i] == nums[i - 1])
                    continue;
                if (nums[i] == -11)
                    continue;
                const int bak = nums[i];
                tmp.emplace_back(nums[i]);
                nums[i] = -11;
                dfs();
                tmp.pop_back();
                nums[i] = bak;
            }
        };
        dfs();
        return res;
    }

    int findMin(vector<int> &nums) {
        int l = 0, r = nums.size() - 1;
        while (l < r) {
            const int mid = l + (r - l) / 2;
            if (nums[mid] > nums[r]) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return nums[l];
    }
};

class LFUCache {
    struct Node {
        int key;
        int val;
        int freq;
    };
    const int capacity;
    int minFreq{};
    std::unordered_map<int, std::list<Node>::iterator> keyIt;
    std::vector<std::list<Node>> freqVec;

  public:
    LFUCache(int capacity) : capacity(capacity) {
        freqVec.emplace_back();
        freqVec.emplace_back();
    }

    int get(int key) {
        auto it = keyIt.find(key);
        if (it == keyIt.end())
            return -1;
        auto listIt = it->second;
        const int rawFreq = listIt->freq;
        const int rawVal = listIt->val;
        auto newNode = *listIt;
        newNode.freq++;
        freqVec[rawFreq].erase(listIt);
        keyIt.erase(it);
        if (freqVec.size() + 1 == rawFreq)
            freqVec.emplace_back();
        freqVec[rawFreq + 1].emplace_front(newNode);
        keyIt.emplace(key, freqVec[rawFreq + 1].begin());
        if (freqVec[rawFreq].empty() && minFreq == rawFreq) {
            minFreq++;
        }
        return rawVal;
    }

    void put(int key, int value) {
        if (get(key) == -1) {
            const Node newNode{key, value, 1};
            if (keyIt.size() == capacity) {
                auto &l = freqVec[minFreq];
                keyIt.erase(l.back().key);
                l.pop_back();
            }
            freqVec[1].emplace_front(newNode);
            keyIt.emplace(key, freqVec[1].begin());
            minFreq = 1;
        } else {
            auto it = keyIt.find(key);
            it->second->val = value;
        }
    }
};

bool canJump(vector<int> &nums) {
    int farthest = 0;
    for (int i = 0; i < nums.size() - 1; i++) {
        farthest = std::max(farthest, i + nums[i]);
        if (farthest == i)
            return false;
    }
    return true;
}

} // namespace B

namespace C {
// #pragma GCC optimize("O3,inline,unroll-loops,fast-math,no-exceptions")
#include <iostream>
#include <list>
#include <vector>

// 禁用 I/O 同步，极大提升 LeetCode 上的运行速度
// static const int _ = []() {
//     std::ios::sync_with_stdio(false);
//     std::cin.tie(nullptr);
//     std::cout.tie(nullptr);
//     return 0;
// }();

class LFUCache {
    struct Node {
        int key;
        int val;
        int freq;
    };

    int capacity;
    int minFreq = 0;
    int currentSize = 0; // 手动维护元素个数，避免额外函数调用

    // 直接利用题目的 key 范围 (0 ~ 100000) 进行数组映射，抛弃 unordered_map
    std::vector<std::list<Node>::iterator> keyIt;
    std::vector<bool> exists; // 辅助数组，用于判断 key 是否存在
    std::vector<std::list<Node>> freqVec;

    inline void ensureFreq(int f) {
        if ((int)freqVec.size() <= f)
            freqVec.resize(f + 1);
    }

    inline void touch(int key) {
        // 直接通过数组索引获取迭代器
        auto it = keyIt[key];
        int f = it->freq;

        ensureFreq(f + 1);

        // 利用原有的 splice 逻辑，依旧高效
        it->freq++;
        freqVec[f + 1].splice(freqVec[f + 1].begin(), freqVec[f], it);

        if (freqVec[f].empty() && minFreq == f) {
            minFreq++;
        }
    }

  public:
    LFUCache(int cap) : capacity(cap) {
        freqVec.resize(2);
        // 初始化足够大的数组直接覆盖所有的 key 可能性
        keyIt.resize(100001);
        exists.resize(100001, false);
    }

    int get(int key) {
        // O(1) 数组寻址判断
        if (!exists[key])
            return -1;

        touch(key);
        return keyIt[key]->val;
    }

    void put(int key, int value) {
        if (capacity == 0)
            return;

        if (exists[key]) {
            keyIt[key]->val = value;
            touch(key);
            return;
        }

        if (currentSize == capacity) {
            auto &l = freqVec[minFreq];
            int evictKey = l.back().key;

            // 淘汰节点
            exists[evictKey] = false;
            l.pop_back();
            currentSize--;
        }

        // 插入新节点
        ensureFreq(1);
        freqVec[1].push_front(Node{key, value, 1});
        keyIt[key] = freqVec[1].begin();
        exists[key] = true;
        minFreq = 1;
        currentSize++;
    }
};

vector<vector<int>> combinationSum2(vector<int> &candidates, int target) {
    std::sort(candidates.rbegin(), candidates.rend());
    std::vector<std::vector<char>> memo(candidates.size(), std::vector<char>(30, -1));
    std::vector<int> tmp;
    std::vector<std::vector<int>> res;
    std::function<int(const int, const int)> dfs = [&](const int start, const int k) -> int {
        if (k == 0) {
            res.emplace_back(tmp);
            return 1;
        }
        if (k < 0 || start == candidates.size())
            return 0;
        for (int i = start; i < candidates.size(); i++) {
            if (i > start && candidates[i] == candidates[i - 1])
                continue;
            if (memo[i][k] == 0)
                return 0;
            tmp.emplace_back(candidates[i]);
            if (dfs(i + 1, k - candidates[i]) == 1)
                memo[i][k] = 1;
            tmp.pop_back();
        }
        return memo[start][k];
    };
    dfs(0, target);
    return res;
}

int maxArea(vector<int> &height) {
    int l = 0, r = height.size() - 1;
    int ret = 0;
    while (l < r) {
        if (height[l] < height[r]) {
            ret = std::max(ret, height[l] * (r - l));
            l++;
        } else {
            ret = std::max(ret, height[r] * (r - l));
            r--;
        }
    }
    return ret;
}

string removeKdigits(string num, int k) {
    std::string stk;
    for (const char i : num) {
        while (!stk.empty() && stk.back() > i && k-- > 0) {
            stk.pop_back();
        }
        stk.push_back(i);
    }
    if (k > 0)
        stk.resize(stk.size() - k);
    size_t pos = stk.find_first_not_of('0');
    if (pos == std::string::npos)
        return "0";
    else
        return stk.substr(pos);
}

int singleNumber(vector<int> &nums) {
    int ret = 0;
    for (const int i : nums) {
        ret ^= i;
    }
    return ret;
}

int maxProfit2(vector<int> &prices) {
    int oneHold = -prices[0], oneSell = 0, twoHold = oneHold, twoSell = 0;
    for (const int i : prices) {
        twoSell = std::max(twoSell, twoHold + i);
        twoHold = std::max(twoHold, oneSell - i);
        oneSell = std::max(oneSell, oneHold + i);
        oneHold = std::max(oneHold, -i);
    }
    return twoSell;
}

int reversePairs(vector<int> &record) {
    int ret = 0;
    auto merge = [&](const int l, const int mid, const int r) {
        int tmp[r - l];
        int p = -1;
        int i = l, j = mid;
        while (i < mid && j < r) {
            if (record[i] <= record[j]) {
                tmp[++p] = record[i++];
            } else {
                ret += mid - i;
                tmp[++p] = record[j++];
            }
        }
        if (i == mid) {
            std::memcpy(tmp + p + 1, record.data() + j, 4 * (r - j));
        } else {
            std::memcpy(tmp + p + 1, record.data() + i, 4 * (mid - i));
        }
        std::memcpy(record.data() + l, tmp, sizeof(tmp));
    };
    std::function<void(const int, const int)> mergeSort = [&](const int l, const int r) {
        if (l >= r)
            return;
        const int mid = l + (r - l) / 2;
        mergeSort(l, mid);
        mergeSort(mid + 1, r);
        merge(l, mid + 1, r + 1);
    };
    mergeSort(0, record.size() - 1);
    // fp("record: {}\n", record);
    return ret;
}

ListNode *rotateRight(ListNode *head, int k) {
    if (head == nullptr || head->next == nullptr)
        return head;
    int cnt = 1;
    ListNode *ptr = head;
    while (ptr->next != nullptr) {
        cnt++;
        ptr = ptr->next;
    }
    ListNode *last = ptr;
    k %= cnt;
    if (k == 0)
        return head;
    ptr = head;
    for (int i = 0; i < cnt - k - 1; i++) {
        ptr = ptr->next;
    }
    ListNode *ret = ptr->next;
    ptr->next = nullptr;
    last->next = head;
    return ret;
}

int threeSumClosest(vector<int> &nums, int target) {
    std::sort(nums.begin(), nums.end());
    int ret = nums[0] + nums[1] + nums[2];
    int minGap = std::abs(target - ret);
    for (int i = 0; i + 3 <= nums.size(); i++) {
        if (nums[i] + nums[i + 1] + nums[i + 2] > target) {

            if (nums[i] + nums[i + 1] + nums[i + 2] - target < minGap) {
                ret = nums[i] + nums[i + 1] + nums[i + 2];
            }

            return ret;
        }
        int l = i + 1, r = nums.size() - 1;
        while (l < r) {
            const int sm = nums[l] + nums[r] + nums[i];
            if (sm > target) {
                if (sm - target < minGap) {
                    ret = sm;
                    minGap = sm - target;
                }
                r--;
            } else if (sm < target) {
                if (target - sm < minGap) {
                    ret = sm;
                    minGap = target - sm;
                }
                l++;
            } else {
                return target;
            }
        }
    }
    return ret;
}

int widthOfBinaryTree(TreeNode *root) {
    if (root == nullptr)
        return 0;
    std::queue<TreeNode *> q;
    q.emplace(root);
    root->val = 0;
    int ret = 0;
    while (!q.empty()) {
        ret = std::max(q.back()->val - q.front()->val, ret);
        // std::cout << q.size() << " " << q.front()->val << " " << q.back()->val << "\n";
        while (!q.empty()) {
            const auto *cur = q.front();
            if (cur->left == nullptr && cur->right == nullptr) {
                q.pop();
            } else {
                break;
            }
        }
        if (q.empty()) {
            break;
        }
        const int qz = q.size();
        // std::cout << qz << " ---- " << q.front()->val << " " << q.back()->val << "\n";
        const int base = q.front()->val;
        for (int i = 0; i < qz; i++) {
            auto *cur = q.front();
            q.pop();
            cur->val -= base;
            if (cur->left) {
                q.emplace(cur->left);
                cur->left->val = 2 * cur->val + 1;
            }
            if (cur->right) {
                q.emplace(cur->right);
                cur->right->val = 2 * cur->val + 2;
            }
        }
    }
    return ret + 1;
}

int diameterOfBinaryTree(TreeNode *root) {
    int ret = 0;
    std::function<int(TreeNode *)> dfs = [&](TreeNode *root) -> int {
        if (root == nullptr)
            return 0;
        const int l = dfs(root->left);
        const int r = dfs(root->right);
        ret = std::max(ret, l + r + 1);
        return std::max(l, r) + 1;
    };
    dfs(root);
    return ret;
}

int findPeakElement(vector<int> &nums) {
    int l = 0, r = nums.size() - 1;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (nums[mid] < nums[mid + 1]) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

string largestNumber(vector<int> &nums) {
    std::vector<std::string> strs;
    for (const int i : nums)
        strs.emplace_back(std::to_string(i));
    std::sort(strs.begin(), strs.end(),
              [](const std::string &a, const std::string &b) -> bool { return a + b > b + a; });
    if (strs.front() == "0")
        return "0";
    return std::accumulate(strs.begin(), strs.end(), std::string{});
}

int maxProduct(vector<int> &nums) {
    int maxSoFar = nums[0];
    int maxEndingHere = nums[0], minEndingHere = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        if (nums[i] < 0) {
            std::swap(maxEndingHere, minEndingHere);
        }
        maxEndingHere = std::max(maxEndingHere * nums[i], nums[i]);
        minEndingHere = std::min(minEndingHere * nums[i], nums[i]);
        maxSoFar = std::max(maxSoFar, maxEndingHere);
    }
    return maxSoFar;
}

vector<vector<int>> pathSum(TreeNode *root, int targetSum) {
    if (root == nullptr)
        return {};
    std::vector<int> tmp;
    std::vector<std::vector<int>> ret;
    std::function<void(TreeNode *, const int)> dfs = [&](TreeNode *root, const int k) {
        if (root == nullptr) {
            if (k == 0) {
                ret.emplace_back(tmp);
            }
            return;
        }
        tmp.emplace_back(root->val);
        if (root->left == nullptr) {
            dfs(root->right, k - root->val);
            tmp.pop_back();
            return;
        }
        if (root->right == nullptr) {
            dfs(root->left, k - root->val);
            tmp.pop_back();
            return;
        }
        dfs(root->left, k - root->val);
        dfs(root->right, k - root->val);
        tmp.pop_back();
    };
    dfs(root, targetSum);
    return ret;
}

int uniquePaths(int m, int n) {
    int dp[m + 1][n + 1];
    std::memset(dp, 0, sizeof(dp));
    dp[1][0] = 1;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dp[i + 1][j + 1] = dp[i][j + 1] + dp[i + 1][j];
        }
    }
    return dp[m][n];
}

int subarraySum(vector<int> &nums, int k) {
    std::unordered_map<int, int> smCnt{{0, 1}};
    int sm = 0;
    int ret = 0;
    for (const int i : nums) {
        sm += i;
        ret += smCnt[sm - k];
        smCnt[sm]++;
    }
    return ret;
}

int rob(vector<int> &nums) {
    int a = 0, b = 0;
    for (const int i : nums) {
        const int c = std::max(b, a + i);
        a = b;
        b = c;
    }
    return b;
}

bool hasPathSum(TreeNode *root, int targetSum) {
    if (root == nullptr)
        return false;
    std::function<bool(TreeNode *, const int)> dfs = [&](TreeNode *root, const int k) -> bool {
        if (root == nullptr) {
            return k == 0;
        }
        if (root->left == nullptr)
            return dfs(root->right, k - root->val);
        if (root->right == nullptr)
            return dfs(root->left, k - root->val);
        return dfs(root->left, k - root->val) || dfs(root->right, k - root->val);
    };
    return dfs(root, targetSum);
}

int minSubArrayLen(int target, vector<int> &nums) {
    typedef std::pair<long, int> pli;
    std::deque<pli> smIdxQ;
    smIdxQ.push_back({0, -1});
    long sm = 0;
    int ret = nums.size() + 1;
    for (int i = 0; i < nums.size(); i++) {
        sm += nums[i];
        while (!smIdxQ.empty() && sm - smIdxQ.front().first >= target) {
            ret = std::min(ret, i - smIdxQ.front().second);
            smIdxQ.pop_front();
        }
        while (!smIdxQ.empty() && sm <= smIdxQ.back().first) {
            smIdxQ.pop_back();
        }
        smIdxQ.emplace_back(sm, i);
    }
    if (ret > nums.size())
        return 0;
    return ret;
}

ListNode *swapPairs(ListNode *head) {
    ListNode dummy(0), *ptr = &dummy;
    while (head != nullptr && head->next != nullptr) {
        ListNode *tmp = head->next->next;
        auto *a = head;
        auto *b = head->next;
        ptr->next = b;
        ptr = b;
        ptr->next = a;
        ptr = a;
        head = tmp;
    }
    ptr->next = head;
    return dummy.next;
}

int calculate(string s) {
    std::vector<int> stk;
    int presign = '+';
    int idx = 0;
    while (idx < s.size()) {
        const char c = s[idx++];
        if (std::isspace(c))
            continue;
        if (std::isdigit(c)) {
            int start = idx - 1;
            while (idx < s.size() && std::isdigit(s[idx])) {
                idx++;
            }
            const int num = std::stoi(s.substr(start, idx - start));
            switch (presign) {
            case '+': {
                stk.emplace_back(num);
                break;
            }
            case '-': {
                stk.emplace_back(-num);
                break;
            }
            case '*': {
                stk.back() *= num;
                break;
            }
            case '/': {
                stk.back() /= num;
                break;
            }
            }
        } else {
            presign = c;
        }
    }
    return std::accumulate(stk.begin(), stk.end(), 0);
}

ListNode *deleteDuplicates111(ListNode *head) {
    if (head == nullptr)
        return nullptr;
    ListNode dummy(head->val + 1), *ptr = &dummy;
    while (head != nullptr) {
        if (ptr->val != head->val) {
            ptr->next = head;
            ptr = head;
        }
        head = head->next;
    }
    ptr->next = nullptr;
    return dummy.next;
}

TreeNode *invertTree(TreeNode *root) {
    if (root == nullptr)
        return nullptr;
    invertTree(root->left);
    invertTree(root->right);
    std::swap(root->left, root->right);
    return root;
}

int majorityElement(vector<int> &nums) {
    int cnt = 0, majority;
    for (const int i : nums) {
        if (cnt == 0) {
            majority = i;
        } else if (i == majority) {
            cnt++;
        } else {
            cnt--;
        }
    }
    return majority;
}

bool wordBreak(string s, vector<string> &wordDict) {
    std::vector<bool> memo(s.size());
    std::function<bool(const int)> dfs = [&](const int start) -> bool {
        if (start == s.size())
            return true;
        if (memo[start])
            return false;
        for (const auto &w : wordDict) {
            if (s.compare(start, w.size(), w) != 0)
                continue;
            if (dfs(start + w.size()))
                return true;
        }
        memo[start] = true;
        return false;
    };
    return dfs(0);
}

void moveZeroes(vector<int> &nums) {
    int p = -1;
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] != 0) {
            std::swap(nums[++p], nums[i]);
        }
    }
}

int findLength(vector<int> &nums1, vector<int> &nums2) {
    const int m = nums1.size(), n = nums2.size();
    std::vector<std::vector<short>> memo(m, std::vector<short>(n, -1));
    int ans = 0;
    std::function<int(const int, const int)> dfs = [&](const int i, const int j) -> int {
        if (i == m || j == n)
            return 0;
        if (memo[i][j] != -1)
            return memo[i][j];
        int ret = 0;
        if (nums1[i] == nums2[j]) {
            ret = dfs(i + 1, j + 1) + 1;
        }
        dfs(i + 1, j);
        dfs(i, j + 1);
        ans = std::max(ans, ret);
        memo[i][j] = ret;
        return ret;
    };
    dfs(0, 0);
    return ans;
}

vector<int> sortArray(vector<int> &nums) {
    int n = nums.size();
    std::function<void(const int)> heapify = [&](const int p) {
        int largest = p;
        const int l = 2 * p + 1;
        if (l < n && nums[largest] < nums[l]) {
            largest = l;
        }
        const int r = 2 * p + 2;
        if (r < n && nums[largest] < nums[r]) {
            largest = r;
        }
        if (largest != p) {
            std::swap(nums[p], nums[largest]);
            heapify(largest);
        }
    };
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(i);
    }
    for (int i = n - 1; i > 0; i--) {
        n--;
        std::swap(nums[0], nums[i]);
        heapify(0);
    }
    return nums;
}

vector<int> searchRange(vector<int> &nums, int target) {
    const int n = nums.size();
    int l = 0, r = n;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (nums[mid] < target) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    if (l == nums.size() || nums[l] != target)
        return {-1, -1};
    const int start = l;
    l = start + 1, r = n;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (nums[mid] == target) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return {start, l - 1};
}

class MinStack {
    std::vector<int> datas;
    std::vector<int> minData;

  public:
    MinStack() {}

    void push(int val) {
        datas.emplace_back(val);
        if (minData.empty() || val <= minData.back()) {
            minData.emplace_back(val);
        } else {
            minData.emplace_back(minData.back());
        }
    }

    void pop() {
        datas.pop_back();
        minData.pop_back();
    }

    int top() { return datas.back(); }

    int getMin() { return minData.back(); }
};

int sumNumbers(TreeNode *root) {
    int ret = 0;
    std::function<void(TreeNode *, const int)> dfs = [&](TreeNode *root, const int num) {
        if (root == nullptr) {
            ret += num;
            return;
        }
        const int newNum = num * 10 + root->val;
        if (root->left == nullptr)
            return dfs(root->right, newNum);
        if (root->right == nullptr)
            return dfs(root->left, newNum);
        dfs(root->left, newNum);
        dfs(root->right, newNum);
    };
    dfs(root, 0);
    return ret;
}

bool isSymmetric(TreeNode *root) {
    std::function<bool(TreeNode *, TreeNode *)> isSymmetricT = [&](TreeNode *left,
                                                                   TreeNode *right) -> bool {
        if (left == nullptr)
            return right == nullptr;
        if (right == nullptr)
            return false;
        return left->val == right->val && isSymmetricT(left->left, right->right) &&
               isSymmetricT(left->right, right->left);
    };
    return isSymmetricT(root->left, root->right);
}

string decodeString(string s) {
    int idx = 0;
    const int sz = s.size();
    auto getCnt = [&]() -> int {
        if (idx == sz || !std::isdigit(s[idx]))
            return 1;
        int start = idx++;
        while (idx < sz && std::isdigit(s[idx]))
            idx++;
        return std::stoi(s.substr(start, idx - start));
    };
    std::function<std::string()> dfs = [&]() -> std::string {
        std::string ret;
        while (idx < sz) {
            const char c = s[idx];
            if (std::isalpha(c)) {
                int start = idx++;
                while (idx < sz && std::isalpha(s[idx]))
                    idx++;
                ret.append(s.begin() + start, s.begin() + idx);
            } else if (std::isdigit(c)) {
                const int cnt = getCnt();
                idx++;
                std::string tmp = dfs();
                for (int i = 0; i < cnt; i++) {
                    ret.append(tmp);
                }
            } else {
                assert(c == ']');
                idx++;
                return ret;
            }
        }
        return ret;
    };
    return dfs();
}

int minPathSum(vector<vector<int>> &grid) {
    const int m = grid.size(), n = grid[0].size();
    uint32_t dp[m + 1][n + 1];
    std::fill(&dp[0][0], &dp[0][0] + (m + 1) * (n + 1), 100000);
    dp[1][0] = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dp[i + 1][j + 1] = std::min(dp[i][j + 1], dp[i + 1][j]) + grid[i][j];
        }
    }
    return dp[m][n];
}

vector<vector<int>> combinationSum(vector<int> &candidates, int target) {
    std::sort(candidates.begin(), candidates.end());
    std::vector<int> tmp;
    std::vector<std::vector<int>> ret;
    std::function<void(const int, const int)> dfs = [&](const int start, const int k) {
        assert(k >= 0);
        if (k == 0 || start == candidates.size()) {
            if (k == 0) {
                ret.emplace_back(tmp);
            }
            return;
        }
        for (int i = start; i < candidates.size(); i++) {
            if (candidates[i] > k)
                return;
            tmp.emplace_back(candidates[i]);
            dfs(i, k - candidates[i]);
            tmp.pop_back();
        }
    };
    dfs(0, target);
    return ret;
}

int rand7() { return 1; }

int rand10() {
    int c;
    do {
        const int a = rand7() - 1;
        const int b = rand7() - 1;
        c = a * 7 + b;
    } while (c > 40);
    return c % 10;
}

int maxDepth(TreeNode *root) {
    if (root == nullptr) {
        return 0;
    }
    return std::max(maxDepth(root->left), maxDepth(root->right)) + 1;
}

int longestConsecutive(vector<int> &nums) {
    std::unordered_set<int> us(nums.begin(), nums.end());
    int ret = 0;
    for (const int i : us) {
        if (us.count(i - 1) == 1)
            continue;
        int idx = i;
        int start = idx++;
        while (us.count(idx))
            idx++;
        ret = std::max(ret, idx - start);
    }
    return ret;
}

int maxAreaOfIsland(vector<vector<int>> &grid) {
    const int m = grid.size(), n = grid[0].size();
    constexpr static int dirs[4][2]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    auto isValid = [&](const int i, const int j) { return i >= 0 && i < m && j >= 0 && j < n; };
    std::function<int(const int, const int)> dfs = [&](const int i, const int j) -> int {
        if (!isValid(i, j))
            return 0;
        if (grid[i][j] == 0)
            return 0;
        grid[i][j] = 0;
        int ret = 0;
        for (const auto &dir : dirs) {
            ret += dfs(i + dir[0], j + dir[1]);
        }
        return ret + 1;
    };
    int ret = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            ret = std::max(ret, dfs(i, j));
        }
    }
    return ret;
}

bool isBalanced(TreeNode *root) {
    bool ret = true;
    std::function<int(TreeNode *)> dfs = [&](TreeNode *root) -> int {
        if (root == nullptr)
            return 0;
        int l = dfs(root->left);
        int r = dfs(root->right);
        ret &= std::abs(l - r) < 2;
        return std::max(l, r) + 1;
    };
    dfs(root);
    return ret;
}

vector<int> preorderTraversal(TreeNode *root) {
    if (root == nullptr)
        return {};
    std::vector<TreeNode *> stk{root};
    std::vector<int> ret;
    while (!stk.empty()) {
        const auto *cur = stk.back();
        stk.pop_back();
        ret.emplace_back(cur->val);
        if (cur->right) {
            stk.emplace_back(cur->right);
        }
        if (cur->left) {
            stk.emplace_back(cur->left);
        }
    }
    return ret;
}

int maximalSquare(vector<vector<char>> &matrix) {
    const int m = matrix.size(), n = matrix[0].size();
    int dp[m + 1][n + 1];
    std::memset(dp, 0, sizeof(dp));
    int ret = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] == '0')
                continue;
            dp[i + 1][j + 1] = std::min(dp[i + 1][j], std::min(dp[i][j + 1], dp[i][j])) + 1;
            ret = std::max(ret, dp[i + 1][j + 1]);
        }
    }
    return ret * ret;
}

bool isPalindrome(ListNode *head) {
    if (head == nullptr || head->next == nullptr)
        return true;
    ListNode *slow = head, *fast = head;
    do {
        slow = slow->next;
        fast = fast->next->next;
    } while (slow != fast);
    ListNode *ptr = nullptr;
    while (slow != nullptr) {
        ListNode *tmp = slow->next;
        slow->next = ptr;
        ptr = slow;
        slow = tmp;
    }
    while (ptr != nullptr) {
        if (ptr->val != head->val)
            return false;
        ptr = ptr->next;
        head = head->next;
    }
    return true;
}

int maxProfit2(vector<int> &prices) {
    int hold = -prices[0], sell = 0;
    for (const int i : prices) {
        sell = std::max(sell, hold + i);
        hold = std::max(hold, sell - i);
    }
    return sell;
}

bool searchMatrix(vector<vector<int>> &matrix, int target) {
    const int m = matrix.size(), n = matrix[0].size();
    int i = 0, j = n - 1;
    while (i < m && j >= 0) {
        if (matrix[i][j] == target)
            return true;
        if (matrix[i][j] > target) {
            j--;
        } else {
            i++;
        }
    }
    return false;
}

string longestCommonPrefix(vector<string> &strs) {
    auto getCommonPrefxLen = [](const std::string &a, const std::string &b) -> std::string {
        const int minL = std::min(a.size(), b.size());
        int i = 0;
        while (i < minL && a[i] == b[i])
            i++;
        return a.substr(0, i);
    };
    std::string x{strs[0]};
    for (const auto &i : strs) {
        x = getCommonPrefxLen(i, x);
    }
    return x;
}

vector<string> generateParenthesis(int n) {
    std::vector<std::string> ret;
    std::string tmp(2 * n, 0);
    std::function<void(const int, const int, const int)> dfs = [&](const int idx, const int l,
                                                                   const int r) {
        if (idx == 2 * n) {
            assert(l == r && r == n);
            ret.emplace_back(tmp);
            return;
        }
        if (l < n) {
            tmp[idx] = '(';
            dfs(idx + 1, l + 1, r);
        }
        if (r < l) {
            tmp[idx] = ')';
            dfs(idx + 1, l, r + 1);
        }
    };
    dfs(0, 0, 0);
    return ret;
}

ListNode *sortList(ListNode *head) {
    auto merge2List = [](ListNode *l, ListNode *r) -> ListNode * {
        ListNode dummy(0), *ptr = &dummy;
        while (l != nullptr && r != nullptr) {
            if (l->val < r->val) {
                ptr->next = l;
                ptr = l;
                l = l->next;
            } else {
                ptr->next = r;
                ptr = r;
                r = r->next;
            }
        }
        if (l != nullptr) {
            ptr->next = l;
        } else {
            ptr->next = r;
        }
        return dummy.next;
    };
    std::function<ListNode *(ListNode *)> mergeSort = [&](ListNode *head) -> ListNode * {
        if (head == nullptr || head->next == nullptr)
            return head;
        ListNode *fast = head->next, *slow = head;
        while (fast != nullptr && fast->next != nullptr) {
            slow = slow->next;
            fast = fast->next->next;
        }
        auto *r = mergeSort(slow->next);
        slow->next = nullptr;
        auto *l = mergeSort(head);
        return merge2List(l, r);
    };
    return mergeSort(head);
}

vector<int> maxSlidingWindow(vector<int> &nums, int k) {
    std::deque<int> idxQ;
    for (int i = 0; i + 1 < k; i++) {
        while (!idxQ.empty() && nums[i] >= nums[idxQ.back()])
            idxQ.pop_back();
        idxQ.emplace_back(i);
    }
    std::vector<int> ret;
    for (int i = k - 1; i < nums.size(); i++) {
        while (!idxQ.empty() && nums[i] >= nums[idxQ.back()])
            idxQ.pop_back();
        idxQ.emplace_back(i);
        ret.emplace_back(nums[idxQ.front()]);
        if (i - idxQ.front() + 1 == k)
            idxQ.pop_front();
    }
    return ret;
}

vector<int> inorderTraversal(TreeNode *root) {
    std::vector<TreeNode *> stk;
    auto pushAllLeft = [&](TreeNode *root) {
        while (root != nullptr) {
            stk.emplace_back(root);
        }
    };
    pushAllLeft(root);
    std::vector<int> ret;
    while (!stk.empty()) {
        const auto *cur = stk.back();
        stk.pop_back();
        ret.emplace_back(cur->val);
        pushAllLeft(cur->right);
    }
    return ret;
}

int mySqrt(int x) {
    long l = 0, r = (long)x + 1;
    while (l < r) {
        const long mid = l + (r - l) / 2;
        if (mid * mid <= x) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l - 1;
}

int longestValidParentheses(string s) {
    int ret = 0;
    int l = 0, r = 0;
    for (const char c : s) {
        if (c == '(') {
            l++;
        } else {
            r++;
            if (l == r) {
                ret = std::max(ret, 2 * l);
            } else if (l < r) {
                l = r = 0;
            }
        }
    }
    l = r = 0;
    for (int i = s.size() - 1; i >= 0; i--) {
        const char c = s[i];
        if (c == ')') {
            r++;
        } else {
            l++;
            if (l == r) {
                ret = std::max(ret, 2 * l);
            } else if (l > r) {
                l = r = 0;
            }
        }
    }
    return ret;
}

void nextPermutation(vector<int> &nums) {
    int i = nums.size() - 2;
    while (i >= 0 && nums[i] >= nums[i + 1])
        i--;
    if (i == -1) {
        std::reverse(nums.begin(), nums.end());
        return;
    }
    int idx = i + 1;
    const int val = nums[i];
    while (idx < nums.size() && val < nums[idx])
        idx++;
    std::swap(nums[i], nums[idx - 1]);
    std::reverse(nums.begin() + i + 1, nums.end());
    fp("nums: {}\n", nums);
}

int myAtoi(string s) {
    const int sz = s.size();
    int idx = 0;
    while (idx < sz && std::isspace(s[idx]))
        idx++;
    if (idx == sz || s[idx] != '+' && s[idx] != '-' && !std::isdigit(s[idx]))
        return 0;
    bool isPositive = true;
    if (s[idx] == '+' || s[idx] == '-') {
        if (s[idx++] == '-')
            isPositive = false;
    }
    if (idx == sz || !std::isdigit(s[idx]))
        return 0;
    int num = 0;
    int a = INT32_MAX / 10, b = INT32_MAX % 10;
    while (idx < sz && std::isdigit(s[idx])) {
        const int d = s[idx++] - '0';
        if (num >= a) {
            if (num > a) {
                if (isPositive) {
                    return INT32_MAX;
                } else {
                    return INT32_MIN;
                }
            } else {
                if (isPositive) {
                    if (d >= b) {
                        return INT32_MAX;
                    }
                } else {
                    if (d >= b + 1) {
                        return INT32_MIN;
                    }
                }
            }
        }
        num = num * 10 + d;
    }
    if (isPositive)
        return num;
    return -num;
}

int coinChange(vector<int> &coins, int amount) {
    int dp[amount + 1];
    dp[0] = 0;
    std::fill(dp + 1, dp + amount + 1, amount + 1);
    for (const int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] = std::min(dp[i], dp[i - coin] + 1);
        }
    }
    if (dp[amount] == amount + 1)
        return -1;
    return dp[amount];
}

string multiply(string num1, string num2) {
    if (num1 == "0" || num2 == "0")
        return "0";
    const int m = num1.size(), n = num2.size();
    int cache[m + n];
    std::memset(cache, 0, sizeof(cache));
    for (int i = 0; i < m; i++) {
        const int d1 = num1[m - 1 - i] - '0';
        for (int j = 0; j < n; j++) {
            const int d2 = num2[n - 1 - j] - '0';
            cache[i + j] += d1 * d2;
        }
    }
    int carry = 0;
    std::string ret;
    for (int i = 0; i < m + n; i++) {
        const int sm = carry + cache[i];
        ret.push_back(sm % 10 + '0');
        carry = sm / 10;
    }
    if (ret.back() == '0')
        ret.pop_back();
    std::reverse(ret.begin(), ret.end());
    return ret;
}

string minWindow(string s, string t) {
    int cnt[128]{}, k = t.size(), l = 0, r = 0, start = -1, minL = s.size() + 1;
    for (const char c : t)
        cnt[c]++;
    while (r < s.size()) {
        if (cnt[s[r++]]-- > 0)
            k--;
        if (k > 0)
            continue;
        while (cnt[s[l++]]++ < 0)
            ;
        k = 1;
        if (r + 1 - l < minL) {
            minL = r + 1 - l;
            start = l - 1;
        }
    }
    if (start == -1)
        return "";
    return s.substr(start, minL);
}

int firstMissingPositive(vector<int> &nums) {
    const int n = nums.size();
    for (int i = 0; i < n; i++) {
        while (nums[i] > 0 && nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
            std::swap(nums[i], nums[nums[i] - 1]);
        }
    }
    for (int i = 0; i < n; i++) {
        if (nums[i] != i + 1) {
            return i + 1;
        }
    }
    return n + 1;
}

vector<vector<int>> subsets(vector<int> &nums) {
    std::vector<int> tmp;
    std::vector<std::vector<int>> ret;
    std::function<void(const int)> dfs = [&](const int start) {
        ret.emplace_back(tmp);
        if (start == nums.size())
            return;
        for (int i = start; i < nums.size(); i++) {
            tmp.emplace_back(nums[i]);
            dfs(i + 1);
            tmp.pop_back();
        }
    };
    dfs(0);
    return ret;
}

string reverseWords(string s) {
    auto split = [&]() -> std::vector<std::string> {
        std::string tmp;
        std::vector<std::string> ret;
        for (const char c : s) {
            if (std::isspace(c)) {
                ret.emplace_back(std::move(tmp));
            } else {
                tmp.push_back(c);
            }
        }
        ret.emplace_back(tmp);
        return ret;
    };
    auto words = split();
    std::string ret;
    for (auto it = words.rbegin(); it != words.rend(); it++) {
        if (it->empty())
            continue;
        ret.append(*it);
        ret.push_back(' ');
    }
    ret.pop_back();
    return ret;
}

ListNode *trainingPlan(ListNode *head, int cnt) {
    ListNode dummy(0);
    head = &dummy;
    ListNode *fast = head;
    for (int i = 0; i < cnt; i++) {
        fast = fast->next;
    }
    while (fast != nullptr) {
        fast = fast->next;
        head = head->next;
    }
    return head;
}

TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder) {
    std::unordered_map<int, int> inValIdx;
    for (int i = 0; i < inorder.size(); i++) {
        inValIdx[inorder[i]] = i;
    }
    std::function<TreeNode *(const int, const int, const int)> dfs =
        [&](const int preIdx, const int inIdx, const int len) -> TreeNode * {
        if (len == 0)
            return nullptr;
        const int rootVal = preorder[preIdx];
        const int anchorIdx = inValIdx[rootVal];
        const int leftPreIdx = preIdx + 1;
        const int leftInIdx = inIdx;
        const int leftLen = anchorIdx - inIdx;
        TreeNode *root = new TreeNode(rootVal);
        root->left = dfs(leftPreIdx, leftInIdx, leftLen);
        const int rightPreIdx = preIdx + leftLen + 1;
        const int rightInIdx = anchorIdx + 1;
        const int rightLen = len - 1 - leftLen;
        root->right = dfs(rightPreIdx, rightInIdx, rightLen);
        return root;
    };
    return dfs(0, 0, preorder.size());
}

ListNode *reverseBetween(ListNode *head, int left, int right) {
    ListNode dummy(0);
    dummy.next = head;
    head = &dummy;
    for (int i = 0; i < left - 1; i++) {
        head = head->next;
    }
    ListNode *ptr = head;
    ListNode *l = head->next;
    ListNode *tmp = l;
    for (int i = 0; i < right + 1 - left; i++) {
        ptr = ptr->next;
    }
    ListNode *r = ptr;
    ListNode *rNext = r->next;
    r->next = nullptr;
    ptr = nullptr;
    while (l != nullptr) {
        ListNode *tmp = l->next;
        l->next = ptr;
        ptr = l;
        l = tmp;
    }
    head->next = ptr;
    tmp->next = rNext;
    return dummy.next;
}

bool hasCycle(ListNode *head) {
    ListNode *fast = head, *slow = head;
    while (fast != nullptr && fast->next != nullptr) {
        fast = fast->next->next;
        slow = slow->next;
        if (fast == slow)
            return true;
    }
    return false;
}

int lengthOfLIS(vector<int> &nums) {
    std::vector<int> lisVec;
    for (const int i : nums) {
        int l = 0, r = lisVec.size();
        while (l < r) {
            const int mid = l + (r - l) / 2;
            if (lisVec[mid] < i) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        if (l == lisVec.size()) {
            lisVec.emplace_back(i);
        } else {
            lisVec[l] = i;
        }
    }
    return lisVec.size();
}

vector<int> spiralOrder(vector<vector<int>> &matrix) {
    const int m = matrix.size(), n = matrix[0].size();
    int u = 0, d = m - 1, l = 0, r = n - 1;
    std::vector<int> ret;
    while (true) {
        for (int i = l; i <= r; i++) {
            ret.emplace_back(matrix[u][i]);
        }
        if (++u > d)
            break;
        for (int i = u; i <= d; i++) {
            ret.emplace_back(matrix[i][r]);
        }
        if (--r < l)
            break;
        for (int i = r; i >= l; i--) {
            ret.emplace_back(matrix[d][i]);
        }
        if (--d < u)
            break;
        for (int i = d; i >= u; i--) {
            ret.emplace_back(matrix[i][l]);
        }
        if (++l > r)
            break;
    }
    return ret;
}

ListNode *mergeKLists(vector<ListNode *> &lists) {
    auto merge2List = [](ListNode *l, ListNode *r) -> ListNode * {
        ListNode dummy(0), *ptr = &dummy;
        while (l != nullptr && r != nullptr) {
            if (l->val < r->val) {
                ptr->next = l;
                ptr = l;
                l = l->next;
            } else {
                ptr->next = r;
                ptr = r;
                r = r->next;
            }
        }
        if (l != nullptr) {
            ptr->next = l;
        } else {
            ptr->next = r;
        }
        return dummy.next;
    };
    std::function<ListNode *(const int, const int)> divideMerge = [&](const int l,
                                                                      const int r) -> ListNode * {
        if (l > r)
            return nullptr;
        if (l == r)
            return lists[l];
        const int mid = l + (r - l) / 2;
        auto *ll = divideMerge(l, mid);
        auto *rl = divideMerge(mid + 1, r);
        return merge2List(ll, rl);
    };
    return divideMerge(0, lists.size() - 1);
}

string addStrings(string num1, string num2) {
    std::string ret;
    int carry = 0, i = 0, j = 0;
    const int z1 = num1.size(), z2 = num2.size();
    while (i < z1 || j < z2 || carry > 0) {
        const int val1 = i < z1 ? num1[z1 - 1 - i++] - '0' : 0;
        const int val2 = j < z2 ? num2[z2 - 1 - j++] - '0' : 0;
        const int sm = val1 + val2 + carry;
        ret.push_back(sm % 10 + '0');
        carry = sm / 10;
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
}

vector<vector<int>> merge(vector<vector<int>> &intervals) {
    std::sort(intervals.begin(), intervals.end());
    std::vector<int> tmp = intervals.front();
    vector<vector<int>> ret;
    for (const auto &i : intervals) {
        if (tmp.back() < i.front()) {
            ret.emplace_back(std::move(tmp));
            tmp = i;
        } else {
            tmp[0] = std::min(tmp[0], i[0]);
            tmp[1] = std::max(tmp[1], i[1]);
        }
    }
    ret.emplace_back(tmp);
    return ret;
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

int trap(vector<int> &height) {
    int maxL = 0, maxR = 0;
    int l = 0, r = height.size() - 1;
    int ret = 0;
    while (l <= r) {
        if (maxL < maxR) {
            ret += std::max(0, maxL - height[l]);
            maxL = std::max(maxL, height[l++]);
        } else {
            ret += std::max(0, maxR - height[r]);
            maxR = std::max(maxR, height[r--]);
        }
    }
    return ret;
}

int maxPathSum(TreeNode *root) {
    int ret = root->val;
    std::function<int(TreeNode *)> dfs = [&](TreeNode *root) -> int {
        if (root == nullptr)
            return 0;
        const int l = dfs(root->left);
        const int r = dfs(root->right);
        ret = std::max(ret, l + r + root->val);
        return std::max(0, std::max(l, r) + root->val);
    };
    dfs(root);
    return ret;
}

vector<string> restoreIpAddresses(string s) {
    const int sz = s.size();
    if (sz > 12)
        return {};
    std::vector<std::string> ret;
    std::vector<std::string> tmp;
    std::function<void(const int)> dfs = [&](const int start) {
        if (start == s.size() || tmp.size() == 4) {
            if (start == s.size() && tmp.size() == 4) {
                std::string x;
                for (const auto &i : tmp) {
                    x.append(i);
                    x.push_back('.');
                }
                x.pop_back();
                ret.emplace_back(x);
            }
            return;
        }
        for (int i = start; i < s.size() && i < start + 3; i++) {
            if (i > start && s[start] == '0')
                break;
            const std::string str = s.substr(start, i - start + 1);
            if (std::stoi(str) > 255)
                break;
            tmp.emplace_back(str);
            dfs(i + 1);
            tmp.pop_back();
        }
    };
    dfs(0);
    return ret;
}

ListNode *deleteDuplicates(ListNode *head) {
    ListNode dummy(0), *ptr = &dummy;
    while (head != nullptr) {
        ListNode *it = head->next;
        while (it != nullptr && it->val == head->val)
            it = it->next;
        if (head->next == it) {
            ptr->next = head;
            ptr = head;
        }
        head = it;
    }
    ptr->next = nullptr;
    return dummy.next;
}

ListNode *removeNthFromEnd(ListNode *head, int n) {
    ListNode dummy(0);
    dummy.next = head;
    head = &dummy;
    ListNode *fast = head, *slow = head;
    for (int i = 0; i <= n; i++) {
        fast = fast->next;
    }
    while (fast != nullptr) {
        fast = fast->next;
        slow = slow->next;
    }
    slow->next = slow->next->next;
    return dummy.next;
}

ListNode *detectCycle(ListNode *head) {
    ListNode *fast = head, *slow = head;
    while (fast != nullptr && fast->next != nullptr) {
        fast = fast->next->next;
        slow = slow->next;
        if (fast == slow)
            break;
    }
    if (fast == nullptr || fast->next == nullptr)
        return nullptr;
    slow = head;
    while (slow != fast) {
        slow = slow->next;
        fast = fast->next;
    }
    return slow;
}

double findMedianSortedArrays(vector<int> &nums1, vector<int> &nums2) {
    const int m = nums1.size(), n = nums2.size();
    auto getKthEle = [&](int k) -> int {
        assert(k > 0 && k <= m + n);
        int i = 0, j = 0;
        while (true) {
            if (i == m)
                return nums2[j + k - 1];
            if (j == n)
                return nums1[i + k - 1];
            if (k == 1)
                return std::min(nums1[i], nums2[j]);
            const int ii = std::min(m - 1, i + k / 2 - 1);
            const int jj = std::min(n - 1, j + k / 2 - 1);
            if (nums1[ii] < nums2[jj]) {
                k -= (ii - i + 1);
                i = ii + 1;
            } else {
                k -= (jj - j + 1);
                j = jj + 1;
            }
        }
    };
    if ((m + n) & 1) {
        return getKthEle((m + n + 1) / 2);
    } else {
        return (getKthEle((m + n) / 2) + getKthEle((m + n) / 2 + 1)) / 2.0;
    }
}

vector<int> rightSideView(TreeNode *root) {
    std::vector<int> ret;
    std::function<void(TreeNode *, const int)> dfs = [&](TreeNode *root, const int level) {
        if (root == nullptr)
            return;
        if (ret.size() == level) {
            ret.emplace_back(root->val);
        } else {
            ret[level] = root->val;
        }
        dfs(root->left, level + 1);
        dfs(root->right, level + 1);
    };
    dfs(root, 0);
    return ret;
}

int compareVersion(string version1, string version2) {
    auto split = [&](const std::string &version) -> std::vector<std::string> {
        std::vector<std::string> ret;
        std::string tmp;
        for (const char c : version) {
            if (c == '.') {
                ret.emplace_back(std::move(tmp));
            } else {
                tmp.push_back(c);
            }
        }
        ret.emplace_back(tmp);
        return ret;
    };
    const auto vec1 = split(version1);
    const auto vec2 = split(version2);
    const int minL = std::min(vec1.size(), vec2.size());
    for (int i = 0; i < minL; i++) {
        const int a = std::stoi(vec1[i]);
        const int b = std::stoi(vec2[i]);
        if (a < b)
            return -1;
        if (a > b)
            return 1;
    }
    for (int i = minL; i < vec1.size(); i++) {
        const int a = std::stoi(vec1[i]);
        if (a > 0)
            return 1;
    }
    for (int i = minL; i < vec2.size(); i++) {
        const int b = std::stoi(vec2[i]);
        if (b > 0)
            return -1;
    }
    return 0;
}

int lengthOfLongestSubstring(string s) {
    char cnt[128]{};
    int l = 0, r = 0, ret = 0;
    while (r < s.size()) {
        if (cnt[s[r++]]++ == 0)
            continue;
        ret = std::max(ret, r - 1 - l);
        while (cnt[s[l++]]-- == 1)
            ;
    }
    return std::max(ret, (int)s.size() - l);
}

vector<int> twoSum(vector<int> &nums, int target) {
    std::unordered_map<int, int> valIdx;
    for (int i = 0; i < nums.size(); i++) {
        if (valIdx.count(target - nums[i])) {
            return {valIdx[target - nums[i]], i};
        } else {
            valIdx.emplace(nums[i], i);
        }
    }
    return {};
}

bool isValid(string s) {
    std::vector<char> stk;
    for (const char c : s) {
        if (c == '(') {
            stk.emplace_back(')');
        } else if (c == '{') {
            stk.emplace_back('}');
        } else if (c == '[') {
            stk.emplace_back(']');
        } else {
            if (stk.empty() || stk.back() != c)
                return false;
            stk.pop_back();
        }
    }
    return stk.empty();
}

class LRUCache {
    std::list<std::pair<int, int>> l;
    std::unordered_map<int, std::list<std::pair<int, int>>::iterator> keyIt;
    const int capcity;

  public:
    LRUCache(int capacity) : capcity(capacity) {}

    int get(int key) {
        auto it = keyIt.find(key);
        if (it == keyIt.end())
            return -1;
        const int ret = it->second->second;
        l.splice(l.begin(), l, it->second);
        it->second = l.begin();
        return ret;
    }

    void put(int key, int value) {
        auto it = keyIt.find(key);
        if (it == keyIt.end()) {
            if (keyIt.size() == capcity) {
                keyIt.erase(l.back().first);
                l.pop_back();
            }
            l.emplace_front(key, value);
            keyIt.emplace(key, l.begin());
        } else {
            it->second->second = value;
            l.splice(l.begin(), l, it->second);
        }
    }
};

ListNode *reverseKGroup(ListNode *head, int k) {
    auto reverseList = [](ListNode *head) -> ListNode * {
        ListNode *ptr = nullptr;
        while (head != nullptr) {
            ListNode *tmp = head->next;
            head->next = ptr;
            ptr = head;
            head = tmp;
        }
        return ptr;
    };
    ListNode dummy(0), *ptr = &dummy;
    ListNode *it = head;
    while (true) {
        int x = k;
        while (it != nullptr && x-- > 1)
            it = it->next;
        if (it == nullptr) {
            ptr->next = head;
            return dummy.next;
        }
        ListNode *nextHead = it->next;
        it->next = nullptr;
        ptr->next = reverseList(head);
        ptr = head;
        head = it = nextHead;
    }
}

vector<vector<int>> threeSum(vector<int> &nums) {
    std::sort(nums.begin(), nums.end());
    std::vector<std::vector<int>> ret;
    for (int i = 0; i + 3 <= nums.size(); i++) {
        if (nums[i] + nums[i + 1] + nums[i + 1] > 0)
            return ret;
        if (nums[i] + nums.back() + nums[nums.size() - 2] < 0)
            continue;
        if (i > 0 && nums[i] == nums[i - 1])
            continue;
        int l = i + 1, r = nums.size() - 1;
        while (l < r) {
            const int sm = nums[i] + nums[l] + nums[r];
            if (sm < 0) {
                l++;
            } else if (sm > 0) {
                r--;
            } else {
                ret.push_back({nums[i], nums[l], nums[r]});
                const int lVal = nums[l++];
                while (l < r && nums[l] == lVal)
                    l++;
                const int rVal = nums[r--];
                while (l < r && nums[r] == rVal)
                    r--;
            }
        }
    }
    return ret;
}

string longestPalindrome(string s) {
    std::string processedStr{"^#"};
    for (const char c : s) {
        processedStr.push_back(c);
        processedStr.push_back('#');
    }
    processedStr.push_back('$');
    int radius[processedStr.size()];
    radius[0] = radius[1] = 0;
    int maxCenterI = 1;
    for (int i = 2; i < processedStr.size() - 2; i++) {
        int r = 1;
        if (maxCenterI + radius[maxCenterI] > i) {
            r = std::min(maxCenterI + radius[maxCenterI] - i, radius[2 * maxCenterI - i]);
        }
        while (processedStr[i + r] == processedStr[i - r])
            r++;
        radius[i] = --r;
        if (i + r > maxCenterI + radius[maxCenterI]) {
            maxCenterI = i;
        }
    }
    const int *maxCenterIdx = std::max_element(radius + 2, radius + processedStr.size() - 2);
    const int idx = maxCenterIdx - radius;
    return s.substr((idx - radius[idx]) / 2, radius[idx]);
}

int search(vector<int> &nums, int target) {
    int l = 0, r = nums.size() - 1;
    while (l < r) {
        const int mid = l + (r - l) / 2;
        if (nums[mid] > nums[r]) {
            if (target > nums[r] && target <= nums[mid]) {
                r = mid;
            } else {
                l = mid + 1;
            }
        } else {
            if (target > nums[mid] && target <= nums[r]) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
    }
    if (nums[l] == target)
        return l;
    return -1;
}

vector<vector<int>> permute(vector<int> &nums) {
    std::vector<int> tmp;
    std::vector<std::vector<int>> ret;
    std::function<void(const int)> dfs = [&](const int start) {
        if (start == nums.size()) {
            ret.emplace_back(tmp);
            return;
        }
        for (int i = start; i < nums.size(); i++) {
            std::swap(nums[i], nums[start]);
            tmp.emplace_back(nums[start]);
            dfs(start + 1);
            tmp.pop_back();
            std::swap(nums[i], nums[start]);
        }
    };
    dfs(0);
    return ret;
}

void merge(vector<int> &nums1, int m, vector<int> &nums2, int n) {
    int p = m + n;
    m--;
    n--;
    while (m >= 0 && n >= 0) {
        if (nums1[m] > nums2[n]) {
            nums1[--p] = nums1[m--];
        } else {
            nums1[--p] = nums2[n--];
        }
    }
    while (n >= 0) {
        nums1[--p] = nums2[n--];
    }
}

int maxProfit(vector<int> &prices) {
    int ret = 0;
    int minPrice = prices[0];
    for (int i = 1; i < prices.size(); i++) {
        ret = std::max(ret, prices[i] - minPrice);
        minPrice = std::min(minPrice, prices[i]);
    }
    return ret;
}

} // namespace C

int main() {
    const auto ret = C::longestPalindrome("babad");
    fp("ret: {}\n", ret);
    return 0;
}
