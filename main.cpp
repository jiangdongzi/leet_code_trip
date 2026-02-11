#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fmt/core.h>
#include <functional>
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

int maxProfit(int k, vector<int> &prices) {
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
    for (int i = 1; i < psm.size(); i++) {
        auto ret = um.emplace(psm[i], 0);
    }
    return false;
}

} // namespace B

int main() {
    // const auto ret = B::isAdditiveNumber("112");
    // fp("ret: {}\n", ret);
    std::unordered_map<int, int> um;
    um[1] = 37;
    auto res = um.emplace(1, 13);
    fp("res: {}\n", res);
    return 0;
}
