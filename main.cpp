#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
    const int m = static_cast<int>(matrix.size());
    if (m == 0) {
        return {};
    }
    const int n = static_cast<int>(matrix[0].size());
    if (n == 0) {
        return {};
    }

    std::vector<int> sm(n, 0);
    int maxSoFar = std::numeric_limits<int>::min();
    int bestTop = 0, bestLeft = 0, bestBottom = 0, bestRight = 0;

    for (int top = 0; top < m; top++) {
        std::fill(sm.begin(), sm.end(), 0);
        for (int bottom = top; bottom < m; bottom++) {
            for (int col = 0; col < n; col++) {
                sm[col] += matrix[bottom][col];
            }

            int maxEndingHere = 0;
            int curLeft = 0;
            for (int col = 0; col < n; col++) {
                if (col == 0 || maxEndingHere <= 0) {
                    maxEndingHere = sm[col];
                    curLeft = col;
                } else {
                    maxEndingHere += sm[col];
                }

                if (maxEndingHere > maxSoFar) {
                    maxSoFar = maxEndingHere;
                    bestTop = top;
                    bestLeft = curLeft;
                    bestBottom = bottom;
                    bestRight = col;
                }
            }
        }
    }

    return {bestTop, bestLeft, bestBottom, bestRight};
}

} // namespace A

int main() {
    fp("hello leetcode + fmt\n");
    //[[-1,0],[0,-1]]
    std::vector<std::vector<int>> matrix{{-1, 0}, {0, -1}};
    auto res = A::getMaxMatrix(matrix);
    fp("result: {}\n", res);
    return 0;
}
