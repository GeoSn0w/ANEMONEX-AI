#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <queue>
#include <stack>
#include <list>
#include <deque>
#include <algorithm>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <chrono>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <typeinfo>
#include <type_traits>
#include <utility>
#include <tuple>
#include <array>
#include <bitset>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <cmath>
#include <optional>
#include <variant>
#include <any>
#include <numeric>
#include <iterator>
#include <regex>
#include <random>

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef signed char i8;
typedef short i16;
typedef int i32;
typedef long long i64;
typedef float f32;
typedef double f64;

#define ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, lo, hi) (MIN(MAX(x, lo), hi))
#define BIT(n) (1u << (n))
#define BITMASK(n) (BIT(n) - 1)

struct Vec2 {
    f32 x, y;
    Vec2() : x(0), y(0) {}
    Vec2(f32 x, f32 y) : x(x), y(y) {}
    Vec2 operator+(const Vec2 &o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2 &o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(f32 s) const { return {x * s, y * s}; }
    Vec2 &operator+=(const Vec2 &o) { x += o.x; y += o.y; return *this; }
    Vec2 &operator-=(const Vec2 &o) { x -= o.x; y -= o.y; return *this; }
    f32 dot(const Vec2 &o) const { return x * o.x + y * o.y; }
    f32 len() const { return sqrtf(x * x + y * y); }
    f32 lenSq() const { return x * x + y * y; }
    Vec2 norm() const { f32 l = len(); return l > 0 ? Vec2(x/l, y/l) : Vec2(); }
    f32 cross(const Vec2 &o) const { return x * o.y - y * o.x; }
    bool operator==(const Vec2 &o) const { return x == o.x && y == o.y; }
};

struct Vec3 {
    f32 x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(f32 x, f32 y, f32 z) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3 &o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3 &o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(f32 s) const { return {x*s, y*s, z*s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    f32 dot(const Vec3 &o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3 cross(const Vec3 &o) const {
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};
    }
    f32 len() const { return sqrtf(x*x + y*y + z*z); }
    Vec3 norm() const { f32 l = len(); return l > 0 ? Vec3(x/l, y/l, z/l) : Vec3(); }
    Vec3 &operator+=(const Vec3 &o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3 &operator-=(const Vec3 &o) { x-=o.x; y-=o.y; z-=o.z; return *this; }
    Vec3 &operator*=(f32 s) { x*=s; y*=s; z*=s; return *this; }
};

struct Mat4 {
    f32 m[4][4];
    Mat4() { memset(m, 0, sizeof(m)); }
    static Mat4 identity() {
        Mat4 r;
        r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.0f;
        return r;
    }
    Mat4 operator*(const Mat4 &o) const {
        Mat4 r;
        for(int i = 0; i < 4; i++)
            for(int j = 0; j < 4; j++)
                for(int k = 0; k < 4; k++)
                    r.m[i][j] += m[i][k] * o.m[k][j];
        return r;
    }
    Vec3 transform(const Vec3 &v) const {
        f32 x = m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3];
        f32 y = m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3];
        f32 z = m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3];
        return {x, y, z};
    }
    static Mat4 translate(f32 tx, f32 ty, f32 tz) {
        Mat4 r = identity();
        r.m[0][3] = tx; r.m[1][3] = ty; r.m[2][3] = tz;
        return r;
    }
    static Mat4 scale(f32 sx, f32 sy, f32 sz) {
        Mat4 r = identity();
        r.m[0][0] = sx; r.m[1][1] = sy; r.m[2][2] = sz;
        return r;
    }
    static Mat4 rotateX(f32 a) {
        Mat4 r = identity();
        r.m[1][1] = cosf(a); r.m[1][2] = -sinf(a);
        r.m[2][1] = sinf(a); r.m[2][2] = cosf(a);
        return r;
    }
    static Mat4 rotateY(f32 a) {
        Mat4 r = identity();
        r.m[0][0] = cosf(a); r.m[0][2] = sinf(a);
        r.m[2][0] = -sinf(a); r.m[2][2] = cosf(a);
        return r;
    }
};

struct Quat {
    f32 w, x, y, z;
    Quat() : w(1), x(0), y(0), z(0) {}
    Quat(f32 w, f32 x, f32 y, f32 z) : w(w), x(x), y(y), z(z) {}
    Quat operator*(const Quat &o) const {
        return {
            w*o.w - x*o.x - y*o.y - z*o.z,
            w*o.x + x*o.w + y*o.z - z*o.y,
            w*o.y - x*o.z + y*o.w + z*o.x,
            w*o.z + x*o.y - y*o.x + z*o.w
        };
    }
    Quat conjugate() const { return {w, -x, -y, -z}; }
    f32 norm() const { return sqrtf(w*w + x*x + y*y + z*z); }
    Quat normalized() const { f32 n = norm(); return {w/n, x/n, y/n, z/n}; }
    static Quat fromAxisAngle(Vec3 axis, f32 angle) {
        axis = axis.norm();
        f32 s = sinf(angle * 0.5f);
        return {cosf(angle * 0.5f), axis.x*s, axis.y*s, axis.z*s};
    }
    Vec3 rotate(const Vec3 &v) const {
        Quat p(0, v.x, v.y, v.z);
        Quat r = (*this) * p * conjugate();
        return {r.x, r.y, r.z};
    }
};

struct AABB {
    Vec3 mn, mx;
    AABB() : mn({1e30f,1e30f,1e30f}), mx({-1e30f,-1e30f,-1e30f}) {}
    AABB(Vec3 mn, Vec3 mx) : mn(mn), mx(mx) {}
    void expand(const Vec3 &p) {
        mn.x = MIN(mn.x, p.x); mn.y = MIN(mn.y, p.y); mn.z = MIN(mn.z, p.z);
        mx.x = MAX(mx.x, p.x); mx.y = MAX(mx.y, p.y); mx.z = MAX(mx.z, p.z);
    }
    bool contains(const Vec3 &p) const {
        return p.x >= mn.x && p.x <= mx.x &&
               p.y >= mn.y && p.y <= mx.y &&
               p.z >= mn.z && p.z <= mx.z;
    }
    bool intersects(const AABB &o) const {
        return mn.x <= o.mx.x && mx.x >= o.mn.x &&
               mn.y <= o.mx.y && mx.y >= o.mn.y &&
               mn.z <= o.mx.z && mx.z >= o.mn.z;
    }
    Vec3 center() const {
        return {(mn.x+mx.x)*0.5f, (mn.y+mx.y)*0.5f, (mn.z+mx.z)*0.5f};
    }
    Vec3 size() const { return {mx.x-mn.x, mx.y-mn.y, mx.z-mn.z}; }
};

template<typename T>
class Pool {
    struct Block {
        alignas(T) u8 data[sizeof(T)];
        bool used;
    };
    std::vector<Block> blocks;
    size_t cap;
public:
    Pool(size_t n) : blocks(n), cap(n) {
        for(auto &b : blocks) b.used = false;
    }
    T* alloc() {
        for(auto &b : blocks) {
            if(!b.used) {
                b.used = true;
                return reinterpret_cast<T*>(b.data);
            }
        }
        return nullptr;
    }
    void free(T *p) {
        p->~T();
        for(auto &b : blocks) {
            if(reinterpret_cast<T*>(b.data) == p) {
                b.used = false;
                return;
            }
        }
    }
    size_t capacity() const { return cap; }
};

class LinearAllocator {
    u8 *buf;
    size_t sz, pos;
public:
    LinearAllocator(size_t n) : buf(new u8[n]), sz(n), pos(0) {}
    ~LinearAllocator() { delete[] buf; }
    void *alloc(size_t n, size_t align = 8) {
        size_t aligned = ALIGN(pos, align);
        if(aligned + n > sz) return nullptr;
        pos = aligned + n;
        return buf + aligned;
    }
    void reset() { pos = 0; }
    size_t used() const { return pos; }
    size_t remaining() const { return sz - pos; }
};

template<typename T, size_t N>
class RingBuffer {
    T buf[N];
    size_t head, tail, count;
public:
    RingBuffer() : head(0), tail(0), count(0) {}
    bool push(const T &v) {
        if(count == N) return false;
        buf[tail] = v;
        tail = (tail + 1) % N;
        count++;
        return true;
    }
    bool pop(T &v) {
        if(count == 0) return false;
        v = buf[head];
        head = (head + 1) % N;
        count--;
        return true;
    }
    bool empty() const { return count == 0; }
    bool full() const { return count == N; }
    size_t size() const { return count; }
    T &peek() { return buf[head]; }
};

template<typename T>
class Intrusive {
public:
    T *prev, *next;
    Intrusive() : prev(nullptr), next(nullptr) {}
};

template<typename T>
class IntrusiveList {
    T *head, *tail;
    size_t n;
public:
    IntrusiveList() : head(nullptr), tail(nullptr), n(0) {}
    void pushBack(T *node) {
        node->prev = tail;
        node->next = nullptr;
        if(tail) tail->next = node;
        else head = node;
        tail = node;
        n++;
    }
    void pushFront(T *node) {
        node->next = head;
        node->prev = nullptr;
        if(head) head->prev = node;
        else tail = node;
        head = node;
        n++;
    }
    void remove(T *node) {
        if(node->prev) node->prev->next = node->next;
        else head = node->next;
        if(node->next) node->next->prev = node->prev;
        else tail = node->prev;
        node->prev = node->next = nullptr;
        n--;
    }
    T *front() { return head; }
    T *back() { return tail; }
    size_t size() const { return n; }
    bool empty() const { return n == 0; }
};

template<typename K, typename V>
class HashMap {
    struct Entry {
        K key;
        V val;
        bool occupied;
        bool tombstone;
        Entry() : occupied(false), tombstone(false) {}
    };
    std::vector<Entry> table;
    size_t count, cap;
    size_t hash(const K &k) const {
        return std::hash<K>{}(k) % cap;
    }
    void rehash() {
        size_t newCap = cap * 2;
        std::vector<Entry> newTable(newCap);
        for(auto &e : table) {
            if(!e.occupied || e.tombstone) continue;
            size_t h = std::hash<K>{}(e.key) % newCap;
            while(newTable[h].occupied) h = (h + 1) % newCap;
            newTable[h] = e;
        }
        table = std::move(newTable);
        cap = newCap;
    }
public:
    HashMap(size_t initCap = 16) : table(initCap), count(0), cap(initCap) {}
    void insert(const K &k, const V &v) {
        if(count * 2 >= cap) rehash();
        size_t h = hash(k);
        while(table[h].occupied && !table[h].tombstone && table[h].key != k)
            h = (h + 1) % cap;
        if(!table[h].occupied || table[h].tombstone) count++;
        table[h].key = k;
        table[h].val = v;
        table[h].occupied = true;
        table[h].tombstone = false;
    }
    V *find(const K &k) {
        size_t h = hash(k);
        size_t probed = 0;
        while(table[h].occupied && probed < cap) {
            if(!table[h].tombstone && table[h].key == k)
                return &table[h].val;
            h = (h + 1) % cap;
            probed++;
        }
        return nullptr;
    }
    bool remove(const K &k) {
        size_t h = hash(k);
        size_t probed = 0;
        while(table[h].occupied && probed < cap) {
            if(!table[h].tombstone && table[h].key == k) {
                table[h].tombstone = true;
                count--;
                return true;
            }
            h = (h + 1) % cap;
            probed++;
        }
        return false;
    }
    size_t size() const { return count; }
};

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int v) : val(v), left(nullptr), right(nullptr) {}
};

class BST {
    TreeNode *root;
    TreeNode *insertNode(TreeNode *n, int v) {
        if(!n) return new TreeNode(v);
        if(v < n->val) n->left = insertNode(n->left, v);
        else if(v > n->val) n->right = insertNode(n->right, v);
        return n;
    }
    TreeNode *minNode(TreeNode *n) {
        while(n->left) n = n->left;
        return n;
    }
    TreeNode *deleteNode(TreeNode *n, int v) {
        if(!n) return nullptr;
        if(v < n->val) n->left = deleteNode(n->left, v);
        else if(v > n->val) n->right = deleteNode(n->right, v);
        else {
            if(!n->left) { TreeNode *r = n->right; delete n; return r; }
            if(!n->right) { TreeNode *r = n->left; delete n; return r; }
            TreeNode *m = minNode(n->right);
            n->val = m->val;
            n->right = deleteNode(n->right, m->val);
        }
        return n;
    }
    bool searchNode(TreeNode *n, int v) {
        if(!n) return false;
        if(v == n->val) return true;
        if(v < n->val) return searchNode(n->left, v);
        return searchNode(n->right, v);
    }
    void inorderNode(TreeNode *n, std::vector<int> &out) {
        if(!n) return;
        inorderNode(n->left, out);
        out.push_back(n->val);
        inorderNode(n->right, out);
    }
    void destroyNode(TreeNode *n) {
        if(!n) return;
        destroyNode(n->left);
        destroyNode(n->right);
        delete n;
    }
public:
    BST() : root(nullptr) {}
    ~BST() { destroyNode(root); }
    void insert(int v) { root = insertNode(root, v); }
    void remove(int v) { root = deleteNode(root, v); }
    bool search(int v) { return searchNode(root, v); }
    std::vector<int> inorder() { std::vector<int> r; inorderNode(root, r); return r; }
};

template<typename T>
struct AVLNode {
    T val;
    AVLNode *left, *right;
    int height;
    AVLNode(T v) : val(v), left(nullptr), right(nullptr), height(1) {}
};

template<typename T>
class AVLTree {
    AVLNode<T> *root;
    int height(AVLNode<T> *n) { return n ? n->height : 0; }
    int bf(AVLNode<T> *n) { return n ? height(n->left) - height(n->right) : 0; }
    void updateH(AVLNode<T> *n) {
        n->height = 1 + MAX(height(n->left), height(n->right));
    }
    AVLNode<T> *rotR(AVLNode<T> *y) {
        AVLNode<T> *x = y->left, *t = x->right;
        x->right = y; y->left = t;
        updateH(y); updateH(x);
        return x;
    }
    AVLNode<T> *rotL(AVLNode<T> *x) {
        AVLNode<T> *y = x->right, *t = y->left;
        y->left = x; x->right = t;
        updateH(x); updateH(y);
        return y;
    }
    AVLNode<T> *balance(AVLNode<T> *n) {
        updateH(n);
        int b = bf(n);
        if(b > 1) {
            if(bf(n->left) < 0) n->left = rotL(n->left);
            return rotR(n);
        }
        if(b < -1) {
            if(bf(n->right) > 0) n->right = rotR(n->right);
            return rotL(n);
        }
        return n;
    }
    AVLNode<T> *insert(AVLNode<T> *n, T v) {
        if(!n) return new AVLNode<T>(v);
        if(v < n->val) n->left = insert(n->left, v);
        else if(v > n->val) n->right = insert(n->right, v);
        return balance(n);
    }
    void destroy(AVLNode<T> *n) {
        if(!n) return;
        destroy(n->left); destroy(n->right); delete n;
    }
public:
    AVLTree() : root(nullptr) {}
    ~AVLTree() { destroy(root); }
    void insert(T v) { root = insert(root, v); }
    int height() { return height(root); }
};

class Trie {
    struct Node {
        std::unordered_map<char, Node*> children;
        bool end;
        Node() : end(false) {}
        ~Node() { for(auto &p : children) delete p.second; }
    };
    Node *root;
public:
    Trie() : root(new Node()) {}
    ~Trie() { delete root; }
    void insert(const std::string &w) {
        Node *cur = root;
        for(char c : w) {
            if(!cur->children.count(c))
                cur->children[c] = new Node();
            cur = cur->children[c];
        }
        cur->end = true;
    }
    bool search(const std::string &w) {
        Node *cur = root;
        for(char c : w) {
            if(!cur->children.count(c)) return false;
            cur = cur->children[c];
        }
        return cur->end;
    }
    bool startsWith(const std::string &prefix) {
        Node *cur = root;
        for(char c : prefix) {
            if(!cur->children.count(c)) return false;
            cur = cur->children[c];
        }
        return true;
    }
    std::vector<std::string> autocomplete(const std::string &prefix) {
        Node *cur = root;
        for(char c : prefix) {
            if(!cur->children.count(c)) return {};
            cur = cur->children[c];
        }
        std::vector<std::string> res;
        std::function<void(Node*, std::string)> dfs = [&](Node *n, std::string s) {
            if(n->end) res.push_back(s);
            for(auto &[c, child] : n->children)
                dfs(child, s + c);
        };
        dfs(cur, prefix);
        return res;
    }
};

template<typename T>
class SegTree {
    std::vector<T> tree;
    size_t n;
    T combine(T a, T b) { return a + b; }
    void build(const std::vector<T> &arr, size_t node, size_t s, size_t e) {
        if(s == e) { tree[node] = arr[s]; return; }
        size_t mid = (s + e) / 2;
        build(arr, 2*node, s, mid);
        build(arr, 2*node+1, mid+1, e);
        tree[node] = combine(tree[2*node], tree[2*node+1]);
    }
    void update(size_t node, size_t s, size_t e, size_t i, T v) {
        if(s == e) { tree[node] = v; return; }
        size_t mid = (s + e) / 2;
        if(i <= mid) update(2*node, s, mid, i, v);
        else update(2*node+1, mid+1, e, i, v);
        tree[node] = combine(tree[2*node], tree[2*node+1]);
    }
    T query(size_t node, size_t s, size_t e, size_t l, size_t r) {
        if(r < s || e < l) return T{};
        if(l <= s && e <= r) return tree[node];
        size_t mid = (s + e) / 2;
        return combine(query(2*node, s, mid, l, r), query(2*node+1, mid+1, e, l, r));
    }
public:
    SegTree(const std::vector<T> &arr) : n(arr.size()), tree(4 * arr.size()) {
        if(!arr.empty()) build(arr, 1, 0, n-1);
    }
    void update(size_t i, T v) { update(1, 0, n-1, i, v); }
    T query(size_t l, size_t r) { return query(1, 0, n-1, l, r); }
};

class UnionFind {
    std::vector<int> parent, rank;
public:
    UnionFind(int n) : parent(n), rank(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) {
        if(parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if(px == py) return false;
        if(rank[px] < rank[py]) std::swap(px, py);
        parent[py] = px;
        if(rank[px] == rank[py]) rank[px]++;
        return true;
    }
    bool connected(int x, int y) { return find(x) == find(y); }
};

class Graph {
    int V;
    std::vector<std::vector<std::pair<int,int>>> adj;
public:
    Graph(int v) : V(v), adj(v) {}
    void addEdge(int u, int v, int w = 1) {
        adj[u].push_back({v, w});
    }
    void addUndirected(int u, int v, int w = 1) {
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    std::vector<int> bfs(int src) {
        std::vector<bool> visited(V, false);
        std::vector<int> order;
        std::queue<int> q;
        visited[src] = true;
        q.push(src);
        while(!q.empty()) {
            int u = q.front(); q.pop();
            order.push_back(u);
            for(auto [v, w] : adj[u]) {
                if(!visited[v]) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }
        return order;
    }
    std::vector<int> dijkstra(int src) {
        std::vector<int> dist(V, INT_MAX);
        std::priority_queue<std::pair<int,int>, std::vector<std::pair<int,int>>, std::greater<>> pq;
        dist[src] = 0;
        pq.push({0, src});
        while(!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if(d > dist[u]) continue;
            for(auto [v, w] : adj[u]) {
                if(dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    pq.push({dist[v], v});
                }
            }
        }
        return dist;
    }
    std::vector<std::vector<int>> floydWarshall() {
        std::vector<std::vector<int>> d(V, std::vector<int>(V, INT_MAX/2));
        for(int i = 0; i < V; i++) d[i][i] = 0;
        for(int u = 0; u < V; u++)
            for(auto [v, w] : adj[u])
                d[u][v] = MIN(d[u][v], w);
        for(int k = 0; k < V; k++)
            for(int i = 0; i < V; i++)
                for(int j = 0; j < V; j++)
                    if(d[i][k] + d[k][j] < d[i][j])
                        d[i][j] = d[i][k] + d[k][j];
        return d;
    }
    std::vector<int> topologicalSort() {
        std::vector<int> indegree(V, 0);
        for(int u = 0; u < V; u++)
            for(auto [v, w] : adj[u]) indegree[v]++;
        std::queue<int> q;
        for(int i = 0; i < V; i++)
            if(indegree[i] == 0) q.push(i);
        std::vector<int> order;
        while(!q.empty()) {
            int u = q.front(); q.pop();
            order.push_back(u);
            for(auto [v, w] : adj[u])
                if(--indegree[v] == 0) q.push(v);
        }
        return order;
    }
    bool hasCycle() {
        std::vector<int> color(V, 0);
        std::function<bool(int)> dfs = [&](int u) -> bool {
            color[u] = 1;
            for(auto [v, w] : adj[u]) {
                if(color[v] == 1) return true;
                if(color[v] == 0 && dfs(v)) return true;
            }
            color[u] = 2;
            return false;
        };
        for(int i = 0; i < V; i++)
            if(color[i] == 0 && dfs(i)) return true;
        return false;
    }
    std::vector<std::vector<int>> kosarajuSCC() {
        std::vector<int> order;
        std::vector<bool> visited(V, false);
        std::function<void(int)> dfs1 = [&](int u) {
            visited[u] = true;
            for(auto [v, w] : adj[u])
                if(!visited[v]) dfs1(v);
            order.push_back(u);
        };
        for(int i = 0; i < V; i++)
            if(!visited[i]) dfs1(i);
        std::vector<std::vector<int>> radj(V);
        for(int u = 0; u < V; u++)
            for(auto [v, w] : adj[u]) radj[v].push_back(u);
        std::fill(visited.begin(), visited.end(), false);
        std::vector<std::vector<int>> sccs;
        std::function<void(int, std::vector<int>&)> dfs2 = [&](int u, std::vector<int> &scc) {
            visited[u] = true;
            scc.push_back(u);
            for(int v : radj[u])
                if(!visited[v]) dfs2(v, scc);
        };
        for(int i = V-1; i >= 0; i--) {
            int u = order[i];
            if(!visited[u]) {
                sccs.emplace_back();
                dfs2(u, sccs.back());
            }
        }
        return sccs;
    }
};

template<typename T>
class MaxHeap {
    std::vector<T> data;
    void siftUp(int i) {
        while(i > 0) {
            int p = (i - 1) / 2;
            if(data[p] < data[i]) { std::swap(data[p], data[i]); i = p; }
            else break;
        }
    }
    void siftDown(int i) {
        int n = data.size();
        while(true) {
            int l = 2*i+1, r = 2*i+2, largest = i;
            if(l < n && data[l] > data[largest]) largest = l;
            if(r < n && data[r] > data[largest]) largest = r;
            if(largest == i) break;
            std::swap(data[i], data[largest]);
            i = largest;
        }
    }
public:
    void push(T v) { data.push_back(v); siftUp(data.size()-1); }
    T pop() { T top = data[0]; data[0] = data.back(); data.pop_back(); if(!data.empty()) siftDown(0); return top; }
    T top() const { return data[0]; }
    bool empty() const { return data.empty(); }
    size_t size() const { return data.size(); }
};

void quickSort(std::vector<int> &arr, int lo, int hi) {
    if(lo >= hi) return;
    int pivot = arr[(lo + hi) / 2];
    int i = lo, j = hi;
    while(i <= j) {
        while(arr[i] < pivot) i++;
        while(arr[j] > pivot) j--;
        if(i <= j) { std::swap(arr[i], arr[j]); i++; j--; }
    }
    quickSort(arr, lo, j);
    quickSort(arr, i, hi);
}

void mergeSort(std::vector<int> &arr, int lo, int hi) {
    if(lo >= hi) return;
    int mid = (lo + hi) / 2;
    mergeSort(arr, lo, mid);
    mergeSort(arr, mid+1, hi);
    std::vector<int> tmp;
    int i = lo, j = mid+1;
    while(i <= mid && j <= hi) {
        if(arr[i] <= arr[j]) tmp.push_back(arr[i++]);
        else tmp.push_back(arr[j++]);
    }
    while(i <= mid) tmp.push_back(arr[i++]);
    while(j <= hi) tmp.push_back(arr[j++]);
    for(int k = lo; k <= hi; k++) arr[k] = tmp[k-lo];
}

void heapSort(std::vector<int> &arr) {
    int n = arr.size();
    for(int i = n/2 - 1; i >= 0; i--) {
        int k = i;
        while(true) {
            int l = 2*k+1, r = 2*k+2, largest = k;
            if(l < n && arr[l] > arr[largest]) largest = l;
            if(r < n && arr[r] > arr[largest]) largest = r;
            if(largest == k) break;
            std::swap(arr[k], arr[largest]); k = largest;
        }
    }
    for(int i = n-1; i > 0; i--) {
        std::swap(arr[0], arr[i]);
        int k = 0;
        while(true) {
            int l = 2*k+1, r = 2*k+2, largest = k;
            if(l < i && arr[l] > arr[largest]) largest = l;
            if(r < i && arr[r] > arr[largest]) largest = r;
            if(largest == k) break;
            std::swap(arr[k], arr[largest]); k = largest;
        }
    }
}

void radixSort(std::vector<int> &arr) {
    if(arr.empty()) return;
    int mx = *std::max_element(arr.begin(), arr.end());
    for(int exp = 1; mx/exp > 0; exp *= 10) {
        std::vector<int> out(arr.size());
        int cnt[10] = {};
        for(int x : arr) cnt[(x/exp)%10]++;
        for(int i = 1; i < 10; i++) cnt[i] += cnt[i-1];
        for(int i = arr.size()-1; i >= 0; i--) {
            out[--cnt[(arr[i]/exp)%10]] = arr[i];
        }
        arr = out;
    }
}

int binarySearch(const std::vector<int> &arr, int target) {
    int lo = 0, hi = arr.size()-1;
    while(lo <= hi) {
        int mid = lo + (hi-lo)/2;
        if(arr[mid] == target) return mid;
        if(arr[mid] < target) lo = mid+1;
        else hi = mid-1;
    }
    return -1;
}

int lowerBound(const std::vector<int> &arr, int target) {
    int lo = 0, hi = arr.size();
    while(lo < hi) {
        int mid = lo + (hi-lo)/2;
        if(arr[mid] < target) lo = mid+1;
        else hi = mid;
    }
    return lo;
}

std::vector<int> kmpSearch(const std::string &text, const std::string &pat) {
    std::string s = pat + "#" + text;
    int n = s.size(), m = pat.size();
    std::vector<int> pi(n, 0);
    for(int i = 1; i < n; i++) {
        int j = pi[i-1];
        while(j > 0 && s[i] != s[j]) j = pi[j-1];
        if(s[i] == s[j]) j++;
        pi[i] = j;
    }
    std::vector<int> matches;
    for(int i = m+1; i < n; i++)
        if(pi[i] == m) matches.push_back(i - 2*m);
    return matches;
}

std::vector<int> zFunction(const std::string &s) {
    int n = s.size();
    std::vector<int> z(n, 0);
    z[0] = n;
    int l = 0, r = 0;
    for(int i = 1; i < n; i++) {
        if(i < r) z[i] = MIN(r-i, z[i-l]);
        while(i+z[i] < n && s[z[i]] == s[i+z[i]]) z[i]++;
        if(i+z[i] > r) { l = i; r = i+z[i]; }
    }
    return z;
}

int editDistance(const std::string &a, const std::string &b) {
    int m = a.size(), n = b.size();
    std::vector<std::vector<int>> dp(m+1, std::vector<int>(n+1));
    for(int i = 0; i <= m; i++) dp[i][0] = i;
    for(int j = 0; j <= n; j++) dp[0][j] = j;
    for(int i = 1; i <= m; i++)
        for(int j = 1; j <= n; j++) {
            if(a[i-1] == b[j-1]) dp[i][j] = dp[i-1][j-1];
            else dp[i][j] = 1 + MIN(dp[i-1][j], MIN(dp[i][j-1], dp[i-1][j-1]));
        }
    return dp[m][n];
}

std::string longestCommonSubsequence(const std::string &a, const std::string &b) {
    int m = a.size(), n = b.size();
    std::vector<std::vector<int>> dp(m+1, std::vector<int>(n+1, 0));
    for(int i = 1; i <= m; i++)
        for(int j = 1; j <= n; j++) {
            if(a[i-1] == b[j-1]) dp[i][j] = dp[i-1][j-1] + 1;
            else dp[i][j] = MAX(dp[i-1][j], dp[i][j-1]);
        }
    std::string lcs;
    int i = m, j = n;
    while(i > 0 && j > 0) {
        if(a[i-1] == b[j-1]) { lcs += a[i-1]; i--; j--; }
        else if(dp[i-1][j] > dp[i][j-1]) i--;
        else j--;
    }
    std::reverse(lcs.begin(), lcs.end());
    return lcs;
}

int knapsack01(const std::vector<int> &w, const std::vector<int> &v, int cap) {
    int n = w.size();
    std::vector<std::vector<int>> dp(n+1, std::vector<int>(cap+1, 0));
    for(int i = 1; i <= n; i++)
        for(int j = 0; j <= cap; j++) {
            dp[i][j] = dp[i-1][j];
            if(j >= w[i-1]) dp[i][j] = MAX(dp[i][j], dp[i-1][j-w[i-1]] + v[i-1]);
        }
    return dp[n][cap];
}

std::vector<int> longestIncreasingSubsequence(const std::vector<int> &arr) {
    std::vector<int> tails;
    std::vector<int> idx(arr.size()), parent(arr.size(), -1);
    for(int i = 0; i < (int)arr.size(); i++) {
        int pos = std::lower_bound(tails.begin(), tails.end(), arr[i]) - tails.begin();
        if(pos == (int)tails.size()) tails.push_back(arr[i]);
        else tails[pos] = arr[i];
        idx[i] = pos;
        if(pos > 0) {
            for(int j = i-1; j >= 0; j--) {
                if(idx[j] == pos-1 && arr[j] < arr[i]) { parent[i] = j; break; }
            }
        }
    }
    std::vector<int> lis;
    int cur = -1;
    for(int i = arr.size()-1; i >= 0; i--)
        if(idx[i] == (int)tails.size()-1 && cur == -1) cur = i;
    while(cur != -1) { lis.push_back(arr[cur]); cur = parent[cur]; }
    std::reverse(lis.begin(), lis.end());
    return lis;
}

class StringView {
    const char *ptr;
    size_t len;
public:
    StringView() : ptr(nullptr), len(0) {}
    StringView(const char *s) : ptr(s), len(strlen(s)) {}
    StringView(const char *s, size_t n) : ptr(s), len(n) {}
    StringView(const std::string &s) : ptr(s.data()), len(s.size()) {}
    char operator[](size_t i) const { return ptr[i]; }
    size_t size() const { return len; }
    bool empty() const { return len == 0; }
    const char *data() const { return ptr; }
    StringView substr(size_t off, size_t n = std::string::npos) const {
        if(off >= len) return {};
        n = MIN(n, len - off);
        return {ptr + off, n};
    }
    bool startsWith(StringView prefix) const {
        if(prefix.len > len) return false;
        return memcmp(ptr, prefix.ptr, prefix.len) == 0;
    }
    bool operator==(StringView o) const {
        return len == o.len && memcmp(ptr, o.ptr, len) == 0;
    }
    std::string str() const { return {ptr, len}; }
};

class StringBuilder {
    std::vector<char> buf;
public:
    StringBuilder &append(StringView s) {
        buf.insert(buf.end(), s.data(), s.data() + s.size());
        return *this;
    }
    StringBuilder &append(char c) {
        buf.push_back(c);
        return *this;
    }
    StringBuilder &appendInt(i64 v) {
        std::string s = std::to_string(v);
        return append(s);
    }
    std::string build() const { return {buf.begin(), buf.end()}; }
    void clear() { buf.clear(); }
    size_t size() const { return buf.size(); }
};

namespace crypto {
    u32 rotl(u32 x, int n) { return (x << n) | (x >> (32-n)); }
    u32 rotr(u32 x, int n) { return (x >> n) | (x << (32-n)); }

    std::array<u8, 16> md5(const u8 *data, size_t len) {
        static const u32 K[64] = {
            0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
            0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
            0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
            0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
            0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
            0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
            0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
            0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
        };
        static const int S[64] = {
            7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
            5, 9,14,20,5, 9,14,20,5, 9,14,20,5, 9,14,20,
            4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
            6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21
        };
        u32 a0=0x67452301, b0=0xefcdab89, c0=0x98badcfe, d0=0x10325476;
        size_t origLen = len;
        size_t padLen = ((len + 8) / 64 + 1) * 64;
        std::vector<u8> msg(padLen, 0);
        memcpy(msg.data(), data, len);
        msg[len] = 0x80;
        u64 bitLen = (u64)origLen * 8;
        memcpy(msg.data() + padLen - 8, &bitLen, 8);
        for(size_t i = 0; i < padLen; i += 64) {
            u32 M[16];
            memcpy(M, msg.data() + i, 64);
            u32 A=a0, B=b0, C=c0, D=d0;
            for(int j = 0; j < 64; j++) {
                u32 F, g;
                if(j < 16) { F=(B&C)|(~B&D); g=j; }
                else if(j < 32) { F=(D&B)|(~D&C); g=(5*j+1)%16; }
                else if(j < 48) { F=B^C^D; g=(3*j+5)%16; }
                else { F=C^(B|~D); g=(7*j)%16; }
                F=F+A+K[j]+M[g];
                A=D; D=C; C=B; B=B+rotl(F,S[j]);
            }
            a0+=A; b0+=B; c0+=C; d0+=D;
        }
        std::array<u8,16> digest;
        memcpy(digest.data(), &a0, 4);
        memcpy(digest.data()+4, &b0, 4);
        memcpy(digest.data()+8, &c0, 4);
        memcpy(digest.data()+12, &d0, 4);
        return digest;
    }

    std::array<u8, 20> sha1(const u8 *data, size_t len) {
        u32 h0=0x67452301, h1=0xEFCDAB89, h2=0x98BADCFE, h3=0x10325476, h4=0xC3D2E1F0;
        size_t padLen = ((len + 8) / 64 + 1) * 64;
        std::vector<u8> msg(padLen, 0);
        memcpy(msg.data(), data, len);
        msg[len] = 0x80;
        u64 bitLen = __builtin_bswap64((u64)len * 8);
        memcpy(msg.data() + padLen - 8, &bitLen, 8);
        for(size_t i = 0; i < padLen; i += 64) {
            u32 w[80];
            for(int j = 0; j < 16; j++) {
                w[j] = ((u32)msg[i+j*4] << 24) | ((u32)msg[i+j*4+1] << 16) |
                        ((u32)msg[i+j*4+2] << 8) | msg[i+j*4+3];
            }
            for(int j = 16; j < 80; j++)
                w[j] = rotl(w[j-3]^w[j-8]^w[j-14]^w[j-16], 1);
            u32 a=h0, b=h1, c=h2, d=h3, e=h4;
            for(int j = 0; j < 80; j++) {
                u32 f, k;
                if(j < 20) { f=(b&c)|(~b&d); k=0x5A827999; }
                else if(j < 40) { f=b^c^d; k=0x6ED9EBA1; }
                else if(j < 60) { f=(b&c)|(b&d)|(c&d); k=0x8F1BBCDC; }
                else { f=b^c^d; k=0xCA62C1D6; }
                u32 tmp = rotl(a,5) + f + e + k + w[j];
                e=d; d=c; c=rotl(b,30); b=a; a=tmp;
            }
            h0+=a; h1+=b; h2+=c; h3+=d; h4+=e;
        }
        std::array<u8,20> digest;
        u32 hh[5] = {h0,h1,h2,h3,h4};
        for(int i = 0; i < 5; i++) {
            u32 v = __builtin_bswap32(hh[i]);
            memcpy(digest.data() + i*4, &v, 4);
        }
        return digest;
    }

    class AES128 {
        static const u8 sbox[256];
        static const u8 invSbox[256];
        u8 rk[11][4][4];
        u8 gmul(u8 a, u8 b) {
            u8 p = 0;
            for(int i = 0; i < 8; i++) {
                if(b & 1) p ^= a;
                bool hiBit = a & 0x80;
                a <<= 1;
                if(hiBit) a ^= 0x1b;
                b >>= 1;
            }
            return p;
        }
        void subBytes(u8 s[4][4]) {
            for(int i=0;i<4;i++) for(int j=0;j<4;j++) s[i][j]=sbox[s[i][j]];
        }
        void shiftRows(u8 s[4][4]) {
            for(int r=1;r<4;r++) {
                u8 tmp[4];
                for(int c=0;c<4;c++) tmp[c] = s[r][(c+r)%4];
                memcpy(s[r], tmp, 4);
            }
        }
        void mixCols(u8 s[4][4]) {
            for(int c=0;c<4;c++) {
                u8 a=s[0][c], b=s[1][c], cx=s[2][c], d=s[3][c];
                s[0][c]=gmul(2,a)^gmul(3,b)^cx^d;
                s[1][c]=a^gmul(2,b)^gmul(3,cx)^d;
                s[2][c]=a^b^gmul(2,cx)^gmul(3,d);
                s[3][c]=gmul(3,a)^b^cx^gmul(2,d);
            }
        }
        void addRoundKey(u8 s[4][4], int round) {
            for(int r=0;r<4;r++) for(int c=0;c<4;c++) s[r][c]^=rk[round][r][c];
        }
    public:
        void setKey(const u8 key[16]);
        void encrypt(const u8 in[16], u8 out[16]);
        void decrypt(const u8 in[16], u8 out[16]);
    };

    const u8 AES128::sbox[256] = {
        0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
        0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
        0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
        0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
        0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
        0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
        0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
        0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
        0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
        0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
        0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
        0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
        0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
        0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
        0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
        0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
    };

    const u8 AES128::invSbox[256] = {
        0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
        0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
        0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
        0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
        0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
        0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
        0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
        0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
        0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
        0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
        0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
        0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
        0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
        0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
        0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
        0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d
    };

    void AES128::setKey(const u8 key[16]) {
        static const u8 rcon[10] = {0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36};
        for(int r=0;r<4;r++) for(int c=0;c<4;c++) rk[0][r][c] = key[r+4*c];
        for(int i=1;i<=10;i++) {
            u8 tmp[4];
            for(int r=0;r<4;r++) tmp[r] = rk[i-1][r][3];
            u8 t0=tmp[0]; tmp[0]=sbox[tmp[1]]^rcon[i-1]; tmp[1]=sbox[tmp[2]]; tmp[2]=sbox[tmp[3]]; tmp[3]=sbox[t0];
            for(int r=0;r<4;r++) rk[i][r][0] = rk[i-1][r][0]^tmp[r];
            for(int c=1;c<4;c++) for(int r=0;r<4;r++) rk[i][r][c] = rk[i-1][r][c]^rk[i][r][c-1];
        }
    }

    void AES128::encrypt(const u8 in[16], u8 out[16]) {
        u8 s[4][4];
        for(int r=0;r<4;r++) for(int c=0;c<4;c++) s[r][c]=in[r+4*c];
        addRoundKey(s, 0);
        for(int round=1;round<10;round++) {
            subBytes(s); shiftRows(s); mixCols(s); addRoundKey(s, round);
        }
        subBytes(s); shiftRows(s); addRoundKey(s, 10);
        for(int r=0;r<4;r++) for(int c=0;c<4;c++) out[r+4*c]=s[r][c];
    }
}

class BigInt {
    std::vector<u32> limbs;
    bool neg;
    static const u64 BASE = 1ULL << 32;
    void trim() { while(limbs.size() > 1 && limbs.back() == 0) limbs.pop_back(); }
public:
    BigInt() : limbs(1, 0), neg(false) {}
    BigInt(i64 v) : neg(v < 0) {
        u64 uv = neg ? (u64)(-(v+1))+1 : (u64)v;
        limbs.push_back(uv & 0xFFFFFFFF);
        if(uv >> 32) limbs.push_back(uv >> 32);
    }
    BigInt(const std::string &s) : neg(false) {
        int start = 0;
        if(!s.empty() && s[0] == '-') { neg = true; start = 1; }
        *this = BigInt(0);
        for(int i = start; i < (int)s.size(); i++) {
            *this = *this * BigInt(10) + BigInt(s[i] - '0');
        }
        this->neg = neg;
    }
    bool isZero() const { return limbs.size() == 1 && limbs[0] == 0; }
    BigInt operator+(const BigInt &o) const {
        if(neg != o.neg) {
            BigInt oc = o; oc.neg = !oc.neg;
            return *this - oc;
        }
        BigInt r;
        r.neg = neg;
        r.limbs.clear();
        u64 carry = 0;
        size_t n = MAX(limbs.size(), o.limbs.size());
        for(size_t i = 0; i < n || carry; i++) {
            u64 a = i < limbs.size() ? limbs[i] : 0;
            u64 b = i < o.limbs.size() ? o.limbs[i] : 0;
            u64 s = a + b + carry;
            r.limbs.push_back(s & 0xFFFFFFFF);
            carry = s >> 32;
        }
        r.trim();
        return r;
    }
    BigInt operator*(const BigInt &o) const {
        BigInt r;
        r.limbs.assign(limbs.size() + o.limbs.size(), 0);
        for(size_t i = 0; i < limbs.size(); i++) {
            u64 carry = 0;
            for(size_t j = 0; j < o.limbs.size() || carry; j++) {
                u64 b = j < o.limbs.size() ? o.limbs[j] : 0;
                u64 cur = (u64)limbs[i] * b + r.limbs[i+j] + carry;
                r.limbs[i+j] = cur & 0xFFFFFFFF;
                carry = cur >> 32;
            }
        }
        r.neg = neg != o.neg;
        r.trim();
        if(r.isZero()) r.neg = false;
        return r;
    }
    BigInt operator-(const BigInt &o) const;
    std::string str() const {
        if(isZero()) return "0";
        BigInt tmp = *this; tmp.neg = false;
        std::string s;
        BigInt ten(10);
        while(!tmp.isZero()) {
            BigInt q, rem;
            divmod(tmp, ten, q, rem);
            s += ('0' + (int)rem.limbs[0]);
            tmp = q;
        }
        if(neg) s += '-';
        std::reverse(s.begin(), s.end());
        return s;
    }
    static void divmod(const BigInt &a, const BigInt &b, BigInt &q, BigInt &r);
};

class LRUCache {
    int cap;
    std::list<std::pair<int,int>> lst;
    std::unordered_map<int, std::list<std::pair<int,int>>::iterator> map;
public:
    LRUCache(int c) : cap(c) {}
    int get(int key) {
        auto it = map.find(key);
        if(it == map.end()) return -1;
        lst.splice(lst.begin(), lst, it->second);
        return it->second->second;
    }
    void put(int key, int val) {
        auto it = map.find(key);
        if(it != map.end()) {
            it->second->second = val;
            lst.splice(lst.begin(), lst, it->second);
            return;
        }
        if((int)lst.size() == cap) {
            map.erase(lst.back().first);
            lst.pop_back();
        }
        lst.push_front({key, val});
        map[key] = lst.begin();
    }
};

class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop;
public:
    ThreadPool(size_t n) : stop(false) {
        for(size_t i = 0; i < n; i++) {
            workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [this] { return stop || !tasks.empty(); });
                        if(stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        for(auto &w : workers) w.join();
    }
    template<typename F>
    std::future<typename std::result_of<F()>::type> submit(F &&f) {
        using R = typename std::result_of<F()>::type;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        std::future<R> fut = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mtx);
            tasks.emplace([task] { (*task)(); });
        }
        cv.notify_one();
        return fut;
    }
};

class SpinLock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
public:
    void lock() { while(flag.test_and_set(std::memory_order_acquire)); }
    void unlock() { flag.clear(std::memory_order_release); }
    bool tryLock() { return !flag.test_and_set(std::memory_order_acquire); }
};

template<typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node *next;
        Node(T v) : data(v), next(nullptr) {}
    };
    std::atomic<Node*> head;
public:
    LockFreeStack() : head(nullptr) {}
    void push(T v) {
        Node *n = new Node(v);
        n->next = head.load(std::memory_order_relaxed);
        while(!head.compare_exchange_weak(n->next, n, std::memory_order_release, std::memory_order_relaxed));
    }
    bool pop(T &v) {
        Node *n = head.load(std::memory_order_relaxed);
        while(n && !head.compare_exchange_weak(n, n->next, std::memory_order_acquire, std::memory_order_relaxed));
        if(!n) return false;
        v = n->data;
        delete n;
        return true;
    }
    bool empty() const { return head.load() == nullptr; }
};

template<typename T>
class LockFreeQueue {
    struct Node {
        T data;
        std::atomic<Node*> next;
        Node() : next(nullptr) {}
        Node(T v) : data(v), next(nullptr) {}
    };
    std::atomic<Node*> head, tail;
public:
    LockFreeQueue() {
        Node *dummy = new Node();
        head = tail = dummy;
    }
    void enqueue(T v) {
        Node *n = new Node(v);
        Node *t;
        while(true) {
            t = tail.load(std::memory_order_acquire);
            Node *next = t->next.load(std::memory_order_acquire);
            if(t == tail.load(std::memory_order_acquire)) {
                if(!next) {
                    if(t->next.compare_exchange_weak(next, n, std::memory_order_release)) {
                        tail.compare_exchange_weak(t, n, std::memory_order_release);
                        return;
                    }
                } else {
                    tail.compare_exchange_weak(t, next, std::memory_order_release);
                }
            }
        }
    }
    bool dequeue(T &v) {
        Node *h, *t, *next;
        while(true) {
            h = head.load(std::memory_order_acquire);
            t = tail.load(std::memory_order_acquire);
            next = h->next.load(std::memory_order_acquire);
            if(h == head.load(std::memory_order_acquire)) {
                if(h == t) {
                    if(!next) return false;
                    tail.compare_exchange_weak(t, next, std::memory_order_release);
                } else {
                    v = next->data;
                    if(head.compare_exchange_weak(h, next, std::memory_order_release)) {
                        delete h;
                        return true;
                    }
                }
            }
        }
    }
};

struct RAII {
    std::function<void()> fn;
    RAII(std::function<void()> f) : fn(f) {}
    ~RAII() { if(fn) fn(); }
    RAII(const RAII&) = delete;
    RAII &operator=(const RAII&) = delete;
};

template<typename T>
class Observable {
    T val;
    std::vector<std::function<void(const T&, const T&)>> listeners;
public:
    Observable(T v) : val(v) {}
    void set(T v) {
        T old = val;
        val = v;
        for(auto &l : listeners) l(old, val);
    }
    T get() const { return val; }
    void subscribe(std::function<void(const T&, const T&)> cb) {
        listeners.push_back(std::move(cb));
    }
    T &operator*() { return val; }
};

template<typename... Ts>
struct TypeList {};

template<typename T, typename... Ts>
struct Head { using type = T; };

template<typename T>
struct IsVoid { static const bool value = false; };
template<>
struct IsVoid<void> { static const bool value = true; };

template<typename T>
struct RemoveRef { using type = T; };
template<typename T>
struct RemoveRef<T&> { using type = T; };
template<typename T>
struct RemoveRef<T&&> { using type = T; };

template<bool B, typename T, typename F>
struct Conditional { using type = F; };
template<typename T, typename F>
struct Conditional<true, T, F> { using type = T; };

template<typename T>
class Expected {
    union {
        T val;
        std::string err;
    };
    bool ok;
public:
    Expected(T v) : val(v), ok(true) {}
    Expected(const std::string &e) : err(e), ok(false) {}
    ~Expected() {
        if(ok) val.~T();
        else err.~basic_string();
    }
    bool hasValue() const { return ok; }
    T &value() {
        if(!ok) throw std::runtime_error(err);
        return val;
    }
    const std::string &error() const { return err; }
    explicit operator bool() const { return ok; }
    T valueOr(T def) const { return ok ? val : def; }
};

namespace serialization {
    class BinaryWriter {
        std::vector<u8> buf;
    public:
        void writeU8(u8 v) { buf.push_back(v); }
        void writeU16(u16 v) { buf.push_back(v&0xFF); buf.push_back(v>>8); }
        void writeU32(u32 v) {
            buf.push_back(v&0xFF); buf.push_back((v>>8)&0xFF);
            buf.push_back((v>>16)&0xFF); buf.push_back(v>>24);
        }
        void writeU64(u64 v) {
            for(int i=0;i<8;i++) { buf.push_back(v&0xFF); v>>=8; }
        }
        void writeFloat(f32 v) { u32 u; memcpy(&u, &v, 4); writeU32(u); }
        void writeString(const std::string &s) {
            writeU32(s.size());
            for(char c : s) buf.push_back((u8)c);
        }
        const std::vector<u8> &data() const { return buf; }
        void clear() { buf.clear(); }
    };

    class BinaryReader {
        const u8 *ptr;
        size_t pos, len;
        bool fail;
    public:
        BinaryReader(const u8 *data, size_t n) : ptr(data), pos(0), len(n), fail(false) {}
        u8 readU8() {
            if(pos >= len) { fail=true; return 0; }
            return ptr[pos++];
        }
        u16 readU16() { u16 a=readU8(), b=readU8(); return a|(b<<8); }
        u32 readU32() {
            u32 a=readU8(), b=readU8(), c=readU8(), d=readU8();
            return a|(b<<8)|(c<<16)|(d<<24);
        }
        u64 readU64() {
            u64 r=0;
            for(int i=0;i<8;i++) r|=(u64)readU8()<<(i*8);
            return r;
        }
        f32 readFloat() { u32 u=readU32(); f32 v; memcpy(&v,&u,4); return v; }
        std::string readString() {
            u32 n=readU32(); std::string s(n,'\0');
            for(char &c : s) c=readU8();
            return s;
        }
        bool failed() const { return fail; }
        bool atEnd() const { return pos >= len; }
    };

    std::string toBase64(const u8 *data, size_t len) {
        static const char enc[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        for(size_t i = 0; i < len; i += 3) {
            u32 b = (u32)data[i] << 16;
            if(i+1 < len) b |= (u32)data[i+1] << 8;
            if(i+2 < len) b |= data[i+2];
            out += enc[(b>>18)&0x3F];
            out += enc[(b>>12)&0x3F];
            out += (i+1 < len) ? enc[(b>>6)&0x3F] : '=';
            out += (i+2 < len) ? enc[b&0x3F] : '=';
        }
        return out;
    }

    std::vector<u8> fromBase64(const std::string &s) {
        static const int dec[256] = {
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
            52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
            -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
            15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
            -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
            41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        };
        std::vector<u8> out;
        int state = 0, val = 0;
        for(char c : s) {
            if(c == '=') break;
            int v = dec[(u8)c];
            if(v < 0) continue;
            val = (val << 6) | v;
            if(++state == 4) {
                out.push_back((val>>16)&0xFF);
                out.push_back((val>>8)&0xFF);
                out.push_back(val&0xFF);
                state = val = 0;
            }
        }
        if(state == 3) { out.push_back((val>>10)&0xFF); out.push_back((val>>2)&0xFF); }
        else if(state == 2) { out.push_back((val>>4)&0xFF); }
        return out;
    }
}

namespace net {
    u16 checksum(const u8 *data, size_t len) {
        u32 sum = 0;
        for(size_t i = 0; i + 1 < len; i += 2)
            sum += ((u16)data[i] << 8) | data[i+1];
        if(len & 1) sum += (u16)data[len-1] << 8;
        while(sum >> 16) sum = (sum & 0xFFFF) + (sum >> 16);
        return ~sum;
    }

    struct IPv4Header {
        u8 ihl : 4, ver : 4;
        u8 dscp;
        u16 totalLen;
        u16 id;
        u16 flags : 3, fragOff : 13;
        u8 ttl;
        u8 proto;
        u16 checksum;
        u32 srcIP, dstIP;
    };

    struct TCPHeader {
        u16 srcPort, dstPort;
        u32 seq, ack;
        u8 dataOff : 4, rsvd : 4;
        u8 flags;
        u16 windowSz;
        u16 checksum;
        u16 urgPtr;
    };

    std::string ipToStr(u32 ip) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d.%d.%d.%d",
            (ip>>24)&0xFF, (ip>>16)&0xFF, (ip>>8)&0xFF, ip&0xFF);
        return buf;
    }

    u32 strToIP(const std::string &s) {
        u32 a,b,c,d;
        sscanf(s.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d);
        return (a<<24)|(b<<16)|(c<<8)|d;
    }

    bool isPrivateIP(u32 ip) {
        if((ip & 0xFF000000) == 0x0A000000) return true;
        if((ip & 0xFFF00000) == 0xAC100000) return true;
        if((ip & 0xFFFF0000) == 0xC0A80000) return true;
        return false;
    }

    class URL {
    public:
        std::string scheme, host, path, query, fragment;
        int port;
        URL() : port(-1) {}
        static URL parse(const std::string &url) {
            URL u;
            size_t pos = url.find("://");
            if(pos != std::string::npos) {
                u.scheme = url.substr(0, pos);
                pos += 3;
            } else {
                pos = 0;
            }
            size_t hostEnd = url.find_first_of("/?#", pos);
            std::string hostPort = url.substr(pos, hostEnd == std::string::npos ? std::string::npos : hostEnd - pos);
            size_t colon = hostPort.rfind(':');
            if(colon != std::string::npos) {
                u.host = hostPort.substr(0, colon);
                u.port = std::stoi(hostPort.substr(colon+1));
            } else {
                u.host = hostPort;
            }
            if(hostEnd == std::string::npos) return u;
            size_t pathEnd = url.find_first_of("?#", hostEnd);
            u.path = url.substr(hostEnd, pathEnd == std::string::npos ? std::string::npos : pathEnd - hostEnd);
            if(pathEnd != std::string::npos && url[pathEnd] == '?') {
                size_t fragStart = url.find('#', pathEnd+1);
                u.query = url.substr(pathEnd+1, fragStart == std::string::npos ? std::string::npos : fragStart - pathEnd - 1);
                if(fragStart != std::string::npos) u.fragment = url.substr(fragStart+1);
            } else if(pathEnd != std::string::npos && url[pathEnd] == '#') {
                u.fragment = url.substr(pathEnd+1);
            }
            return u;
        }
        std::string str() const {
            std::string r;
            if(!scheme.empty()) r += scheme + "://";
            r += host;
            if(port >= 0) r += ":" + std::to_string(port);
            r += path;
            if(!query.empty()) r += "?" + query;
            if(!fragment.empty()) r += "#" + fragment;
            return r;
        }
    };

    class HttpRequest {
    public:
        std::string method, path, version;
        std::unordered_map<std::string, std::string> headers;
        std::string body;
        static HttpRequest parse(const std::string &raw) {
            HttpRequest r;
            std::istringstream ss(raw);
            std::string line;
            std::getline(ss, line);
            if(!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ls(line);
            ls >> r.method >> r.path >> r.version;
            while(std::getline(ss, line)) {
                if(!line.empty() && line.back() == '\r') line.pop_back();
                if(line.empty()) break;
                size_t colon = line.find(':');
                if(colon != std::string::npos) {
                    std::string k = line.substr(0, colon);
                    std::string v = line.substr(colon+1);
                    while(!v.empty() && v.front() == ' ') v = v.substr(1);
                    r.headers[k] = v;
                }
            }
            std::string rest;
            while(std::getline(ss, line)) rest += line + "\n";
            r.body = rest;
            return r;
        }
        std::string serialize() const {
            std::string r = method + " " + path + " " + version + "\r\n";
            for(auto &[k,v] : headers) r += k + ": " + v + "\r\n";
            r += "\r\n" + body;
            return r;
        }
    };
}

namespace json {
    enum class Type { Null, Bool, Int, Float, String, Array, Object };

    struct Value {
        Type type;
        bool b;
        i64 i;
        f64 f;
        std::string s;
        std::vector<Value> arr;
        std::map<std::string, Value> obj;

        Value() : type(Type::Null), b(false), i(0), f(0.0) {}
        Value(bool v) : type(Type::Bool), b(v), i(0), f(0.0) {}
        Value(i64 v) : type(Type::Int), b(false), i(v), f(0.0) {}
        Value(f64 v) : type(Type::Float), b(false), i(0), f(v) {}
        Value(const std::string &v) : type(Type::String), b(false), i(0), f(0.0), s(v) {}
        Value(const std::vector<Value> &v) : type(Type::Array), b(false), i(0), f(0.0), arr(v) {}
        Value(const std::map<std::string,Value> &v) : type(Type::Object), b(false), i(0), f(0.0), obj(v) {}

        bool isNull() const { return type == Type::Null; }
        Value &operator[](const std::string &k) { return obj[k]; }
        Value &operator[](size_t idx) { return arr[idx]; }
        const Value &operator[](const std::string &k) const { return obj.at(k); }

        std::string dump(int indent = 0) const {
            std::string pad(indent * 2, ' ');
            std::string childPad((indent+1) * 2, ' ');
            switch(type) {
            case Type::Null: return "null";
            case Type::Bool: return b ? "true" : "false";
            case Type::Int: return std::to_string(i);
            case Type::Float: { char buf[32]; snprintf(buf, 32, "%g", f); return buf; }
            case Type::String: {
                std::string r = "\"";
                for(char c : s) {
                    if(c == '"') r += "\\\"";
                    else if(c == '\\') r += "\\\\";
                    else if(c == '\n') r += "\\n";
                    else if(c == '\r') r += "\\r";
                    else if(c == '\t') r += "\\t";
                    else r += c;
                }
                return r + "\"";
            }
            case Type::Array: {
                if(arr.empty()) return "[]";
                std::string r = "[\n";
                for(size_t i=0;i<arr.size();i++) {
                    r += childPad + arr[i].dump(indent+1);
                    if(i+1 < arr.size()) r += ",";
                    r += "\n";
                }
                return r + pad + "]";
            }
            case Type::Object: {
                if(obj.empty()) return "{}";
                std::string r = "{\n";
                size_t n = 0;
                for(auto &[k,v] : obj) {
                    r += childPad + "\"" + k + "\": " + v.dump(indent+1);
                    if(++n < obj.size()) r += ",";
                    r += "\n";
                }
                return r + pad + "}";
            }
            }
            return "";
        }
    };

    class Parser {
        const std::string &src;
        size_t pos;
        void skip() { while(pos < src.size() && isspace(src[pos])) pos++; }
        Value parseVal();
        Value parseStr() {
            pos++;
            std::string r;
            while(pos < src.size() && src[pos] != '"') {
                if(src[pos] == '\\') {
                    pos++;
                    switch(src[pos]) {
                    case '"': r+='"'; break; case '\\': r+='\\'; break;
                    case 'n': r+='\n'; break; case 'r': r+='\r'; break;
                    case 't': r+='\t'; break; default: r+=src[pos];
                    }
                } else r += src[pos];
                pos++;
            }
            pos++;
            return Value(r);
        }
        Value parseNum() {
            size_t start = pos;
            bool isFloat = false;
            if(src[pos] == '-') pos++;
            while(pos < src.size() && isdigit(src[pos])) pos++;
            if(pos < src.size() && src[pos] == '.') { isFloat = true; pos++; while(pos < src.size() && isdigit(src[pos])) pos++; }
            if(pos < src.size() && (src[pos] == 'e' || src[pos] == 'E')) { isFloat = true; pos++; if(pos < src.size() && (src[pos]=='+' || src[pos]=='-')) pos++; while(pos < src.size() && isdigit(src[pos])) pos++; }
            std::string t = src.substr(start, pos-start);
            if(isFloat) return Value(std::stod(t));
            return Value((i64)std::stoll(t));
        }
        Value parseArr() {
            pos++;
            std::vector<Value> arr;
            skip();
            if(src[pos] == ']') { pos++; return Value(arr); }
            while(true) {
                skip();
                arr.push_back(parseVal());
                skip();
                if(src[pos] == ']') { pos++; break; }
                pos++;
            }
            return Value(arr);
        }
        Value parseObj() {
            pos++;
            std::map<std::string,Value> obj;
            skip();
            if(src[pos] == '}') { pos++; return Value(obj); }
            while(true) {
                skip();
                Value k = parseStr();
                skip(); pos++;
                skip();
                Value v = parseVal();
                obj[k.s] = v;
                skip();
                if(src[pos] == '}') { pos++; break; }
                pos++;
            }
            return Value(obj);
        }
    public:
        Parser(const std::string &s) : src(s), pos(0) {}
        Value parse() { skip(); return parseVal(); }
    };

    Value Parser::parseVal() {
        skip();
        if(pos >= src.size()) return Value();
        char c = src[pos];
        if(c == '"') return parseStr();
        if(c == '[') return parseArr();
        if(c == '{') return parseObj();
        if(c == 't') { pos += 4; return Value(true); }
        if(c == 'f') { pos += 5; return Value(false); }
        if(c == 'n') { pos += 4; return Value(); }
        if(c == '-' || isdigit(c)) return parseNum();
        return Value();
    }

    Value parse(const std::string &s) { return Parser(s).parse(); }
}

class EventLoop {
    using Handler = std::function<void()>;
    std::queue<Handler> pending;
    std::map<int, Handler> timers;
    std::mutex mtx;
    std::atomic<bool> running;
    int nextTimer;
public:
    EventLoop() : running(false), nextTimer(0) {}
    void post(Handler h) {
        std::lock_guard<std::mutex> lock(mtx);
        pending.push(std::move(h));
    }
    int setInterval(int ms, Handler h) {
        int id = nextTimer++;
        timers[id] = std::move(h);
        return id;
    }
    void clearInterval(int id) { timers.erase(id); }
    void run(int maxIter = 1000) {
        running = true;
        int iter = 0;
        while(running && iter++ < maxIter) {
            Handler h;
            {
                std::lock_guard<std::mutex> lock(mtx);
                if(pending.empty()) break;
                h = std::move(pending.front());
                pending.pop();
            }
            h();
        }
    }
    void stop() { running = false; }
};

template<typename State, typename Input>
class StateMachine {
public:
    using TransitionFn = std::function<State(State, Input)>;
    using ActionFn = std::function<void(State, Input, State)>;
private:
    State cur;
    TransitionFn transitionFn;
    std::vector<ActionFn> actions;
public:
    StateMachine(State init, TransitionFn fn) : cur(init), transitionFn(fn) {}
    void onTransition(ActionFn fn) { actions.push_back(std::move(fn)); }
    State process(Input in) {
        State next = transitionFn(cur, in);
        for(auto &a : actions) a(cur, in, next);
        cur = next;
        return cur;
    }
    State state() const { return cur; }
};

class Regex {
    std::regex re;
public:
    Regex(const std::string &pattern) : re(pattern) {}
    bool match(const std::string &s) { return std::regex_match(s, re); }
    bool search(const std::string &s) { return std::regex_search(s, re); }
    std::vector<std::string> findAll(const std::string &s) {
        std::vector<std::string> res;
        std::sregex_iterator it(s.begin(), s.end(), re), end;
        for(; it != end; ++it) res.push_back((*it)[0].str());
        return res;
    }
    std::string replace(const std::string &s, const std::string &repl) {
        return std::regex_replace(s, re, repl);
    }
};

class CSV {
public:
    static std::vector<std::vector<std::string>> parse(const std::string &data) {
        std::vector<std::vector<std::string>> rows;
        std::istringstream ss(data);
        std::string line;
        while(std::getline(ss, line)) {
            if(!line.empty() && line.back() == '\r') line.pop_back();
            std::vector<std::string> row;
            std::string field;
            bool inQuote = false;
            for(size_t i = 0; i < line.size(); i++) {
                char c = line[i];
                if(c == '"') {
                    if(inQuote && i+1 < line.size() && line[i+1] == '"') { field += '"'; i++; }
                    else inQuote = !inQuote;
                } else if(c == ',' && !inQuote) {
                    row.push_back(field); field.clear();
                } else {
                    field += c;
                }
            }
            row.push_back(field);
            rows.push_back(row);
        }
        return rows;
    }
    static std::string serialize(const std::vector<std::vector<std::string>> &rows) {
        std::string out;
        for(auto &row : rows) {
            for(size_t i=0; i<row.size(); i++) {
                if(i) out += ',';
                bool needQuote = row[i].find_first_of(",\"\n\r") != std::string::npos;
                if(needQuote) {
                    out += '"';
                    for(char c : row[i]) { if(c=='"') out += '"'; out += c; }
                    out += '"';
                } else {
                    out += row[i];
                }
            }
            out += '\n';
        }
        return out;
    }
};

template<size_t N>
class Bitset {
    u64 words[(N+63)/64];
    static const size_t nw = (N+63)/64;
public:
    Bitset() { memset(words, 0, sizeof(words)); }
    void set(size_t i) { words[i/64] |= (1ULL << (i%64)); }
    void clear(size_t i) { words[i/64] &= ~(1ULL << (i%64)); }
    bool test(size_t i) const { return (words[i/64] >> (i%64)) & 1; }
    void flip(size_t i) { words[i/64] ^= (1ULL << (i%64)); }
    Bitset operator&(const Bitset &o) const {
        Bitset r;
        for(size_t i=0;i<nw;i++) r.words[i]=words[i]&o.words[i];
        return r;
    }
    Bitset operator|(const Bitset &o) const {
        Bitset r;
        for(size_t i=0;i<nw;i++) r.words[i]=words[i]|o.words[i];
        return r;
    }
    Bitset operator^(const Bitset &o) const {
        Bitset r;
        for(size_t i=0;i<nw;i++) r.words[i]=words[i]^o.words[i];
        return r;
    }
    size_t count() const {
        size_t n = 0;
        for(size_t i=0;i<nw;i++) n += __builtin_popcountll(words[i]);
        return n;
    }
    bool any() const {
        for(size_t i=0;i<nw;i++) if(words[i]) return true;
        return false;
    }
    bool none() const { return !any(); }
};

namespace math {
    u64 gcd(u64 a, u64 b) { while(b) { a%=b; std::swap(a,b); } return a; }
    u64 lcm(u64 a, u64 b) { return a / gcd(a,b) * b; }
    bool isPrime(u64 n) {
        if(n < 2) return false;
        if(n == 2) return true;
        if(n % 2 == 0) return false;
        for(u64 i = 3; i*i <= n; i += 2)
            if(n % i == 0) return false;
        return true;
    }
    std::vector<u64> sieve(u64 n) {
        std::vector<bool> notPrime(n+1, false);
        std::vector<u64> primes;
        for(u64 i=2;i<=n;i++) {
            if(!notPrime[i]) {
                primes.push_back(i);
                for(u64 j=i*i;j<=n;j+=i) notPrime[j]=true;
            }
        }
        return primes;
    }
    u64 modPow(u64 base, u64 exp, u64 mod) {
        u64 r = 1;
        base %= mod;
        while(exp > 0) {
            if(exp & 1) r = (__uint128_t)r * base % mod;
            base = (__uint128_t)base * base % mod;
            exp >>= 1;
        }
        return r;
    }
    bool millerRabin(u64 n, u64 a) {
        if(n % a == 0) return n == a;
        u64 d = n-1; int r = 0;
        while(!(d&1)) { d>>=1; r++; }
        u64 x = modPow(a, d, n);
        if(x == 1 || x == n-1) return true;
        for(int i=0;i<r-1;i++) {
            x = (__uint128_t)x*x%n;
            if(x == n-1) return true;
        }
        return false;
    }
    bool isPrimeFast(u64 n) {
        if(n < 2) return false;
        for(u64 a : {2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL,29ULL,31ULL,37ULL})
            if(n == a) return true;
            else if(!millerRabin(n, a)) return false;
        return true;
    }
    u64 pollardRho(u64 n) {
        if(n % 2 == 0) return 2;
        u64 x = rand() % (n-2) + 2, y = x;
        u64 c = rand() % (n-1) + 1, d = 1;
        while(d == 1) {
            x = ((__uint128_t)x*x + c) % n;
            y = ((__uint128_t)y*y + c) % n;
            y = ((__uint128_t)y*y + c) % n;
            d = gcd(x > y ? x-y : y-x, n);
        }
        return d;
    }
    i64 extGcd(i64 a, i64 b, i64 &x, i64 &y) {
        if(b == 0) { x=1; y=0; return a; }
        i64 x1, y1;
        i64 g = extGcd(b, a%b, x1, y1);
        x = y1; y = x1 - (a/b)*y1;
        return g;
    }
    i64 modInverse(i64 a, i64 mod) {
        i64 x, y;
        i64 g = extGcd(a, mod, x, y);
        if(g != 1) return -1;
        return (x % mod + mod) % mod;
    }
    std::vector<int> nthPermutation(int n, int k) {
        std::vector<int> elems(n);
        std::iota(elems.begin(), elems.end(), 0);
        std::vector<int> perm;
        i64 f = 1;
        for(int i=2;i<n;i++) f *= i;
        k--;
        for(int i=n;i>0;i--) {
            int idx = k / f;
            perm.push_back(elems[idx]);
            elems.erase(elems.begin() + idx);
            if(i > 1) f /= (i-1);
        }
        return perm;
    }
}

class MemMapFile {
    u8 *data;
    size_t sz;
public:
    MemMapFile(const char *path) : data(nullptr), sz(0) {
        FILE *f = fopen(path, "rb");
        if(!f) return;
        fseek(f, 0, SEEK_END);
        sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        data = new u8[sz];
        fread(data, 1, sz, f);
        fclose(f);
    }
    ~MemMapFile() { delete[] data; }
    u8 *ptr() { return data; }
    size_t size() const { return sz; }
    bool valid() const { return data != nullptr; }
};

class Logger {
public:
    enum Level { Debug, Info, Warn, Error };
private:
    Level minLevel;
    std::vector<std::function<void(Level, const std::string&)>> sinks;
    std::mutex mtx;
    static std::string levelStr(Level l) {
        switch(l) { case Debug: return "DEBUG"; case Info: return "INFO"; case Warn: return "WARN"; case Error: return "ERROR"; }
        return "?";
    }
public:
    Logger(Level min = Info) : minLevel(min) {}
    void addSink(std::function<void(Level, const std::string&)> fn) {
        std::lock_guard<std::mutex> lock(mtx);
        sinks.push_back(std::move(fn));
    }
    void log(Level l, const std::string &msg) {
        if(l < minLevel) return;
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        char timeBuf[32];
        strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", localtime(&t));
        std::string line = std::string(timeBuf) + " [" + levelStr(l) + "] " + msg;
        std::lock_guard<std::mutex> lock(mtx);
        for(auto &s : sinks) s(l, line);
    }
    void debug(const std::string &msg) { log(Debug, msg); }
    void info(const std::string &msg) { log(Info, msg); }
    void warn(const std::string &msg) { log(Warn, msg); }
    void error(const std::string &msg) { log(Error, msg); }
};

template<typename... Args>
class Signal {
    std::vector<std::function<void(Args...)>> slots;
public:
    void connect(std::function<void(Args...)> fn) { slots.push_back(std::move(fn)); }
    void emit(Args... args) { for(auto &s : slots) s(args...); }
    size_t count() const { return slots.size(); }
    void disconnectAll() { slots.clear(); }
};

class IniParser {
public:
    std::map<std::string, std::map<std::string,std::string>> sections;
    bool parse(const std::string &data) {
        std::istringstream ss(data);
        std::string line, section;
        while(std::getline(ss, line)) {
            if(!line.empty() && line.back() == '\r') line.pop_back();
            size_t start = line.find_first_not_of(" \t");
            if(start == std::string::npos) continue;
            line = line.substr(start);
            if(line[0] == ';' || line[0] == '#') continue;
            if(line[0] == '[') {
                size_t end = line.find(']');
                if(end == std::string::npos) return false;
                section = line.substr(1, end-1);
            } else {
                size_t eq = line.find('=');
                if(eq == std::string::npos) continue;
                std::string k = line.substr(0, eq), v = line.substr(eq+1);
                while(!k.empty() && isspace(k.back())) k.pop_back();
                while(!v.empty() && isspace(v.front())) v = v.substr(1);
                sections[section][k] = v;
            }
        }
        return true;
    }
    std::string get(const std::string &section, const std::string &key, const std::string &def = "") const {
        auto it = sections.find(section);
        if(it == sections.end()) return def;
        auto jt = it->second.find(key);
        return jt == it->second.end() ? def : jt->second;
    }
};

struct Fiber {
    using Fn = std::function<void()>;
    std::thread t;
    std::mutex mtx;
    std::condition_variable cv;
    bool active, done;
    Fn fn;
    Fiber(Fn f) : active(false), done(false), fn(f) {
        t = std::thread([this] {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this] { return active; });
            fn();
            done = true;
            cv.notify_one();
        });
    }
    ~Fiber() { if(t.joinable()) t.join(); }
    void resume() {
        std::unique_lock<std::mutex> lock(mtx);
        active = true;
        cv.notify_one();
        cv.wait(lock, [this] { return done; });
    }
};

class ByteBuffer {
    std::vector<u8> data;
    size_t rpos, wpos;
public:
    ByteBuffer() : rpos(0), wpos(0) {}
    ByteBuffer(size_t cap) : data(cap), rpos(0), wpos(0) {}
    void writeU8(u8 v) {
        if(wpos >= data.size()) data.push_back(v);
        else data[wpos] = v;
        wpos++;
    }
    void writeBytes(const u8 *src, size_t n) {
        for(size_t i=0;i<n;i++) writeU8(src[i]);
    }
    u8 readU8() { return rpos < data.size() ? data[rpos++] : 0; }
    void readBytes(u8 *dst, size_t n) { for(size_t i=0;i<n;i++) dst[i]=readU8(); }
    size_t readAvailable() const { return data.size() - rpos; }
    void reset() { rpos = wpos = 0; }
    const u8 *ptr() const { return data.data(); }
    size_t size() const { return wpos; }
};

namespace compress {
    std::vector<u8> rleEncode(const u8 *data, size_t len) {
        std::vector<u8> out;
        for(size_t i=0;i<len;) {
            u8 v = data[i];
            size_t cnt = 1;
            while(i+cnt < len && data[i+cnt] == v && cnt < 255) cnt++;
            out.push_back((u8)cnt); out.push_back(v);
            i += cnt;
        }
        return out;
    }
    std::vector<u8> rleDecode(const u8 *data, size_t len) {
        std::vector<u8> out;
        for(size_t i=0;i+1<len;i+=2) {
            u8 cnt=data[i], v=data[i+1];
            for(u8 j=0;j<cnt;j++) out.push_back(v);
        }
        return out;
    }
    std::vector<u8> lz77Encode(const u8 *data, size_t len) {
        std::vector<u8> out;
        size_t i = 0;
        while(i < len) {
            int bestLen=0, bestOff=0;
            for(int off=1;off<=MIN(i,255);off++) {
                int ml=0;
                while(ml<255 && i+ml<len && data[i-off+ml]==data[i+ml]) ml++;
                if(ml>bestLen) { bestLen=ml; bestOff=off; }
            }
            if(bestLen >= 3) {
                out.push_back(0xFF); out.push_back((u8)bestOff); out.push_back((u8)bestLen);
                i += bestLen;
            } else {
                out.push_back(data[i++]);
            }
        }
        return out;
    }
}

template<typename T>
class Matrix {
    std::vector<std::vector<T>> data;
    size_t rows, cols;
public:
    Matrix(size_t r, size_t c, T init = T{}) : data(r, std::vector<T>(c, init)), rows(r), cols(c) {}
    T &at(size_t r, size_t c) { return data[r][c]; }
    const T &at(size_t r, size_t c) const { return data[r][c]; }
    Matrix operator+(const Matrix &o) const {
        Matrix r(rows, cols);
        for(size_t i=0;i<rows;i++) for(size_t j=0;j<cols;j++) r.data[i][j]=data[i][j]+o.data[i][j];
        return r;
    }
    Matrix operator*(const Matrix &o) const {
        Matrix r(rows, o.cols);
        for(size_t i=0;i<rows;i++) for(size_t j=0;j<o.cols;j++) for(size_t k=0;k<cols;k++) r.data[i][j]+=data[i][k]*o.data[k][j];
        return r;
    }
    Matrix transpose() const {
        Matrix r(cols, rows);
        for(size_t i=0;i<rows;i++) for(size_t j=0;j<cols;j++) r.data[j][i]=data[i][j];
        return r;
    }
    std::vector<T> gaussElim(std::vector<T> b) {
        size_t n = rows;
        std::vector<std::vector<T>> a(data);
        for(size_t i=0;i<n;i++) a[i].push_back(b[i]);
        for(size_t col=0,row=0;col<n && row<n;col++) {
            size_t pivot = row;
            for(size_t i=row+1;i<n;i++) if(abs(a[i][col])>abs(a[pivot][col])) pivot=i;
            if(a[pivot][col]==0) continue;
            std::swap(a[row], a[pivot]);
            T div=a[row][col];
            for(size_t j=col;j<=n;j++) a[row][j]/=div;
            for(size_t i=0;i<n;i++) if(i!=row) {
                T f=a[i][col];
                for(size_t j=col;j<=n;j++) a[i][j]-=f*a[row][j];
            }
            row++;
        }
        std::vector<T> x(n);
        for(size_t i=0;i<n;i++) x[i]=a[i][n];
        return x;
    }
    size_t numRows() const { return rows; }
    size_t numCols() const { return cols; }
};

namespace signals {
    std::vector<f64> fft(std::vector<f64> x) {
        size_t n = x.size();
        if(n <= 1) return x;
        std::vector<f64> even, odd;
        for(size_t i=0;i<n;i+=2) { even.push_back(x[i]); if(i+1<n) odd.push_back(x[i+1]); }
        auto E = fft(even), O = fft(odd);
        std::vector<f64> r(n);
        for(size_t k=0;k<n/2;k++) {
            f64 t = cos(-2.0*M_PI*k/n) * O[k];
            r[k] = E[k] + t;
            r[k+n/2] = E[k] - t;
        }
        return r;
    }
    std::vector<f64> convolve(const std::vector<f64> &a, const std::vector<f64> &b) {
        size_t n = a.size() + b.size() - 1;
        std::vector<f64> r(n, 0);
        for(size_t i=0;i<a.size();i++) for(size_t j=0;j<b.size();j++) r[i+j]+=a[i]*b[j];
        return r;
    }
    f64 movingAvg(const std::vector<f64> &data, size_t window, size_t pos) {
        size_t start = pos >= window ? pos-window+1 : 0;
        f64 sum = 0;
        for(size_t i=start;i<=pos&&i<data.size();i++) sum+=data[i];
        return sum / (pos - start + 1);
    }
}

struct RGB {
    u8 r, g, b;
    RGB(u8 r, u8 g, u8 b) : r(r), g(g), b(b) {}
    RGB blend(const RGB &o, f32 t) const {
        return {(u8)(r*(1-t)+o.r*t), (u8)(g*(1-t)+o.g*t), (u8)(b*(1-t)+o.b*t)};
    }
    u32 toHex() const { return ((u32)r<<16)|((u32)g<<8)|b; }
    static RGB fromHex(u32 hex) { return {(u8)(hex>>16), (u8)(hex>>8), (u8)hex}; }
    f32 luminance() const { return 0.299f*r/255 + 0.587f*g/255 + 0.114f*b/255; }
};

class Image {
    std::vector<RGB> pixels;
    int w, h;
public:
    Image(int w, int h, RGB fill = {0,0,0}) : pixels(w*h, fill), w(w), h(h) {}
    RGB &at(int x, int y) { return pixels[y*w+x]; }
    void drawLine(int x0, int y0, int x1, int y1, RGB c) {
        int dx=abs(x1-x0), sx=x0<x1?1:-1;
        int dy=-abs(y1-y0), sy=y0<y1?1:-1;
        int err=dx+dy;
        while(true) {
            if(x0>=0&&x0<w&&y0>=0&&y0<h) at(x0,y0)=c;
            if(x0==x1&&y0==y1) break;
            int e2=2*err;
            if(e2>=dy) { err+=dy; x0+=sx; }
            if(e2<=dx) { err+=dx; y0+=sy; }
        }
    }
    void fillRect(int x, int y, int bw, int bh, RGB c) {
        for(int j=y;j<y+bh&&j<h;j++) for(int i=x;i<x+bw&&i<w;i++) at(i,j)=c;
    }
    void drawCircle(int cx, int cy, int r, RGB c) {
        int x=0, y=r, d=3-2*r;
        while(y>=x) {
            auto put = [&](int px, int py) { if(px>=0&&px<w&&py>=0&&py<h) at(px,py)=c; };
            put(cx+x,cy+y); put(cx-x,cy+y); put(cx+x,cy-y); put(cx-x,cy-y);
            put(cx+y,cy+x); put(cx-y,cy+x); put(cx+y,cy-x); put(cx-y,cy-x);
            if(d<0) d+=4*x+6; else { d+=4*(x-y)+10; y--; }
            x++;
        }
    }
    void gaussianBlur(int radius) {
        std::vector<f32> kernel;
        f32 sigma = radius / 3.0f, sum = 0;
        for(int i=-radius;i<=radius;i++) {
            f32 v = expf(-(f32)(i*i)/(2*sigma*sigma));
            kernel.push_back(v); sum+=v;
        }
        for(auto &v : kernel) v/=sum;
        std::vector<RGB> tmp(pixels);
        for(int y=0;y<h;y++) for(int x=0;x<w;x++) {
            f32 r=0,g=0,b=0;
            for(int k=-radius;k<=radius;k++) {
                int nx = CLAMP(x+k, 0, w-1);
                f32 w_ = kernel[k+radius];
                r+=tmp[y*w+nx].r*w_; g+=tmp[y*w+nx].g*w_; b+=tmp[y*w+nx].b*w_;
            }
            pixels[y*w+x]={u8(r),u8(g),u8(b)};
        }
        tmp = pixels;
        for(int y=0;y<h;y++) for(int x=0;x<w;x++) {
            f32 r=0,g=0,b=0;
            for(int k=-radius;k<=radius;k++) {
                int ny = CLAMP(y+k, 0, h-1);
                f32 w_ = kernel[k+radius];
                r+=tmp[ny*w+x].r*w_; g+=tmp[ny*w+x].g*w_; b+=tmp[ny*w+x].b*w_;
            }
            pixels[y*w+x]={u8(r),u8(g),u8(b)};
        }
    }
    int width() const { return w; }
    int height() const { return h; }
};

class CLI {
    std::map<std::string, std::string> flags;
    std::vector<std::string> positional;
public:
    bool parse(int argc, char **argv) {
        for(int i=1;i<argc;i++) {
            std::string arg(argv[i]);
            if(arg.size()>=2 && arg[0]=='-' && arg[1]=='-') {
                size_t eq = arg.find('=', 2);
                if(eq != std::string::npos) flags[arg.substr(2,eq-2)] = arg.substr(eq+1);
                else if(i+1<argc && argv[i+1][0]!='-') flags[arg.substr(2)] = argv[++i];
                else flags[arg.substr(2)] = "1";
            } else if(arg[0]=='-') {
                if(i+1<argc && argv[i+1][0]!='-') flags[arg.substr(1)] = argv[++i];
                else flags[arg.substr(1)] = "1";
            } else {
                positional.push_back(arg);
            }
        }
        return true;
    }
    std::string get(const std::string &name, const std::string &def="") const {
        auto it = flags.find(name);
        return it==flags.end() ? def : it->second;
    }
    bool has(const std::string &name) const { return flags.count(name)>0; }
    const std::vector<std::string> &args() const { return positional; }
    int getInt(const std::string &name, int def=0) const {
        auto it = flags.find(name);
        return it==flags.end() ? def : std::stoi(it->second);
    }
};

class ObjectPool {
    struct Header { Header *next; };
    u8 *slab;
    Header *free_list;
    size_t objSz, count;
public:
    ObjectPool(size_t objSize, size_t maxObjs) : objSz(ALIGN(objSize, 8)), count(maxObjs) {
        slab = new u8[objSz * maxObjs];
        free_list = nullptr;
        for(size_t i=0;i<maxObjs;i++) {
            Header *h = reinterpret_cast<Header*>(slab + i*objSz);
            h->next = free_list;
            free_list = h;
        }
    }
    ~ObjectPool() { delete[] slab; }
    void *alloc() {
        if(!free_list) return nullptr;
        void *p = free_list;
        free_list = free_list->next;
        return p;
    }
    void free(void *p) {
        Header *h = reinterpret_cast<Header*>(p);
        h->next = free_list;
        free_list = h;
    }
};

int crcTable[256];
bool crcTableInit = false;
void initCRC() {
    for(int i=0;i<256;i++) {
        int c=i;
        for(int j=0;j<8;j++) c = (c>>1)^((c&1)?0xEDB88320:0);
        crcTable[i]=c;
    }
    crcTableInit=true;
}
u32 crc32(const u8 *data, size_t len) {
    if(!crcTableInit) initCRC();
    u32 crc = 0xFFFFFFFF;
    for(size_t i=0;i<len;i++) crc=(crc>>8)^crcTable[(crc^data[i])&0xFF];
    return ~crc;
}

u32 fnv1a(const u8 *data, size_t len) {
    u32 h = 2166136261u;
    for(size_t i=0;i<len;i++) { h^=data[i]; h*=16777619u; }
    return h;
}

u64 xxhash64(const u8 *data, size_t len) {
    auto rol64 = [](u64 x, int n) -> u64 { return (x<<n)|(x>>(64-n)); };
    const u64 P1=11400714785074694791ULL, P2=14029467366897019727ULL, P3=1609587929392839161ULL, P4=9650029242287828579ULL, P5=2870177450012600261ULL;
    u64 h;
    const u8 *p = data, *end = data+len;
    if(len >= 32) {
        u64 v1=P1+P2, v2=P2, v3=0, v4=-P1;
        while(p+32<=end) {
            u64 a; memcpy(&a,p,8); v1=(rol64((v1+a*P2),31))*P1; p+=8;
            memcpy(&a,p,8); v2=(rol64((v2+a*P2),31))*P1; p+=8;
            memcpy(&a,p,8); v3=(rol64((v3+a*P2),31))*P1; p+=8;
            memcpy(&a,p,8); v4=(rol64((v4+a*P2),31))*P1; p+=8;
        }
        h=rol64(v1,1)+rol64(v2,7)+rol64(v3,12)+rol64(v4,18);
    } else {
        h = P5;
    }
    h += len;
    while(p+4<=end) { u32 a; memcpy(&a,p,4); h=rol64(h^((u64)a*P1),23)*P2+P3; p+=4; }
    while(p<end) { h=rol64(h^(*p*P5),11)*P1; p++; }
    h^=h>>33; h*=P2; h^=h>>29; h*=P3; h^=h>>32;
    return h;
}

u32 rotl32(u32 x, int n) { return (x<<n)|(x>>(32-n)); }

u32 murmur3(const u8 *data, size_t len, u32 seed) {
    const u32 c1=0xcc9e2d51, c2=0x1b873593;
    u32 h = seed;
    const u32 *blocks = (const u32*)data;
    size_t nblocks = len/4;
    for(size_t i=0;i<nblocks;i++) {
        u32 k; memcpy(&k,blocks+i,4);
        k*=c1; k=rotl32(k,15); k*=c2;
        h^=k; h=rotl32(h,13); h=h*5+0xe6546b64;
    }
    const u8 *tail = data + nblocks*4;
    u32 k=0;
    switch(len&3) {
    case 3: k^=(u32)tail[2]<<16; [[fallthrough]];
    case 2: k^=(u32)tail[1]<<8; [[fallthrough]];
    case 1: k^=tail[0]; k*=c1; k=rotl32(k,15); k*=c2; h^=k;
    }
    h^=len; h^=h>>16; h*=0x85ebca6b; h^=h>>13; h*=0xc2b2ae35; h^=h>>16;
    return h;
}

class BloomFilter {
    std::vector<u64> bits;
    int nhash;
    size_t sz;
    void setBit(size_t i) { bits[i/64] |= 1ULL<<(i%64); }
    bool testBit(size_t i) const { return (bits[i/64]>>(i%64))&1; }
public:
    BloomFilter(size_t size, int numHash) : bits((size+63)/64, 0), nhash(numHash), sz(size) {}
    void add(const u8 *data, size_t len) {
        for(int i=0;i<nhash;i++) {
            u32 h = murmur3(data, len, i*0x12345678) % sz;
            setBit(h);
        }
    }
    bool mightContain(const u8 *data, size_t len) const {
        for(int i=0;i<nhash;i++) {
            u32 h = murmur3(data, len, i*0x12345678) % sz;
            if(!testBit(h)) return false;
        }
        return true;
    }
};

class SkipList {
    struct Node {
        int val;
        std::vector<Node*> next;
        Node(int v, int level) : val(v), next(level, nullptr) {}
    };
    Node *head;
    int maxLevel, curLevel;
    std::mt19937 rng;
    int randLevel() {
        int l = 1;
        while(l < maxLevel && (rng()%2 == 0)) l++;
        return l;
    }
public:
    SkipList(int maxL = 16) : maxLevel(maxL), curLevel(1), rng(42) {
        head = new Node(INT_MIN, maxL);
    }
    ~SkipList() {
        Node *n = head;
        while(n) { Node *next = n->next[0]; delete n; n = next; }
    }
    void insert(int v) {
        std::vector<Node*> update(maxLevel, head);
        Node *cur = head;
        for(int i=curLevel-1;i>=0;i--) {
            while(cur->next[i] && cur->next[i]->val < v) cur = cur->next[i];
            update[i] = cur;
        }
        int level = randLevel();
        if(level > curLevel) {
            for(int i=curLevel;i<level;i++) update[i]=head;
            curLevel = level;
        }
        Node *n = new Node(v, level);
        for(int i=0;i<level;i++) { n->next[i]=update[i]->next[i]; update[i]->next[i]=n; }
    }
    bool search(int v) {
        Node *cur = head;
        for(int i=curLevel-1;i>=0;i--) while(cur->next[i] && cur->next[i]->val < v) cur=cur->next[i];
        cur = cur->next[0];
        return cur && cur->val == v;
    }
    bool remove(int v) {
        std::vector<Node*> update(maxLevel, head);
        Node *cur = head;
        for(int i=curLevel-1;i>=0;i--) {
            while(cur->next[i] && cur->next[i]->val < v) cur=cur->next[i];
            update[i]=cur;
        }
        cur=cur->next[0];
        if(!cur || cur->val != v) return false;
        for(int i=0;i<curLevel;i++) {
            if(update[i]->next[i]!=cur) break;
            update[i]->next[i]=cur->next[i];
        }
        delete cur;
        while(curLevel>1 && !head->next[curLevel-1]) curLevel--;
        return true;
    }
};

void radixSortStrings(std::vector<std::string> &strs) {
    if(strs.empty()) return;
    size_t maxLen = 0;
    for(auto &s : strs) maxLen = MAX(maxLen, s.size());
    for(int i=maxLen-1;i>=0;i--) {
        int cnt[257]={};
        for(auto &s : strs) cnt[(i<(int)s.size()?((u8)s[i]+1):0)+1]++;
        for(int j=1;j<257;j++) cnt[j]+=cnt[j-1];
        std::vector<std::string> aux(strs.size());
        for(auto &s : strs) aux[cnt[(i<(int)s.size()?(u8)s[i]:0)]++] = s;
        strs = aux;
    }
}

struct SuffixArray {
    std::string s;
    std::vector<int> sa, rank_, lcp_;
    SuffixArray(const std::string &str) : s(str + '\0') {
        int n = s.size();
        sa.resize(n); rank_.resize(n); lcp_.resize(n, 0);
        std::iota(sa.begin(), sa.end(), 0);
        for(int i=0;i<n;i++) rank_[i] = s[i];
        for(int k=1;k<n;k*=2) {
            auto cmp = [&](int a, int b) {
                if(rank_[a] != rank_[b]) return rank_[a] < rank_[b];
                int ra = a+k < n ? rank_[a+k] : -1;
                int rb = b+k < n ? rank_[b+k] : -1;
                return ra < rb;
            };
            std::sort(sa.begin(), sa.end(), cmp);
            std::vector<int> tmp(n);
            tmp[sa[0]] = 0;
            for(int i=1;i<n;i++) tmp[sa[i]] = tmp[sa[i-1]] + (cmp(sa[i-1],sa[i])?1:0);
            rank_ = tmp;
        }
        std::vector<int> inv(n);
        for(int i=0;i<n;i++) inv[sa[i]]=i;
        for(int i=0,h=0;i<n;i++) {
            if(inv[i]>0) {
                int j=sa[inv[i]-1];
                while(i+h<n&&j+h<n&&s[i+h]==s[j+h]) h++;
                lcp_[inv[i]]=h;
                if(h>0) h--;
            }
        }
    }
    std::vector<int> search(const std::string &pat) {
        int lo=0, hi=sa.size()-1, n=pat.size();
        while(lo<=hi) {
            int mid=(lo+hi)/2;
            if(s.substr(sa[mid],n)<pat) lo=mid+1; else hi=mid-1;
        }
        std::vector<int> res;
        for(int i=lo;i<(int)sa.size()&&s.substr(sa[i],n)==pat;i++) res.push_back(sa[i]);
        return res;
    }
};

class Tokenizer {
public:
    enum class TokType { Ident, Number, String, Op, Punct, End };
    struct Token { TokType type; std::string val; int line; };
private:
    std::string src;
    size_t pos, ln;
    Token makeToken(TokType t, std::string v) { return {t, v, (int)ln}; }
    bool isOp(char c) { return strchr("+-*/=<>!&|^~%", c)!=nullptr; }
public:
    Tokenizer(const std::string &s) : src(s), pos(0), ln(1) {}
    Token next() {
        while(pos<src.size()&&isspace(src[pos])) { if(src[pos]=='\n') ln++; pos++; }
        if(pos>=src.size()) return makeToken(TokType::End, "");
        char c = src[pos];
        if(c=='"'||c=='\'') {
            char delim=c; pos++;
            std::string v;
            while(pos<src.size()&&src[pos]!=delim) {
                if(src[pos]=='\\') { pos++; switch(src[pos]) { case 'n':v+='\n';break; case 't':v+='\t';break; default:v+=src[pos]; } }
                else v+=src[pos];
                pos++;
            }
            pos++;
            return makeToken(TokType::String, v);
        }
        if(isalpha(c)||c=='_') {
            std::string v;
            while(pos<src.size()&&(isalnum(src[pos])||src[pos]=='_')) v+=src[pos++];
            return makeToken(TokType::Ident, v);
        }
        if(isdigit(c)||((c=='-'||c=='+'))) {
            std::string v;
            if(c=='0'&&pos+1<src.size()&&src[pos+1]=='x') { v+="0x"; pos+=2; while(pos<src.size()&&isxdigit(src[pos])) v+=src[pos++]; }
            else { while(pos<src.size()&&(isdigit(src[pos])||src[pos]=='.')) v+=src[pos++]; }
            return makeToken(TokType::Number, v);
        }
        if(isOp(c)) {
            std::string v;
            while(pos<src.size()&&isOp(src[pos])) v+=src[pos++];
            return makeToken(TokType::Op, v);
        }
        return makeToken(TokType::Punct, std::string(1, src[pos++]));
    }
    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while(true) {
            Token t = next();
            tokens.push_back(t);
            if(t.type == TokType::End) break;
        }
        return tokens;
    }
};

template<typename T>
class PersistentVector {
    struct Node {
        T val;
        std::shared_ptr<Node> left, right;
        int size;
        Node(T v) : val(v), size(1) {}
        Node(std::shared_ptr<Node> l, std::shared_ptr<Node> r)
            : left(l), right(r), size((l?l->size:0)+(r?r->size:0)) {}
    };
    std::shared_ptr<Node> root;
    int n;
    std::shared_ptr<Node> update(std::shared_ptr<Node> nd, int lo, int hi, int i, T v) {
        if(lo==hi) return std::make_shared<Node>(v);
        int mid=(lo+hi)/2;
        if(i<=mid) return std::make_shared<Node>(update(nd->left,lo,mid,i,v),nd->right);
        return std::make_shared<Node>(nd->left,update(nd->right,mid+1,hi,i,v));
    }
    T query(std::shared_ptr<Node> nd, int lo, int hi, int i) {
        if(lo==hi) return nd->val;
        int mid=(lo+hi)/2;
        if(i<=mid) return query(nd->left,lo,mid,i);
        return query(nd->right,mid+1,hi,i);
    }
public:
    PersistentVector(int sz) : n(sz) {}
    PersistentVector set(int i, T v) const {
        PersistentVector pv = *this;
        pv.root = update(root, 0, n-1, i, v);
        return pv;
    }
    T get(int i) const { return query(root, 0, n-1, i); }
};

namespace vm {
    enum class Op { PUSH, POP, ADD, SUB, MUL, DIV, MOD, AND, OR, XOR, NOT, JMP, JZ, JNZ, CALL, RET, LOAD, STORE, CMP, HALT };
    struct Instr { Op op; i64 arg; };
    struct VM {
        std::vector<Instr> code;
        std::vector<i64> stack;
        std::vector<i64> mem;
        int ip, sp;
        bool halted;
        VM(size_t memSz=1024) : mem(memSz, 0), ip(0), sp(-1), halted(false) {}
        void push(i64 v) { stack.push_back(v); sp++; }
        i64 pop() { i64 v=stack.back(); stack.pop_back(); sp--; return v; }
        void step() {
            if(ip>=(int)code.size()||halted) return;
            Instr &ins = code[ip++];
            switch(ins.op) {
            case Op::PUSH: push(ins.arg); break;
            case Op::POP: pop(); break;
            case Op::ADD: { i64 b=pop(),a=pop(); push(a+b); break; }
            case Op::SUB: { i64 b=pop(),a=pop(); push(a-b); break; }
            case Op::MUL: { i64 b=pop(),a=pop(); push(a*b); break; }
            case Op::DIV: { i64 b=pop(),a=pop(); push(b?a/b:0); break; }
            case Op::MOD: { i64 b=pop(),a=pop(); push(b?a%b:0); break; }
            case Op::AND: { i64 b=pop(),a=pop(); push(a&b); break; }
            case Op::OR: { i64 b=pop(),a=pop(); push(a|b); break; }
            case Op::XOR: { i64 b=pop(),a=pop(); push(a^b); break; }
            case Op::NOT: push(~pop()); break;
            case Op::JMP: ip=ins.arg; break;
            case Op::JZ: { i64 v=pop(); if(v==0) ip=ins.arg; break; }
            case Op::JNZ: { i64 v=pop(); if(v!=0) ip=ins.arg; break; }
            case Op::LOAD: push(mem[ins.arg]); break;
            case Op::STORE: mem[ins.arg]=pop(); break;
            case Op::CMP: { i64 b=pop(),a=pop(); push(a==b?0:a<b?-1:1); break; }
            case Op::HALT: halted=true; break;
            default: break;
            }
        }
        void run() { while(!halted&&ip<(int)code.size()) step(); }
        i64 top() { return stack.empty()?0:stack.back(); }
    };
}

struct Config {
    std::string appName;
    int port;
    bool debug;
    std::string logPath;
    size_t maxConns;
    int timeout;
    std::map<std::string, std::string> extra;
    Config() : port(8080), debug(false), logPath("app.log"), maxConns(100), timeout(30) {}
    bool loadFromIni(const std::string &data) {
        IniParser p;
        if(!p.parse(data)) return false;
        appName = p.get("app","name","myapp");
        port = std::stoi(p.get("server","port","8080"));
        debug = p.get("app","debug","0")=="1";
        logPath = p.get("app","log","app.log");
        maxConns = std::stoul(p.get("server","max_connections","100"));
        timeout = std::stoi(p.get("server","timeout","30"));
        for(auto &[k,v] : p.sections["extra"]) extra[k]=v;
        return true;
    }
};

template<typename R, typename... Args>
class Memoize {
    std::map<std::tuple<Args...>, R> cache;
    std::function<R(Args...)> fn;
public:
    Memoize(std::function<R(Args...)> f) : fn(f) {}
    R operator()(Args... args) {
        auto key = std::make_tuple(args...);
        auto it = cache.find(key);
        if(it != cache.end()) return it->second;
        return cache[key] = fn(args...);
    }
    void clear() { cache.clear(); }
    size_t cacheSize() const { return cache.size(); }
};

void printHex(const u8 *data, size_t len) {
    for(size_t i=0;i<len;i++) {
        printf("%02x", data[i]);
        if((i+1)%16==0) printf("\n");
        else if((i+1)%8==0) printf("  ");
        else printf(" ");
    }
    if(len%16) printf("\n");
}

std::vector<u8> hexToBytes(const std::string &s) {
    std::vector<u8> out;
    for(size_t i=0;i+1<s.size();i+=2) {
        u8 b = (u8)strtoul(s.substr(i,2).c_str(), nullptr, 16);
        out.push_back(b);
    }
    return out;
}

std::string bytesToHex(const u8 *data, size_t len) {
    std::string s;
    char buf[3];
    for(size_t i=0;i<len;i++) { snprintf(buf,3,"%02x",data[i]); s+=buf; }
    return s;
}

bool isPalindrome(const std::string &s) {
    int l=0, r=s.size()-1;
    while(l<r) { if(s[l]!=s[r]) return false; l++; r--; }
    return true;
}

std::string reverseWords(std::string s) {
    std::reverse(s.begin(),s.end());
    size_t i=0;
    while(i<s.size()) {
        size_t j=i;
        while(j<s.size()&&!isspace(s[j])) j++;
        std::reverse(s.begin()+i,s.begin()+j);
        i=j+1;
    }
    return s;
}

int romanToInt(const std::string &s) {
    std::map<char,int> m={{'I',1},{'V',5},{'X',10},{'L',50},{'C',100},{'D',500},{'M',1000}};
    int r=0;
    for(int i=0;i<(int)s.size();i++) {
        if(i+1<(int)s.size()&&m[s[i]]<m[s[i+1]]) r-=m[s[i]];
        else r+=m[s[i]];
    }
    return r;
}

std::string intToRoman(int n) {
    int vals[]={1000,900,500,400,100,90,50,40,10,9,5,4,1};
    const char *syms[]={"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
    std::string r;
    for(int i=0;i<13;i++) while(n>=vals[i]) { r+=syms[i]; n-=vals[i]; }
    return r;
}

u32 nextPow2(u32 n) {
    n--; n|=n>>1; n|=n>>2; n|=n>>4; n|=n>>8; n|=n>>16;
    return n+1;
}

int popcount(u64 x) { return __builtin_popcountll(x); }
int ctz(u64 x) { return __builtin_ctzll(x); }
int clz(u64 x) { return __builtin_clzll(x); }
u64 reverseBits(u64 x) {
    x=((x&0x5555555555555555ULL)<<1)|((x>>1)&0x5555555555555555ULL);
    x=((x&0x3333333333333333ULL)<<2)|((x>>2)&0x3333333333333333ULL);
    x=((x&0x0F0F0F0F0F0F0F0FULL)<<4)|((x>>4)&0x0F0F0F0F0F0F0F0FULL);
    x=((x&0x00FF00FF00FF00FFULL)<<8)|((x>>8)&0x00FF00FF00FF00FFULL);
    x=(x<<48)|(((x>>16)&0xFFFF)<<32)|(((x>>32)&0xFFFF)<<16)|(x>>48);
    return x;
}

struct PhysicsBody {
    Vec3 pos, vel, acc, force;
    f32 mass, invMass;
    f32 restitution, friction;
    bool isStatic;
    PhysicsBody(f32 m=1, bool stat=false) : mass(m), invMass(stat?0:1/m), restitution(0.5f), friction(0.3f), isStatic(stat) {}
    void applyForce(Vec3 f) { if(!isStatic) force += f; }
    void integrate(f32 dt) {
        if(isStatic) return;
        acc = force * invMass;
        vel += acc * dt;
        pos += vel * dt;
        force = {};
    }
    void applyImpulse(Vec3 imp) { if(!isStatic) vel += imp * invMass; }
};

struct CollisionInfo {
    bool colliding;
    Vec3 normal;
    f32 depth;
    PhysicsBody *a, *b;
};

CollisionInfo checkSphereVsSphere(PhysicsBody &a, PhysicsBody &b, f32 ra, f32 rb) {
    CollisionInfo r{};
    Vec3 diff = b.pos - a.pos;
    f32 dist = diff.len();
    if(dist < ra+rb) {
        r.colliding = true;
        r.normal = dist > 1e-6f ? diff * (1/dist) : Vec3(0,1,0);
        r.depth = ra+rb-dist;
        r.a = &a; r.b = &b;
    }
    return r;
}

void resolveCollision(CollisionInfo &c) {
    if(!c.colliding) return;
    Vec3 rv = c.b->vel - c.a->vel;
    f32 velAlongNormal = rv.dot(c.normal);
    if(velAlongNormal > 0) return;
    f32 e = MIN(c.a->restitution, c.b->restitution);
    f32 j = -(1+e)*velAlongNormal/(c.a->invMass+c.b->invMass);
    Vec3 impulse = c.normal * j;
    c.a->applyImpulse(-impulse);
    c.b->applyImpulse(impulse);
    f32 slop = 0.01f, percent = 0.2f;
    Vec3 correction = c.normal * (MAX(c.depth-slop,0)/(c.a->invMass+c.b->invMass)*percent);
    c.a->pos -= correction * c.a->invMass;
    c.b->pos += correction * c.b->invMass;
}

template<typename T>
struct Optional {
    bool has;
    alignas(T) u8 storage[sizeof(T)];
    Optional() : has(false) {}
    Optional(T v) : has(true) { new(storage) T(v); }
    Optional(const Optional &o) : has(o.has) { if(has) new(storage) T(*reinterpret_cast<const T*>(o.storage)); }
    ~Optional() { if(has) reinterpret_cast<T*>(storage)->~T(); }
    T &value() { return *reinterpret_cast<T*>(storage); }
    const T &value() const { return *reinterpret_cast<const T*>(storage); }
    T valueOr(T def) const { return has ? value() : def; }
    explicit operator bool() const { return has; }
    void reset() { if(has) { reinterpret_cast<T*>(storage)->~T(); has=false; } }
};

template<typename F>
class LazyEval {
    mutable F fn;
    mutable bool evaluated;
    mutable typename std::result_of<F()>::type result;
public:
    LazyEval(F f) : fn(f), evaluated(false) {}
    const typename std::result_of<F()>::type &get() const {
        if(!evaluated) { result = fn(); evaluated = true; }
        return result;
    }
    void invalidate() { evaluated = false; }
};

template<typename T>
class SortedList {
    std::vector<T> data;
public:
    void insert(T v) {
        auto it = std::lower_bound(data.begin(), data.end(), v);
        data.insert(it, v);
    }
    bool contains(T v) const {
        auto it = std::lower_bound(data.begin(), data.end(), v);
        return it != data.end() && *it == v;
    }
    void remove(T v) {
        auto it = std::lower_bound(data.begin(), data.end(), v);
        if(it != data.end() && *it == v) data.erase(it);
    }
    size_t size() const { return data.size(); }
    T &operator[](size_t i) { return data[i]; }
    const std::vector<T> &raw() const { return data; }
};

int main(int argc, char **argv) {
    {
        std::vector<int> v = {5,3,8,1,9,2,7,4,6};
        quickSort(v, 0, v.size()-1);
        for(int x : v) printf("%d ", x); printf("\n");
    }
    {
        BST tree;
        for(int x : {5,3,7,1,4,6,8}) tree.insert(x);
        auto r = tree.inorder();
        for(int x : r) printf("%d ", x); printf("\n");
    }
    {
        Trie t;
        t.insert("hello"); t.insert("world"); t.insert("help");
        printf("search hello: %d\n", t.search("hello"));
        auto ac = t.autocomplete("hel");
        for(auto &s : ac) printf("  %s\n", s.c_str());
    }
    {
        Graph g(5);
        g.addUndirected(0,1); g.addUndirected(0,2); g.addUndirected(1,3); g.addUndirected(2,4);
        auto bfs = g.bfs(0);
        for(int x : bfs) printf("%d ", x); printf("\n");
    }
    {
        HashMap<std::string,int> hm;
        hm.insert("foo", 42); hm.insert("bar", 7);
        auto *v = hm.find("foo");
        if(v) printf("foo=%d\n", *v);
    }
    {
        ThreadPool pool(4);
        std::vector<std::future<int>> futs;
        for(int i=0;i<8;i++) futs.push_back(pool.submit([i]{ return i*i; }));
        for(auto &f : futs) printf("%d ", f.get()); printf("\n");
    }
    {
        std::string s = "hello world";
        std::string p = "world";
        auto matches = kmpSearch(s, p);
        for(int x : matches) printf("match at %d\n", x);
    }
    {
        auto d = json::parse("{\"name\":\"GeoSn0w\",\"age\":25,\"tags\":[\"ios\",\"security\"]}");
        printf("%s\n", d.dump(0).c_str());
    }
    {
        u8 data[] = "Hello World";
        auto digest = crypto::sha1(data, 11);
        printf("sha1: ");
        for(u8 b : digest) printf("%02x", b);
        printf("\n");
    }
    {
        SkipList sl;
        for(int x : {3,1,4,1,5,9,2,6}) sl.insert(x);
        printf("search 5: %d\n", sl.search(5));
        printf("search 7: %d\n", sl.search(7));
    }
    {
        auto url = net::URL::parse("https://geo.example.com:443/path/to/page?q=test&v=1#anchor");
        printf("scheme=%s host=%s port=%d path=%s\n",
            url.scheme.c_str(), url.host.c_str(), url.port, url.path.c_str());
    }
    {
        Matrix<f64> A(2,2), B(2,2);
        A.at(0,0)=1; A.at(0,1)=2; A.at(1,0)=3; A.at(1,1)=4;
        B.at(0,0)=5; B.at(0,1)=6; B.at(1,0)=7; B.at(1,1)=8;
        auto C = A*B;
        printf("C[0][0]=%g C[1][1]=%g\n", C.at(0,0), C.at(1,1));
    }
    {
        vm::VM v;
        v.code = {
            {vm::Op::PUSH, 10}, {vm::Op::PUSH, 20},
            {vm::Op::ADD, 0}, {vm::Op::STORE, 0},
            {vm::Op::LOAD, 0}, {vm::Op::HALT, 0}
        };
        v.run();
        printf("vm result: %lld\n", v.top());
    }
    {
        BloomFilter bf(1024, 3);
        const char *s = "GeoSn0w";
        bf.add((u8*)s, strlen(s));
        printf("bloom 'GeoSn0w': %d\n", bf.mightContain((u8*)s, strlen(s)));
        const char *s2 = "nothere";
        printf("bloom 'nothere': %d\n", bf.mightContain((u8*)s2, strlen(s2)));
    }
    {
        printf("lcs('ABCBDAB','BDCAB'): %s\n", longestCommonSubsequence("ABCBDAB","BDCAB").c_str());
    }
    {
        printf("edit dist: %d\n", editDistance("kitten","sitting"));
    }
    {
        auto primes = math::sieve(50);
        for(u64 p : primes) printf("%llu ", p); printf("\n");
    }
    {
        u8 data[] = "The quick brown fox";
        printf("crc32: %08X\n", crc32(data, sizeof(data)-1));
        printf("fnv1a: %08X\n", fnv1a(data, sizeof(data)-1));
    }
    return 0;
}
