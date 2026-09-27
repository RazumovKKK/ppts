#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>


using namespace std;
using Clock = chrono::steady_clock;


void mul_ijk(const vector<double>& A, const vector<double>& B, vector<double>& C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mul_ikj(const vector<double>& A, const vector<double>& B, vector<double>& C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            for (int j = 0; j < n; j++ )
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mul_jik(const vector<double>& A, const vector<double>& B, vector<double>& C, int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0 ; i < n; i++)
        {
            for (int k = 0; k < n; k++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mul_jki(const vector<double>& A, const vector<double>& B, vector<double>& C, int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int k = 0; k < n; k++)
        {
            for (int i = 0; i < n; i++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];

            }
        }
    }
}

void mul_kij(const vector<double>& A, const vector<double>& B, vector<double>& C, int n)
{
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mul_kji(const vector<double>& A, const vector<double>& B, vector<double>& C, int n)
{
    for (int k = 0; k < n; k++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

typedef void (*MulFunc)(const vector<double>&, const vector<double>&, vector<double>&, int);

double max_rel_diff(const vector<double>& X, const vector<double>& Y)
{
    double res = 0;
    for (size_t i = 0; i < X.size(); i++)
    {
        res = max(res, fabs(X[i] - Y[i]) / fabs(Y[i]));
    }

    return res;
}

volatile double sink = 0;

int main(int argc, char* argv[]) {
#ifdef _WIN32
    // привязка к логическому CPU 0 P-ядро на i7-12700H
    SetProcessAffinityMask(GetCurrentProcess(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif

#ifdef _MSC_VER
    cout << "MSVC " << _MSC_FULL_VER << endl;
#endif

    // i7-12700H, P-ядро: L1d = 48 KB, L2 = 1.25 MB, L3 = 24 MB
    // 160 < L2, 480 между L2 и L3, 832 чуть меньше L3, 1760заметно больше L3
    vector<int> sizes = { 160, 480, 832, 1760 };
    if (argc > 1)
    {
        sizes.clear();
        for (int i = 1; i < argc; i++)
        {
            sizes.push_back(stoi(argv[i]));
        }
    }

    MulFunc funcs[6] = { mul_ijk, mul_ikj, mul_jik, mul_jki, mul_kij, mul_kji };
    string names[6] = { "ijk", "ikj", "jik", "jki", "kij", "kji" };

    cout << setw(6) << "N" << setw(10) << "MiB";
    for (int f = 0; f < 6; f++) cout << setw(12) << names[f];
    cout << "   (ms)" << endl;

    for (int n : sizes)
    {
        vector<double> A(n * n), B(n * n), C(n * n), ref(n * n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i * n + j] = 1.0 + (i + 2 * j) % 10 * 0.1;
                B[i * n + j] = 1.0 + (3 * i + j) % 10 * 0.1;
            }
        }
        cout << setw(6) << n << setw(10) << fixed << setprecision(1) << 3.0 * n * n * 8 / (1 << 20) << flush;

        int warmups = (n >= 1000) ? 1 : 2;

        for (int f = 0; f < 6; f++)
        {
            // прогрев
            for (int w = 0; w < warmups; w++)
            {
                fill(C.begin(), C.end(), 0.0);
                funcs[f](A, B, C, n);
            }

            if (f == 0)
            {
                ref = C;
            }
            else if (max_rel_diff(C, ref) > 1e-12)
            {
                return 1;
            }
            double total = 0;
            int iters = 0;
            while (total < 1.0)
            {
                fill(C.begin(), C.end(), 0.0);
                auto t1 = Clock::now();
                funcs[f](A, B, C, n);
                auto t2 = Clock::now();
                total += chrono::duration<double>(t2 - t1).count();
                iters++;
                sink = sink + C[(iters * 7919) % (n * n)];
            }
            cout << setw(12) << setprecision(3) << total / iters * 1000 << flush;
        }
        cout << endl;
    }
    return 0;
}