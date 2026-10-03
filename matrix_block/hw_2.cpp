#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <omp.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace std;

using Clock = chrono::steady_clock;

void mul_ikj(const vector<double>& A, const vector<double>& B, vector<double>& C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            for (int j = 0; j < n; j++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mul_block(const vector<double>& A, const vector<double>& B, vector<double>& C, int n, int bs)
{
    const double* a = A.data();
    const double* b = B.data();
    double* c = C.data();
#pragma omp parallel for schedule(static) firstprivate(a, b, c, n, bs)
    for (int i = 0; i < n; i += bs)
    {
        for (int j = 0; j < n; j += bs)
        {
            for (int k = 0; k < n; k += bs)
            {
                for (int i1 = i; i1 < min(i + bs, n); i1++)
                {
                    for (int k1 = k; k1 < min(k + bs, n); k1++)
                    {
                        double r = a[i1 * n + k1];
                        for (int j1 = j; j1 < min(j + bs, n); j1++)
                        {
                            c[i1 * n + j1] += r * b[k1 * n + j1];
                        }
                    }
                }
            }
        }
    }
}

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

double measure(const vector<double>& A, const vector<double>& B, vector<double>& C, int n, int bs)
{
    // прогрев
    int warmups = (n >= 1000) ? 1 : 2;
    for (int w = 0; w < warmups; w++)
    {
        fill(C.begin(), C.end(), 0.0);
        if (bs == 0) mul_ikj(A, B, C, n);
        else mul_block(A, B, C, n, bs);
    }

    double total = 0;
    int iters = 0;
    while (total < 1.0)
    {
        fill(C.begin(), C.end(), 0.0);
        auto t1 = Clock::now();
        if (bs == 0) mul_ikj(A, B, C, n);
        else mul_block(A, B, C, n, bs);
        auto t2 = Clock::now();
        total += chrono::duration<double>(t2 - t1).count();
        iters++;
        sink = sink + C[(iters * 7919) % (n * n)];
    }
    return total / iters * 1000;
}

int main(int argc, char* argv[])
{
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif

#ifdef _MSC_VER
    cout << "MSVC " << _MSC_FULL_VER << endl;
#endif

    // i7-12700H 6 P-ядер + 8 E-ядер, 20 логических
    // P-ядро: L1d = 48 KB, L2 = 1.25 MB, L3 = 24 MB
    vector<int> sizes = { 160, 480, 832, 1760 };
    if (argc > 1)
    {
        sizes.clear();
        for (int i = 1; i < argc; i++)
        {
            sizes.push_back(stoi(argv[i]));
        }
    }
    // 3 блока по bs*bs*8 
    // 32 (24 KB L1), 64 (96 KB L2), 128 (384 KB L2)
    int blocks[3] = { 32, 64, 128 };
    int threads[8] = { 1, 2, 4, 6, 8, 12, 16, 20 };

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

        mul_ikj(A, B, ref, n);
        double t_seq = measure(A, B, C, n, 0);

        cout << endl << "N = " << n << "  (" << fixed << setprecision(1) << 3.0 * n * n * 8 / (1 << 20) 
            << " MiB),  seq ikj = " << setprecision(3) << t_seq << " ms" << endl;

        cout << setw(6) << "BS";
        for (int p = 0; p < 8; p++)
        {
            cout << setw(11) << ("p=" + to_string(threads[p]));
        }
        cout << "   (ms)" << endl;

        for (int b = 0; b < 3; b++)
        {
            cout << setw(6) << blocks[b] << flush;
            for (int p = 0; p < 8; p++)
            {
                omp_set_num_threads(threads[p]);
                double t = measure(A, B, C, n, blocks[b]);

                if (max_rel_diff(C, ref) > 1e-12)
                {
                    cout << endl << "error: N=" << n << " BS=" << blocks[b] << " p=" << threads[p] << endl;
                    return 1;
                }
                cout << setw(11) << setprecision(3) << t << flush;
            }
            cout << endl;
        }
    }
    return 0;
}