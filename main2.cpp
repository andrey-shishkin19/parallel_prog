#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <omp.h>

class Matrix {
private:
    std::vector<std::vector<long long>> data;
    size_t n;

public:
    Matrix(size_t size = 0) : n(size), data(size, std::vector<long long>(size, 0)) {}

    bool readFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;

        file >> n;
        if (n == 0) return false;

        data.assign(n, std::vector<long long>(n, 0));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                if (!(file >> data[i][j])) return false;

        return true;
    }

    bool writeToFile(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        file << n << "\n";
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                file << data[i][j];
                if (j < n - 1) file << " ";
            }
            file << "\n";
        }
        return true;
    }

    Matrix multiplyParallel(const Matrix& other, int num_threads) const {
        Matrix result(n);
#pragma omp parallel for num_threads(num_threads) collapse(2)
        for (int i = 0; i < static_cast<int>(n); ++i)
            for (int j = 0; j < static_cast<int>(n); ++j) {
                long long sum = 0;
                for (int k = 0; k < static_cast<int>(n); ++k)
                    sum += data[i][k] * other.data[k][j];
                result.data[i][j] = sum;
            }
        return result;
    }

    Matrix multiplySequential(const Matrix& other) const {
        Matrix result(n);
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < n; ++k) {
                long long aik = data[i][k];
                if (aik != 0)
                    for (size_t j = 0; j < n; ++j)
                        result.data[i][j] += aik * other.data[k][j];
            }
        return result;
    }

    size_t getSize() const { return n; }
};

int main() {
    std::string fileA, fileB, fileResult;
    int num_threads;

    std::cout << "Введите имя файла с матрицей A: ";
    std::cin >> fileA;
    std::cout << "Введите имя файла с матрицей B: ";
    std::cin >> fileB;
    std::cout << "Введите имя файла для сохранения результата: ";
    std::cin >> fileResult;
    std::cout << "Введите количество потоков (1, 2, 4, 8): ";
    std::cin >> num_threads;

    Matrix A, B;
    if (!A.readFromFile(fileA) || !B.readFromFile(fileB)) {
        std::cerr << "Ошибка при чтении матриц" << std::endl;
        return 1;
    }

    if (A.getSize() != B.getSize()) {
        std::cerr << "Ошибка: размеры матриц не совпадают" << std::endl;
        return 1;
    }

    size_t n = A.getSize();

    auto start = std::chrono::high_resolution_clock::now();
    Matrix C = A.multiplyParallel(B, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!C.writeToFile(fileResult)) {
        std::cerr << "Ошибка при сохранении результата" << std::endl;
        return 1;
    }

    auto start_seq = std::chrono::high_resolution_clock::now();
    Matrix C_seq = A.multiplySequential(B);
    auto end_seq = std::chrono::high_resolution_clock::now();
    auto duration_seq = std::chrono::duration_cast<std::chrono::milliseconds>(end_seq - start_seq);

    std::ofstream resultsFile("results_parallel.csv", std::ios::app);
    if (resultsFile.is_open()) {
        resultsFile.seekp(0, std::ios::end);
        if (resultsFile.tellp() == 0)
            resultsFile << "size,threads,time_ms,seq_time_ms\n";
        resultsFile << n << "," << num_threads << "," << duration.count() << "," << duration_seq.count() << "\n";
        resultsFile.close();
    }

    std::cout << "Результат умножения сохранён в файл: " << fileResult << std::endl;
    std::cout << "Результаты эксперимента добавлены в results_parallel.csv" << std::endl;

    return 0;
}
