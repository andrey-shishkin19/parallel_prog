#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t n;

public:
    Matrix(size_t size = 0) : n(size), data(size, std::vector<double>(size, 0.0)) {}

    bool readFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Ошибка: не удалось открыть файл " << filename << std::endl;
            return false;
        }

        file >> n;
        if (n == 0) {
            std::cerr << "Ошибка: неверный размер матрицы в файле " << filename << std::endl;
            return false;
        }

        data.assign(n, std::vector<double>(n, 0.0));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (!(file >> data[i][j])) {
                    std::cerr << "Ошибка: недостаточно данных в файле " << filename << std::endl;
                    return false;
                }
            }
        }

        file.close();
        return true;
    }

    bool writeToFile(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Ошибка: не удалось создать файл " << filename << std::endl;
            return false;
        }

        file << n << "\n";
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                file << std::fixed << std::setprecision(6) << data[i][j];
                if (j < n - 1) file << " ";
            }
            file << "\n";
        }

        file.close();
        return true;
    }

    Matrix multiply(const Matrix& other) const {
        if (n != other.n) {
            std::cerr << "Ошибка: размеры матриц не совпадают для умножения" << std::endl;
            return Matrix(0);
        }

        Matrix result(n);

        // Классический алгоритм перемножения матриц
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = 0; k < n; ++k) {
                double aik = data[i][k];
                if (aik != 0.0) {
                    for (size_t j = 0; j < n; ++j) {
                        result.data[i][j] += aik * other.data[k][j];
                    }
                }
            }
        }

        return result;
    }

    size_t getSize() const { return n; }

    size_t getMemorySize() const {
        return n * n * sizeof(double) + sizeof(n) + sizeof(data);
    }
};

int main() {
    std::string fileA, fileB, fileResult;

    std::cout << "Введите имя файла с матрицей A: ";
    std::cin >> fileA;

    std::cout << "Введите имя файла с матрицей B: ";
    std::cin >> fileB;

    std::cout << "Введите имя файла для сохранения результата: ";
    std::cin >> fileResult;

    Matrix A, B;

    if (!A.readFromFile(fileA)) {
        return 1;
    }

    if (!B.readFromFile(fileB)) {
        return 1;
    }

    if (A.getSize() != B.getSize()) {
        std::cerr << "Ошибка: размеры матриц не совпадают ("
            << A.getSize() << " != " << B.getSize() << ")" << std::endl;
        return 1;
    }

    size_t n = A.getSize();
    size_t taskSize = 2 * n * n * sizeof(double) + n * n * sizeof(double);

    std::cout << "\n=== Информация о задаче ===" << std::endl;
    std::cout << "Размер матриц: " << n << "x" << n << std::endl;
    std::cout << "Объем входных данных: " << (2 * n * n * sizeof(double)) / 1024.0 << " KB" << std::endl;
    std::cout << "Объем выходных данных: " << (n * n * sizeof(double)) / 1024.0 << " KB" << std::endl;
    std::cout << "Общий объем задачи: " << taskSize / 1024.0 << " KB" << std::endl;

    std::cout << "\nВыполняется умножение матриц..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    Matrix C = A.multiply(B);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!C.writeToFile(fileResult)) {
        return 1;
    }

    std::cout << "\n=== Результаты ===" << std::endl;
    std::cout << "Время выполнения: " << duration.count() << " мс" << std::endl;
    std::cout << "Результат сохранен в файл: " << fileResult << std::endl;

    // Запись в файл результатов для аналитики
    std::ofstream resultsFile("results.txt", std::ios::app);
    if (resultsFile.is_open()) {
        resultsFile << n << "x" << n << "    "
            << (2 * n * n * sizeof(double)) / 1024.0 << " KB    "
            << (n * n * sizeof(double)) / 1024.0 << " KB    "
            << taskSize / 1024.0 << " KB    "
            << duration.count() << " ms\n";
        resultsFile.close();
    }

    return 0;
}