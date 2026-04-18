#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
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
        if (!file.is_open()) {
            std::cerr << "Ошибка: не удалось открыть файл " << filename << std::endl;
            return false;
        }

        file >> n;
        if (n == 0) {
            std::cerr << "Ошибка: неверный размер матрицы в файле " << filename << std::endl;
            return false;
        }

        data.assign(n, std::vector<long long>(n, 0));
        
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
                file << data[i][j];
                if (j < n - 1) file << " ";
            }
            file << "\n";
        }
        
        file.close();
        return true;
    }

    Matrix multiplyParallel(const Matrix& other, int num_threads) const {
        if (n != other.n) {
            std::cerr << "Ошибка: размеры матриц не совпадают для умножения" << std::endl;
            return Matrix(0);
        }

        Matrix result(n);
        
        // Параллельное умножение матриц с использованием OpenMP
        #pragma omp parallel for num_threads(num_threads) collapse(2)
        for (int i = 0; i < static_cast<int>(n); ++i) {
            for (int j = 0; j < static_cast<int>(n); ++j) {
                long long sum = 0;
                for (int k = 0; k < static_cast<int>(n); ++k) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        
        return result;
    }

    Matrix multiplySequential(const Matrix& other) const {
        if (n != other.n) {
            std::cerr << "Ошибка: размеры матриц не совпадают для умножения" << std::endl;
            return Matrix(0);
        }

        Matrix result(n);
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = 0; k < n; ++k) {
                long long aik = data[i][k];
                if (aik != 0) {
                    for (size_t j = 0; j < n; ++j) {
                        result.data[i][j] += aik * other.data[k][j];
                    }
                }
            }
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
    
    std::cout << "Введите количество потоков (1, 2, 4, 8...): ";
    std::cin >> num_threads;
    
    // Проверка количества потоков
    int max_threads = omp_get_max_threads();
    if (num_threads > max_threads) {
        std::cout << "Предупреждение: запрошено " << num_threads 
                  << " потоков, но доступно только " << max_threads << std::endl;
        num_threads = max_threads;
    }
    
    Matrix A, B;
    
    if (!A.readFromFile(fileA)) {
        std::cerr << "Ошибка при чтении матрицы A" << std::endl;
        system("pause");
        return 1;
    }
    
    if (!B.readFromFile(fileB)) {
        std::cerr << "Ошибка при чтении матрицы B" << std::endl;
        system("pause");
        return 1;
    }
    
    if (A.getSize() != B.getSize()) {
        std::cerr << "Ошибка: размеры матриц не совпадают (" 
                  << A.getSize() << " != " << B.getSize() << ")" << std::endl;
        system("pause");
        return 1;
    }
    
    size_t n = A.getSize();
    size_t taskSize = 2 * n * n * sizeof(long long) + n * n * sizeof(long long);
    
    std::cout << "Информация о задаче" << std::endl;
    std::cout << "Размер матриц: " << n << "x" << n << std::endl;
    std::cout << "Тип данных: long long (целые числа)" << std::endl;
    std::cout << "Объем входных данных: " << (2 * n * n * sizeof(long long)) / 1024.0 << " KB" << std::endl;
    std::cout << "Объем выходных данных: " << (n * n * sizeof(long long)) / 1024.0 << " KB" << std::endl;
    std::cout << "Общий объем задачи: " << taskSize / 1024.0 << " KB" << std::endl;
    
    std::cout << "Параллельные настройки" << std::endl;
    std::cout << "Количество потоков: " << num_threads << std::endl;
    std::cout << "Максимально доступно потоков: " << max_threads << std::endl;
    
    std::cout << "\nВыполняется параллельное умножение матриц..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    Matrix C = A.multiplyParallel(B, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!C.writeToFile(fileResult)) {
        std::cerr << "Ошибка при сохранении результата" << std::endl;
        system("pause");
        return 1;
    }
    
    // Дополнительно замеряем последовательное выполнение для сравнения
    std::cout << "\nВыполняется последовательное умножение для сравнения..." << std::endl;
    auto start_seq = std::chrono::high_resolution_clock::now();
    Matrix C_seq = A.multiplySequential(B);
    auto end_seq = std::chrono::high_resolution_clock::now();
    auto duration_seq = std::chrono::duration_cast<std::chrono::milliseconds>(end_seq - start_seq);
    
    double speedup = static_cast<double>(duration_seq.count()) / duration.count();
    double efficiency = speedup / num_threads * 100;
    
    std::cout << std::endl;
    std::cout << "Время выполнения (параллельное, " << num_threads << " потоков): " 
              << duration.count() << " мс" << std::endl;
    std::cout << "Время выполнения (последовательное): " << duration_seq.count() << " мс" << std::endl;
    std::cout << "Ускорение (Speedup): " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "Эффективность: " << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
    std::cout << "Результат сохранен в файл: " << fileResult << std::endl;
    
    // Запись в CSV файл для аналитики
    std::ofstream resultsFile("results_parallel.csv", std::ios::app);
    if (resultsFile.is_open()) {
        // Проверяем, пустой ли файл, чтобы записать заголовок
        resultsFile.seekp(0, std::ios::end);
        if (resultsFile.tellp() == 0) {
            resultsFile << "size,threads,time_ms,seq_time_ms,speedup,efficiency\n";
        }
        resultsFile << n << "," << num_threads << "," << duration.count() 
                   << "," << duration_seq.count() << "," << speedup << "," << efficiency << "\n";
        resultsFile.close();
    }
    
    std::cout << "\nРезультаты эксперимента добавлены в файл results_parallel.csv" << std::endl;
    
    system("pause");
    return 0;
}
