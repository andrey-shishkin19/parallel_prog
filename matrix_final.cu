%%writefile matrix_final.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cuda_runtime.h>
#include <iomanip>

__global__ void matrixMultiplyKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

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
        if (n == 0) return false;
        data.assign(n, std::vector<double>(n, 0.0));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                file >> data[i][j];
        file.close();
        std::cout << "Загружена матрица " << filename << " размер " << n << "x" << n << std::endl;
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
        std::cout << "Результат сохранен в файл: " << filename << std::endl;
        return true;
    }

    Matrix multiplyCPU(const Matrix& other) const {
        Matrix result(n);
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < n; ++k)
                for (size_t j = 0; j < n; ++j)
                    result.data[i][j] += data[i][k] * other.data[k][j];
        return result;
    }

    Matrix multiplyCUDA(const Matrix& other, int blockSize) const {
        Matrix result(n);
        size_t size = n * n * sizeof(double);
        
        double *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);
        
        std::vector<double> flatA(n * n), flatB(n * n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                flatA[i * n + j] = data[i][j];
                flatB[i * n + j] = other.data[i][j];
            }
        
        cudaMemcpy(d_A, flatA.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, flatB.data(), size, cudaMemcpyHostToDevice);
        
        dim3 threadsPerBlock(blockSize, blockSize);
        dim3 blocksPerGrid((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);
        
        matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
        cudaDeviceSynchronize();
        
        std::vector<double> flatC(n * n);
        cudaMemcpy(flatC.data(), d_C, size, cudaMemcpyDeviceToHost);
        
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                result.data[i][j] = flatC[i * n + j];
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        
        return result;
    }
    
    bool compare(const Matrix& other) const {
        if (n != other.n) return false;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                if (std::abs(data[i][j] - other.data[i][j]) > 1e-6) 
                    return false;
        return true;
    }
    
    size_t getSize() const { return n; }
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
    
    if (!A.readFromFile(fileA)) return 1;
    if (!B.readFromFile(fileB)) return 1;
    
    if (A.getSize() != B.getSize()) {
        std::cerr << "Ошибка: размеры матриц не совпадают!" << std::endl;
        return 1;
    }
    
    size_t n = A.getSize();
    std::cout << "\nРазмер матриц: " << n << "x" << n << std::endl;
    
    // CPU умножение
    std::cout << "\nВыполняется умножение на CPU..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    Matrix C_cpu = A.multiplyCPU(B);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Время на CPU: " << cpu_time << " мс" << std::endl;
    
    // Сохранение результата
    std::cout << "\nСохранение результата..." << std::endl;
    if (!C_cpu.writeToFile(fileResult)) {
        return 1;
    }
    
    // CSV - открываем для дозаписи
    std::ofstream csv("results.csv", std::ios::app);
    
    // Проверяем, существует ли файл, если нет - создаем заголовок
    std::ifstream check("results.csv");
    bool isNew = !check.good();
    check.close();
    
    if (isNew) {
        csv << "MatrixSize,BlockConfiguration,SequentialTime_ms,ParallelTime_ms\n";
    }
    
    // Тестирование блоков
    std::cout << "\nТестирование GPU с разными блоками..." << std::endl;
    int blocks[] = {4, 8, 16, 32};
    
    for (int bs : blocks) {
        if (bs > n) continue;
        
        std::cout << "Блок " << bs << "x" << bs << ": ";
        start = std::chrono::high_resolution_clock::now();
        Matrix C_cuda = A.multiplyCUDA(B, bs);
        end = std::chrono::high_resolution_clock::now();
        auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        if (C_cuda.compare(C_cpu)) {
            std::cout << gpu_time << " мс (ускорение " << std::fixed << std::setprecision(2) << (double)cpu_time / gpu_time << "x)" << std::endl;
            csv << n << "," << bs << "x" << bs << "," << cpu_time << "," << gpu_time << "\n";
        } else {
            std::cout << "ОШИБКА!" << std::endl;
        }
    }
    
    csv.close();
    std::cout << "\nГотово! Результаты сохранены в results.csv" << std::endl;
    
    return 0;
}