#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstring>
#include <mpi.h>

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

    double getData(size_t i, size_t j) const { return data[i][j]; }
    void setData(size_t i, size_t j, double value) { data[i][j] = value; }
    size_t getSize() const { return n; }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    std::string fileA, fileB, fileResult;

    if (rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << "Лабораторная работа №3" << std::endl;
        std::cout << "Параллельное умножение матриц (MPI)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Количество доступных процессов: " << num_processes << std::endl;
        std::cout << std::endl;

        std::cout << "Введите имя файла с матрицей A: ";
        std::cin >> fileA;

        std::cout << "Введите имя файла с матрицей B: ";
        std::cin >> fileB;

        std::cout << "Введите имя файла для сохранения результата: ";
        std::cin >> fileResult;

        std::cout << "\nИспользуется процессов: " << num_processes << std::endl;
    }

    const int MAX_LEN = 256;
    char fileA_char[MAX_LEN] = { 0 };
    char fileB_char[MAX_LEN] = { 0 };
    char fileResult_char[MAX_LEN] = { 0 };

    if (rank == 0) {
        strncpy(fileA_char, fileA.c_str(), MAX_LEN - 1);
        strncpy(fileB_char, fileB.c_str(), MAX_LEN - 1);
        strncpy(fileResult_char, fileResult.c_str(), MAX_LEN - 1);
    }

    MPI_Bcast(fileA_char, MAX_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(fileB_char, MAX_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(fileResult_char, MAX_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    fileA = std::string(fileA_char);
    fileB = std::string(fileB_char);
    fileResult = std::string(fileResult_char);

    Matrix A, B;

    if (rank == 0) {
        if (!A.readFromFile(fileA)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (!B.readFromFile(fileB)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int n_int = 0;
    if (rank == 0) {
        n_int = static_cast<int>(A.getSize());
    }
    MPI_Bcast(&n_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    size_t n = static_cast<size_t>(n_int);

    if (rank != 0) {
        A = Matrix(n);
        B = Matrix(n);
    }

    std::vector<double> A_flat(n * n);
    if (rank == 0) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A_flat[i * n + j] = A.getData(i, j);
            }
        }
    }
    MPI_Bcast(A_flat.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A.setData(i, j, A_flat[i * n + j]);
            }
        }
    }

    std::vector<double> B_flat(n * n);
    if (rank == 0) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                B_flat[i * n + j] = B.getData(i, j);
            }
        }
    }
    MPI_Bcast(B_flat.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                B.setData(i, j, B_flat[i * n + j]);
            }
        }
    }

    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Информация о задаче" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Размер матриц: " << n << "x" << n << std::endl;
        std::cout << "Количество процессов: " << num_processes << std::endl;
        std::cout << "Объем данных: " << (2.0 * n * n * sizeof(double)) / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "\nВыполняется умножение..." << std::endl;
    }

    Matrix C(n);

    int rows_per_process = n_int / num_processes;
    int remainder = n_int % num_processes;

    int start_row = rank * rows_per_process;
    if (rank < remainder) {
        start_row += rank;
        rows_per_process++;
    }
    else {
        start_row += remainder;
    }

    std::vector<double> local_result(rows_per_process * n_int, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int i = 0; i < rows_per_process; ++i) {
        int global_i = start_row + i;
        for (size_t k = 0; k < n; ++k) {
            double aik = A.getData(global_i, k);
            if (aik != 0.0) {
                for (size_t j = 0; j < n; ++j) {
                    local_result[i * n_int + j] += aik * B.getData(k, j);
                }
            }
        }
    }

    if (rank == 0) {
        for (int i = 0; i < rows_per_process; ++i) {
            int global_i = start_row + i;
            for (int j = 0; j < n_int; ++j) {
                C.setData(global_i, j, local_result[i * n_int + j]);
            }
        }

        for (int p = 1; p < num_processes; ++p) {
            int p_rows = n_int / num_processes;
            int p_remainder = n_int % num_processes;
            if (p < p_remainder) {
                p_rows++;
            }

            std::vector<double> temp_result(p_rows * n_int);
            MPI_Recv(temp_result.data(), p_rows * n_int, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int p_start_row = p * (n_int / num_processes);
            if (p < remainder) {
                p_start_row += p;
            }
            else {
                p_start_row += remainder;
            }

            for (int i = 0; i < p_rows; ++i) {
                int global_i = p_start_row + i;
                for (int j = 0; j < n_int; ++j) {
                    C.setData(global_i, j, temp_result[i * n_int + j]);
                }
            }
        }
    }
    else {
        MPI_Send(local_result.data(), rows_per_process * n_int, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();
    double mpi_time_ms = (end_time - start_time) * 1000.0;

    if (rank == 0) {
        // Сохраняем результат
        C.writeToFile(fileResult);

        // Последовательное вычисление для сравнения
        auto start_seq = std::chrono::high_resolution_clock::now();
        Matrix C_seq = A.multiply(B);
        auto end_seq = std::chrono::high_resolution_clock::now();
        double seq_time_ms = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();

        double speedup = seq_time_ms / mpi_time_ms;
        double efficiency = (speedup / num_processes) * 100.0;

        std::cout << "\n========================================" << std::endl;
        std::cout << "РЕЗУЛЬТАТЫ" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Последовательное время: " << std::fixed << std::setprecision(2) << seq_time_ms << " мс" << std::endl;
        std::cout << "Параллельное время (" << num_processes << " процессов): " << std::fixed << std::setprecision(2) << mpi_time_ms << " мс" << std::endl;
        std::cout << "Ускорение: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "Эффективность: " << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
        std::cout << "Результат сохранён в: " << fileResult << std::endl;

        // Запись в CSV файл для построения графиков
        std::ofstream resultsFile("results_for_graph.csv", std::ios::app);
        if (resultsFile.is_open()) {
            // Проверяем, пустой ли файл
            resultsFile.seekp(0, std::ios::end);
            if (resultsFile.tellp() == 0) {
                // Заголовок для удобного построения графиков
                resultsFile << "MatrixSize,Processes_1,Processes_2,Processes_4,Processes_8\n";
            }
            resultsFile.close();
        }

        // Теперь нам нужно записывать данные для разных процессов
        // Открываем файл для чтения/записи
        std::fstream resultsFileRW("results_for_graph.csv", std::ios::in | std::ios::out);

        if (resultsFileRW.is_open()) {
            std::vector<std::string> lines;
            std::string line;
            bool found = false;

            // Читаем все строки
            while (std::getline(resultsFileRW, line)) {
                lines.push_back(line);
            }

            // Ищем строку с нужным размером матрицы
            std::string target = std::to_string(n_int) + ",";
            for (auto& l : lines) {
                if (l.find(target) == 0) {
                    // Нашли строку, добавляем данные для текущего количества процессов
                    std::stringstream ss(l);
                    std::string token;
                    std::vector<std::string> tokens;
                    while (std::getline(ss, token, ',')) {
                        tokens.push_back(token);
                    }

                    // Определяем индекс для количества процессов
                    int col_index = -1;
                    if (num_processes == 1) col_index = 1;
                    else if (num_processes == 2) col_index = 2;
                    else if (num_processes == 4) col_index = 3;
                    else if (num_processes == 8) col_index = 4;
                    else col_index = 5; // для других значений

                    // Добавляем или обновляем значение
                    while (tokens.size() <= col_index) {
                        tokens.push_back("");
                    }
                    tokens[col_index] = std::to_string(mpi_time_ms);

                    // Пересобираем строку
                    l = tokens[0];
                    for (size_t i = 1; i < tokens.size(); ++i) {
                        l += "," + tokens[i];
                    }
                    found = true;
                    break;
                }
            }

            if (!found) {
                // Создаем новую строку
                std::string new_line = std::to_string(n_int);
                for (int p = 1; p <= 8; p *= 2) {
                    if (p == num_processes) {
                        new_line += "," + std::to_string(mpi_time_ms);
                    }
                    else {
                        new_line += ",";
                    }
                }
                lines.push_back(new_line);
            }

            // Перезаписываем файл
            resultsFileRW.close();
            std::ofstream outFile("results_for_graph.csv");
            for (const auto& l : lines) {
                outFile << l << "\n";
            }
            outFile.close();

            std::cout << "\nДанные добавлены в results_for_graph.csv" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}