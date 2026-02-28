#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace chrono;

// Функция чтения матрицы из файла
vector<vector<double>> readMatrix(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        exit(1);
    }

    int n;
    file >> n;

    vector<vector<double>> matrix(n, vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file >> matrix[i][j];
        }
    }

    file.close();
    return matrix;
}

// Функция записи матрицы в файл
void writeMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    int n = matrix.size();

    file << n << "\n";

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[i][j] << " ";
        }
        file << "\n";
    }

    file.close();
}

// Функция записи результатов эксперимента
void writeResults(const string& filename, int n, long long read_time,
    long long mult_time, long long write_time, long long total_time) {
    ofstream file(filename, ios::app);  // для добавления

    // Проверяем, пустой ли файл
    file.seekp(0, ios::end);
    if (file.tellp() == 0) {
        // Заголовок
        file << "Размер  Элементов  Операций    Чтение(мс)  Умножение(мс)  Запись(мс)  Всего(мс)\n";
    }

    // Запись данных
    file << n << "       "
        << n * n << "         "
        << (long long)n * n * n << "       "
        << read_time << "          "
        << mult_time << "             "
        << write_time << "         "
        << total_time << "\n";

    file.close();
}

// Функция умножения матриц
vector<vector<double>> multiply(const vector<vector<double>>& A,
    const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

int main() {
    string fileA, fileB, fileC;
    string resultsFile = "results.txt";  // файл для сохранения результатов

    cout << "Введите имя файла с первой матрицей (например, A100.txt): ";
    cin >> fileA;

    cout << "Введите имя файла со второй матрицей (например, B100.txt): ";
    cin >> fileB;

    cout << "Введите имя файла для результата (например, result.txt): ";
    cin >> fileC;

    cout << "\n\n";

    cout << "Читаем файлы.\n";
    auto start_total = high_resolution_clock::now();
    auto start_read = high_resolution_clock::now();

    auto A = readMatrix(fileA);
    auto B = readMatrix(fileB);

    auto end_read = high_resolution_clock::now();
    auto read_time = duration_cast<milliseconds>(end_read - start_read).count();

    int n = A.size();
    cout << "Размер матриц: " << n << " x " << n << "\n";
    cout << "Время чтения: " << read_time << " мс\n\n";

    cout << "Умножаем матрицы.\n";
    auto start_mult = high_resolution_clock::now();

    auto C = multiply(A, B);

    auto end_mult = high_resolution_clock::now();
    auto mult_time = duration_cast<milliseconds>(end_mult - start_mult).count();

    cout << "Время умножения: " << mult_time << " мс\n\n";

    cout << "Записываем результат.\n";
    auto start_write = high_resolution_clock::now();

    writeMatrix(fileC, C);

    auto end_write = high_resolution_clock::now();
    auto write_time = duration_cast<milliseconds>(end_write - start_write).count();
    auto end_total = high_resolution_clock::now();
    auto total_time = duration_cast<milliseconds>(end_total - start_total).count();

    cout << "Время записи: " << write_time << " мс\n\n";

    writeResults(resultsFile, n, read_time, mult_time, write_time, total_time);
    cout << "Результаты сохранены в файл: " << resultsFile << "\n\n";

    cout << "Результаты:\n";
    cout << "Размер матриц:      " << n << " x " << n << "\n";
    cout << "Объём данных:        " << n * n * 2 << " элементов\n";
    cout << "Количество операций: " << (long long)n * n * n << "\n\n";

    cout << "Время выполнения:\n";
    cout << "Чтение:    " << read_time << " мс\n";
    cout << "Умножение: " << mult_time << " мс\n";
    cout << "Запись:    " << write_time << " мс\n";
    cout << "Всего:     " << total_time << " мс\n";

    return 0;
}