#include "generator.h"


using namespace std;

int main() {

    float arr[10];
    generate_array(arr, 10);
    for (int i = 0; i < 10; i++) {
        cout << arr[i] << endl;
    }


    float ** matrix;
    matrix = new float*[10];
    for (int i = 0; i < 10; i++) {
        matrix[i] = new float[20];
    }

    generate_matrix(matrix, 10, 20);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 20; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }

    // free_matrix(matrix, 10);
}