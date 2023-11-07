#include <iostream>
#include <random>  

void generate_array(float* array, int len) {

    std::random_device rd;  
    std::mt19937 gen(rd());  
    std::uniform_real_distribution<> dis(0, 1);  

    for (int i = 0; i < len; i++) {
        array[i] = dis(gen);
    }

    return ;
}

void generate_matrix(float** matrix, int height, int width) {

    std::random_device rd;  
    std::mt19937 gen(rd());  
    std::uniform_real_distribution<> dis(0, 1);  

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            matrix[i][j] = dis(gen);
        }
    }

    return ;
}

void free_matrix(float** matrix, int height) {

    for (int i = 0; i < height; i++) {
        free(matrix[i]);
    }

    free(matrix);
}