#include <iostream>
using namespace std;


int main(void){
    int N = 10;
    float H[N*2][N*2];

    for (size_t i = 0; i < N*2; i++)
    {
        for (size_t j = 0; j < N*2; j++)
        {
            H[i][j] = 0;
            
        }
        
    }

    for (int k = 0; k < N*2; k++) {
        for (int i = 0; i < N*2; i++) {
            cout << "cube[" << k << "][" << i << "] = " << H[k][i] << endl;

        }
    }
    
}