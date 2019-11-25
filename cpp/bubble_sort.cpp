#include <iostream>

using namespace std;

/*
*Compare adj elements and swap
*/
int* getArray(int n){
    int* array = new int[n];
    for(int i=0; i<n; i++){
        cin >> array[i];
    }
    return array;
}

int* bubSort(int n, int* array){
    int swaps=0, temp=0;
    for(int i=0; i<n-1; i++){ 
        // if the last element has to be first element n-1 min cycles are required so cannot be less than n-1
        for(int j=0;j<n-i-1;j++){
            // Keep pushing the largest element to the end this way we can ignore last i elements in subsequent loops 
            // for last i elements swaps won't be required so leave them hence n-i-1
            if(array[j]>array[j+1]){
                // for(int i=0; i<n; i++){
                //   cout << array[i] << " ";  
                // }
                // cout << endl;
                swaps++;
                temp = array[j];
                array[j] = array[j+1];
                array[j+1] = temp;
            }
            // else{
            // for(int i=0; i<n; i++){
            //       cout << array[i] << " ";  
            //     }
            // cout << endl;    
            // }
        }
    }
    // for(int i=0; i<n; i++){
    //   cout << array[i] << " ";  
    // } 
    // cout <<endl;
    cout << swaps;
    return array;
}

int main(){
    int n;
    cin >> n;
    int * array = getArray(n);
    bubSort(n, array);
}