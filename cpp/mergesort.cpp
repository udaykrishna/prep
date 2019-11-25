#include <iostream>

using namespace std;

int *getArray(int n){
    int * array = new int[n];
    for(int i=0;i<n;i++)
        cin >> array[i];
    return array;
}

int main(){
    int n;
    cin >> n;
    int * array = getArray(n);
}