/*
* for array of len n
* for i in (0,n-1)
*    select min element from (i,n-1) and swap its position with element at position i
*/
#include <iostream>
using namespace std;

int* getArray(int n){
    int* array = new int[n];
    for(int i=0;i<n;i++)
        cin >> array[i];
    return array;
}

int* selSort(int n, int* array, int x){
    int temp;
    for(int i=0; i<n; i++){
        int min_ele = i;
        for(int j=i; j<n ;j++){
            if(array[j]<array[min_ele]){
                min_ele = j;
            }
        }
        temp = array[i];
        array[i] = array[min_ele];
        array[min_ele] = temp;
        
        if(i==x){
            for(int k=0;k<n;k++){
                cout << array[k] << " ";
            }
        }
    }
}
int main(){
    int n, x;
    cin >> n; cin >> x;
    int* array = getArray(n);
    selSort(n, array, x-1);
}