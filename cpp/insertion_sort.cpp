#include <iostream>
#include <map>
#include <vector>

/*
* get elem
* check if prev elem is greater than current elem if no elem skip
* if yes move current elem back and do it till this condition is false;
*
*/
using namespace std;

int * getArray(int n){
    int* array = new int[n];
    for(int i=0;i<n;i++){
        cin >> array[i];
    }
    return array;
}

int main(){
    int j, n, temp;
    cin >> n;
    int * array = getArray(n);
    vector<int> initial(array, array+n);
    map<int, int> map_;
    for(int i=0;i<n;i++){
        j = i;
        temp = array[j];
        map_[array[j]]=j;
        while (j>0&&temp<array[j-1]){
            array[j] = array[j-1];
            map_[array[j]]=j;
            j-=1;
            
        }
        array[j] = temp;
        map_[array[j]]=j;
        
    }
    for(int l=0;l<n;l++)
        cout << map_[initial[l]]+1 << " ";
    cout << endl;
}