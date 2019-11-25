#include <iostream>
#include <vector>

using namespace std;
/*
Sieve of eratosthenes

given a number `n` finding all the prime numbers less than or equal to it.

algorithm
1. list down all numbers starting from 2 to n and assume them to be prime
2. start with first entry and start marking all its multiples as non primes, starting with the entries square
3. repeat step 2 for all entries that are prime till we reach an entry 'k' such that square of the entry 'k' exceeds `n`
*/

void getPrime(int n){
    int cur_val;
    vector<bool> num(n+1, true);
    for(int i=2;i*i<=n;i++){
        if(num[i]==true){
            for(int j=i*i;j<=n;j+=i){
                num[j]=false;
            }
        }
    }
    for(int i=2;i<=n;i++){
        if(num[i]){
            cout << i;
            if(i!=n)
                cout << " ";
        }
    }
}
int main(){
    int t, n;
    cin >> t;
    for(int nt=0; nt<t;nt++){
        cin >> n;
        getPrime(n);
    }
}