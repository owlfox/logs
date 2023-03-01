#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main()
{
    int number_of_tosses;
    double x, y, distance_squared;
    int number_in_circle = 0;

    cout << "Enter the number of tosses: ";
    cin >> number_of_tosses;

    srand(time(NULL));  // initialize random seed

    for (int toss = 0; toss < number_of_tosses; toss++) {
        x = (double)rand() / RAND_MAX * 2 - 1; // generate random double between -1 and 1
        y = (double)rand() / RAND_MAX * 2 - 1; // generate random double between -1 and 1
        distance_squared = x * x + y * y;
        if (distance_squared <= 1)
            number_in_circle++;
    }

    double pi_estimate = 4 * (double)number_in_circle / (double)number_of_tosses;

    cout << "The estimated value of pi is: " << pi_estimate << endl;

    return 0;
}
