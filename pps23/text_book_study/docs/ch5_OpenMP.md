
## Exercises

* 5.1 If it’s defined, the _OPENMP macro is a decimal int . Write a program that prints
its value. What is the significance of the value?
-> See https://www.openmp.org/spec-html/5.0/openmpse10.html
It is about the meta data of date to the CMP version you are with. Sample code as below:
```c++
    #include <unordered_map>
    #include <iostream>

    int main(int argc, char *argv[])
    {
    std::unordered_map<unsigned,std::string> map{
        {199810,"1.0"},
        {200203,"2.0"},
        {200505,"2.5"},
        {200805,"3.0"},
        {201107,"3.1"},
        {201307,"4.0"},
        {201511,"4.5"},
        {201811,"5.0"},
        {202011,"5.1"},
        {202111,"5.2"}
    };
    std::cout << "We have OpenMP " << map.at(_OPENMP) << ".\n";
    return 0;
    }
```