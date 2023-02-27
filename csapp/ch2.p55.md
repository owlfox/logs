# 2.55
Refer to [sample code](./exercises/src/run.ch2.p55.c)

To reproduce:
```
cmake ./ -B build
cmake --build ./build/
./build/bin/ch2.p55 
```

Console output of a int=4:
```
04 00 00 00
```

Hence my intel 12gen CPU is little endian.