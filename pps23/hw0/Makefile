.phony: clean

default:
	cmake ./ -B build
	cmake --build ./build/

bin="./build/pi_estimate"

gprof:
	cmake ./ -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build ./build/
	echo 100000 | ${bin} 
	gprof ${bin} gmon.out > gprof_analysis.log

clean: gmon.out
	rm $^
