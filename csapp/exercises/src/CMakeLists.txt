#-- Sample code from the book Modern Cmake For C++
# This is the main entry our problem will use
add_library(sut STATIC calc.c run.c)
target_include_directories(sut PUBLIC .)

# This allows unit testing the main function, but it is empty at the moment
add_executable(bootstrap bootstrap.c)
target_link_libraries(bootstrap PRIVATE sut)
#-- END of sample code

add_executable(ch2.p55 run.ch2.p55.c)
target_include_directories(sut PUBLIC .)