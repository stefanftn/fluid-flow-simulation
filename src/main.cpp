#include <iostream>
#include <chrono>
#include "serial.h"
#include "parallel.h"

int STEPS = 20000;
int REFINE = 50;

int main() {
    /*auto start_s = std::chrono::high_resolution_clock::now();
    serial(STEPS, REFINE);
    auto end_s = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms_s = end_s - start_s;
    std::cout << "Serial: " << duration_ms_s.count() << "ms\n";*/

    auto start_p = std::chrono::high_resolution_clock::now();
    parallel(STEPS, REFINE);
    auto end_p = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms_p = end_p - start_p;
    std::cout << "Parallel: " << duration_ms_p.count() << "ms\n";

    return 0;
}
