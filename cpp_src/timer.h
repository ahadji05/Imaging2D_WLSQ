#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <string>
#include <chrono>

extern "C"
{

//timer uses milliseconds to catch floating point precision later
//when we want to summarize all timings before exit in seconds.
class timer {

    public:
        std::chrono::system_clock::time_point clickStart;
        std::chrono::system_clock::time_point clickStop;
        std::chrono::duration<double, std::milli> elapsed;
        std::string name;
        int ncalls;

        timer(const std::string & name) : name(name){
            ncalls = 0;
        }

        ~timer(){};

        void start(){
            ncalls += 1;
            clickStart = std::chrono::high_resolution_clock::now();
        }

        void stop(){
            clickStop = std::chrono::high_resolution_clock::now();
            elapsed += clickStop - clickStart;
        }

        void dispInfo(){
            std::cout << name << ": ";
            std::cout << "calls(" << ncalls << ") total ";
            std::cout << elapsed.count() / 1000.0 << " seconds." << std::endl;
        }

};

} // end extern C

#endif