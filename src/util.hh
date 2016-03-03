#ifndef UTIL_HH
#define UTIL_HH

#include <map>
#include <algorithm>
#include <random>
#include <ctime>


namespace ut {
    /*  some rndgen utilities */

    class Rand {
    public:
        Rand(unsigned int seed_ = rand());

        void seed(unsigned int seed);
        float fRand(float from, float to);
        double dRand(double from, double to);
        int iRand(int from, int to);
        bool bRand(void);

    private:
        std::random_device rd;
        std::default_random_engine rdEng;
    };


    /*  utilities for flipping pairs and maps */

    template<typename A, typename B>
    std::pair<B,A> flip_pair(const std::pair<A,B> &p)
    {
        return std::pair<B,A>(p.second, p.first);
    }


    template<typename A, typename B>
    std::map<B,A> flip_map(const std::map<A,B> &src)
    {
        std::map<B,A> dst;
        std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()),
                       flip_pair<A,B>);
        return dst;
    }
}


#endif // UTIL_HH
