#include "util.hh"


using namespace ut;

/*  some rndgen utilities */

//std::random_device Rand::rd = std::random_device();
//std::default_random_engine Rand::rdEng = std::default_random_engine(rd());

Rand::Rand(unsigned int seed_) {
    seed(seed_);
}

void Rand::seed(unsigned int seed) {
    rdEng.seed(seed);
}


float Rand::fRand(float from, float to) { // normalized float precision rand
    return from+(to-from)*((rdEng()%10000)/10000.0f);
}


double Rand::dRand(double from, double to) {
    return from+(to-from)*((rdEng()%100000000)/100000000.0f);
}


int Rand::iRand(int from, int to) {
    return from + rdEng()%(to-from+1);
}


bool Rand::bRand(void) {
    return rdEng()%2 ? true : false;
}
