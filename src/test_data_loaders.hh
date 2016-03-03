#ifndef TEST_DATA_LOADERS_HH
#define TEST_DATA_LOADERS_HH


#include "data_loader.hh"

#include <random>


class XorDataLoader : public DataLoader<double> {
public:
    XorDataLoader(bool useRandomDataEntries_, int seed = 0, unsigned int numEntries = 0);

    size_t loadNewEntry(void);

    void setSeed(int seed);

private:
    bool useRandomDataEntries;

    std::random_device randomDevice;
    std::default_random_engine randomEngine;
};


#endif // TEST_DATA_LOADERS_HH
