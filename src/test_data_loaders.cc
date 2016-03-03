#include "test_data_loaders.hh"


XorDataLoader::XorDataLoader(bool useRandomDataEntries_, int seed, unsigned int numEntries) :
    DataLoader(),
    useRandomDataEntries(useRandomDataEntries_),
    randomEngine(randomDevice())
    {
        setSeed(seed);

        if (!useRandomDataEntries) {
            for (unsigned int i=0; i<numEntries; ++i)
                loadNewEntry();
        }
    }

size_t XorDataLoader::loadNewEntry(void) {
    if (useRandomDataEntries)
        data.clear();

    std::vector<double> input, output;
    input.push_back((double)rand.iRand(0, 1));
    input.push_back((double)rand.iRand(0, 1));
    output.push_back((int)input[0]^(int)input[1]);

    data.push_back(std::make_pair(input, output));

    return data.size();
}

void XorDataLoader::setSeed(int seed) {
    randomEngine.seed(seed);
}


