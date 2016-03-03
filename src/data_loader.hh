#ifndef DATA_LOADER_HH
#define DATA_LOADER_HH


#include "util.hh"

#include <vector>


template<typename T>
class DataLoader {
public:
    DataLoader(void) {}
    virtual ~DataLoader(void) {}

    /*  should be implemented in such way that it returns id of the newly loaded entry */
    virtual size_t loadNewEntry(void) = 0;

    /*  returns 0 if using random-generated data entries */
    unsigned int getDataEntriesNumber(void) const {
        return data.size();
    }

    /*  get inputs/outputs. dataEntryId is ignored when
        using random-generated data entries */
    const std::vector<T>& getInput(const unsigned int dataEntryId = 0) const {
        return data.at(dataEntryId).first;
    }
    const std::vector<T>& getOutput(const unsigned int dataEntryId = 0) const {
        return data.at(dataEntryId).second;
    }

protected:
    std::vector<std::pair<std::vector<T>,std::vector<T>>> data;
    ut::Rand rand;
};


#endif // DATA_LOADER_HH
