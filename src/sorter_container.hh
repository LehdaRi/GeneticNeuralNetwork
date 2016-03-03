/**
    Sorter container for multiple metrics, created for use in genetic algorithm.
**/

#ifndef SORTER_CONTAINER_HH
#define SORTER_CONTAINER_HH


#include <vector>
#include <map>
#include <unordered_map>
#include <cstdlib>
#include <iostream> // TEMP


template<typename D, typename C, unsigned int S> // D: data pointers, C: comparison type, S: number of comparison metrics
class SorterContainer {
public:
    enum Direction {
        ASCENDING,
        DESCENDING
    };

    enum Exception {
        INVALID_KEY,
        INVALID_METRIC_ID,
        INVALID_NUM_ENTRIES,
        INVALID_ID,
        DATA_NOT_SORTED,
    };

    SorterContainer(void) {}
    ~SorterContainer(void) {
        for (auto& entry : data) {
            delete[] entry.second;
        }
    }

    // Operator overloads

    C* operator[] (D* key) {
        if (data.find(key) == data.end())
            throw INVALID_KEY;
        return data[key];
    }

    // For adding new S-sized entry to the container
    void addEntry(D* key, C* comp) {
        C* newComp = new C[S];
        memcpy(newComp, comp, sizeof(C)*S);
        data[key] = newComp;
    }

    void addEntry(D* key, C (&&comp)[S]) {
        C* newComp = new C[S];
        memcpy(newComp, comp, sizeof(comp));
        data[key] = newComp;
    }

    // For modifying existing entries (two first ones will add new entry if key is not found)
    void modifyEntry(D* key, C* comp) {
        C* newComp = new C[S];
        memcpy(newComp, comp, sizeof(C)*S);
        if (data.find(key) != data.end())
            delete[] data[key];
        data[key] = newComp;
    }

    void modifyEntry(D* key, C (&&comp)[S]) {
        C* newComp = new C[S];
        memcpy(newComp, comp, sizeof(comp));
        if (data.find(key) != data.end())
            delete[] data[key];
        data[key] = newComp;
    }

    void modifyEntry(D* key, unsigned int metricId, const C& comp) {
        if (data.find(key) == data.end())
            throw INVALID_KEY;
        if (metricId >= S)
            throw INVALID_METRIC_ID;

        data[key][metricId] = comp;
    }

    // For sorting the data
    void sort(unsigned int metricId, Direction direction) {
        sortData[metricId].clear();

        std::vector<std::pair<D*, C>> sortVector;

        for (auto& entry : data)
            sortVector.push_back(std::make_pair(entry.first, entry.second[metricId]));

        auto maxEntryIt = sortVector.begin();
        while (!sortVector.empty()) {
            if (direction == DESCENDING) {
                for (auto it = sortVector.begin(); it != sortVector.end(); ++it)
                    if (it->second > maxEntryIt->second)
                        maxEntryIt = it;
            }
            else if (direction == ASCENDING) {
                for (auto it = sortVector.begin(); it != sortVector.end(); ++it)
                    if (it->second < maxEntryIt->second)
                        maxEntryIt = it;
            }

            sortData[metricId].push_back(maxEntryIt->first);
            sortVector.erase(maxEntryIt);
            maxEntryIt = sortVector.begin();
        }
    }

    // Fetches best ranked entries from the sorted data
    // NOTE: data must be sorted according to all metrics before best entries can be fetched
    std::vector<D*> getBestEntries(unsigned int numEntries) const {
        if (numEntries > data.size())
            throw INVALID_NUM_ENTRIES;

        for (unsigned int i=0; i<S; ++i) {
            if (sortData[i].size() < data.size())
                throw DATA_NOT_SORTED;
        }

        std::vector<D*> bestEntries;
        std::unordered_map<D*, std::pair<unsigned int, unsigned int>> searchedData; //number of entries, average distance from the best element
        std::vector<D*> foundEntries;

        for (unsigned int i=0; i<data.size(); ++i) {
            foundEntries.clear();

            for (unsigned int j=0; j<S; ++j) {
                searchedData[sortData[j][i]].first++;
                searchedData[sortData[j][i]].second += i;

                if (searchedData[sortData[j][i]].first == S)
                    foundEntries.push_back(sortData[j][i]);
            }

            if (foundEntries.empty())
                continue;

            if (foundEntries.size() + bestEntries.size() < numEntries)
                bestEntries.insert(bestEntries.end(), foundEntries.begin(), foundEntries.end());
            else {
                while (bestEntries.size() < numEntries) {
                    auto bestEntryIt = foundEntries.begin();
                    for (auto it=++foundEntries.begin(); it!=foundEntries.end(); ++it)
                        if (searchedData[*it].second < searchedData[*bestEntryIt].second)
                            bestEntryIt = it;

                    bestEntries.push_back(*bestEntryIt);
                    foundEntries.erase(bestEntryIt);
                }
                std::cout << "i: " << i << std::endl;
                break;
            }
        }

        return bestEntries;
    }

    // get data pointer from sorted data structure
    D* getSortedEntry(unsigned int metricId, unsigned int id) const {
        if (sortData[metricId].size() < data.size())
            throw DATA_NOT_SORTED;

        if (metricId >= S)
            throw INVALID_METRIC_ID;

        if (id > data.size())
            throw INVALID_ID;

        return sortData[metricId][id];
    }

    // get sorted comparison value from sorted data structure
    const C& getSortedComparisonValue(unsigned int metricId, unsigned int id) {
        if (sortData[metricId].size() < data.size())
            throw DATA_NOT_SORTED;

        if (metricId >= S)
            throw INVALID_METRIC_ID;

        if (id > data.size())
            throw INVALID_ID;

        return data[sortData[metricId][id]][metricId];
    }

private:
    std::map<D*, C*> data;
    std::vector<D*> sortData[S];
};


#endif // SORTER_CONTAINER_HH
