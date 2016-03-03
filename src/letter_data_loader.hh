#ifndef LETTER_DATA_LOADER_HH
#define LETTER_DATA_LOADER_HH


#include "data_loader.hh"

#include <string>
#include <SFML/Graphics.hpp>


class LetterDataLoader : public DataLoader<float> {
public:
    LetterDataLoader(const std::string& dir,
                     const unsigned ltrW,
                     const unsigned ltrH,
                     const unsigned numHor,
                     const unsigned numVert);
    ~LetterDataLoader(void) {}

    size_t loadNewEntry(void);
};


#endif // LETTER_DATA_LOADER_HH
