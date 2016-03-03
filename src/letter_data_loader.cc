#include "letter_data_loader.hh"

#include <sstream>
#include <iostream>


LetterDataLoader::LetterDataLoader(const std::string& dir,
                                   const unsigned ltrW,
                                   const unsigned ltrH,
                                   const unsigned numHor,
                                   const unsigned numVert) {
    for (char c='A'; c<='Z'; ++c) {
        sf::Image img;
        std::stringstream ss;
        ss << dir << "/" << c << ".png";

        std::cout << "loading " << ss.str() << ".. ";
        img.loadFromFile(ss.str());

        for (unsigned j=0; j<numVert; ++j) {
            for (unsigned i=0; i<numHor; ++i) {
                std::vector<float> in, out;

                //std::cout << "loading letter " << c << "(" << i << "," << j << ")" << std::endl;

                for (unsigned y=0; y<ltrH; ++y) {
                    for (unsigned x=0; x<ltrW; ++x) {
                        in.push_back(img.getPixel(ltrW*i+x, ltrH*j+y).r/255.0f);
                    }
                }

                for (unsigned k=0; k<26; ++k) {
                    out.push_back(0.0f);
                }

                out[(unsigned)(c-'A')] = 1.0f;

                data.push_back(std::make_pair(std::move(in), std::move(out)));
            }
        }

        std::cout << "done" << std::endl;
    }
}

size_t LetterDataLoader::loadNewEntry(void) {
    return 0;
}
