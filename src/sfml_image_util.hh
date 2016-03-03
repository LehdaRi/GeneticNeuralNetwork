#ifndef SFML_IMAGE_UTIL_HH
#define SFML_IMAGE_UTIL_HH

#include <vector>
#include <SFML/Graphics.hpp>

namespace sfImgUtil {

    enum BlendMode {
        BLEND_ADD,
        BLEND_SUB,
        BLEND_DIFF
    };

    sf::Image blendImages(const sf::Image& img1, const sf::Image& img2, BlendMode bm);
    std::vector<sf::Vector2u> getRandomPointsFromBrightness(const sf::Image& img, unsigned int nPoints);
    sf::Image downsample16(const sf::Image& src);

    template <typename T>
    void fillImage(sf::Image& img, const std::vector<T>& data) {
        sf::Vector2u imgSize = img.getSize();

        if (imgSize.x*imgSize.y != data.size())
            return;

        for (unsigned y=0; y<imgSize.y; ++y)
            for (unsigned x=0; x<imgSize.x; ++x)
                img.setPixel(x, y, sf::Color(data[y*imgSize.x + x]*255, data[y*imgSize.x + x]*255, data[y*imgSize.x + x]*255));
    }
}

#endif
