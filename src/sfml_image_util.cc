#include "sfml_image_util.hh"
#include "util.hh"

#include <iostream> //TEMP

sf::Image sfImgUtil::blendImages(const sf::Image& img1, const sf::Image& img2, BlendMode bm) {
    sf::Image newImg;

    sf::Vector2u
        img1Size(img1.getSize()),
        img2Size(img2.getSize());

    if (img1Size.x != img2Size.x || img1Size.y != img2Size.y) {
        return newImg;
        //TODO exception
    }

    newImg.create(img1Size.x, img1Size.y);

    switch (bm) {
    case BLEND_ADD:
        for (unsigned int y=0; y<img1Size.y; ++y) {
            for (unsigned int x=0; x<img1Size.x; ++x) {
                sf::Color
                    img1Pix(img1.getPixel(x, y)),
                    img2Pix(img2.getPixel(x, y));

                newImg.setPixel(x, y,
                                sf::Color(img1Pix.r+img2Pix.r,
                                          img1Pix.g+img2Pix.g,
                                          img1Pix.b+img2Pix.b));
            }
        }
    break;
    case BLEND_SUB:
        for (unsigned int y=0; y<img1Size.y; ++y) {
            for (unsigned int x=0; x<img1Size.x; ++x) {
                sf::Color
                    img1Pix(img1.getPixel(x, y)),
                    img2Pix(img2.getPixel(x, y));

                newImg.setPixel(x, y,
                                sf::Color(img1Pix.r-img2Pix.r,
                                          img1Pix.g-img2Pix.g,
                                          img1Pix.b-img2Pix.b));
            }
        }
    break;
    case BLEND_DIFF:
        for (unsigned int y=0; y<img1Size.y; ++y) {
            for (unsigned int x=0; x<img1Size.x; ++x) {
                sf::Color
                    img1Pix(img1.getPixel(x, y)),
                    img2Pix(img2.getPixel(x, y));

                newImg.setPixel(x, y,
                                sf::Color(abs(img1Pix.r-img2Pix.r),
                                          abs(img1Pix.g-img2Pix.g),
                                          abs(img1Pix.b-img2Pix.b)));
            }
        }
    break;
    }

    return newImg;
}

std::vector<sf::Vector2u> sfImgUtil::getRandomPointsFromBrightness(const sf::Image& img, unsigned int nPoints) {
    ut::Rand rand;

    sf::Vector2u imgSize = img.getSize();

    double* prob = new double[imgSize.x*imgSize.y];
    float maxBrightness = 0.0f;

    for (unsigned int y=0; y<imgSize.y; ++y) {
        for (unsigned int x=0; x<imgSize.x; ++x) {
            sf::Color pix = img.getPixel(x, y);

            float brightness = (pix.r + pix.g + pix.b)/765.0f;
            if (brightness > maxBrightness)
                maxBrightness = brightness;
        }
    }

    std::vector<sf::Vector2u> points;

    while (points.size() < nPoints) {
        unsigned int
            x(rand.iRand(0, imgSize.x-1)),
            y(rand.iRand(0, imgSize.y-1));

        sf::Color pix = img.getPixel(x, y);
        if (rand.fRand(0.0f, maxBrightness) < (pix.r + pix.g + pix.b)/765.0f)
            points.push_back(sf::Vector2u(x, y));
    }

    delete[] prob;

    return points;
}

sf::Image sfImgUtil::downsample16(const sf::Image& src) {
    sf::Image dest;
    unsigned
        w(static_cast<unsigned>(src.getSize().x/16.0f)),
        h(static_cast<unsigned>(src.getSize().y/16.0f));
    dest.create(w, h);

    for (size_t y=0; y<h; ++y) {
        for (size_t x=0; x<w; ++x) {

            float r(0.0f), g(0.0f), b(0.0f); // 16x downsampling
            for (size_t yi=0; yi<16; ++yi) {
                for (size_t xi=0; xi<16; ++xi) {
                    sf::Color p = src.getPixel(x*16+xi, y*16+yi);
                    r += p.r;
                    g += p.g;
                    b += p.b;
                }
            }
            r /= 256.0f;
            g /= 256.0f;
            b /= 256.0f;
            dest.setPixel(x, h-1-y, sf::Color((int)r, (int)g, (int)b));
        }
    }

    return dest;
}
