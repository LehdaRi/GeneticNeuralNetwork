#include "src/node_genetics.hh"
#include "src/test_data_loaders.hh"
#include "src/letter_data_loader.hh"
#include "src/util.hh"
#include "src/sfml_image_util.hh"

#include <iostream>
#include <map>
#include <ctime>
#include <cmath>
#include <SFML/Graphics.hpp>


#define TESTID 1
#define IMGSPRITEMACRO(name, data, window, w, h, posx, posy, sx, sy) name##Img.create(w, h); \
                                                                     sfImgUtil::fillImage(name##Img, data); \
                                                                     name##Tex.loadFromImage(name##Img); \
                                                                     name##Spr.setPosition(posx, posy); \
                                                                     name##Spr.setScale(sx, sy); \
                                                                     name##Spr.setTexture(name##Tex); \
                                                                     window.draw(name##Spr);
#if TESTID == 6

void test(void) {
    // load the network
    std::cout << "loading network.. ";
    NeuralNetwork nn("letters.ffnn");
    std::cout << "done." << std::endl;

    //set up the window
    sf::RenderWindow window;
    window.create(sf::VideoMode(1024, 768), "Genetic Neural Networks");

    //visualisation
    sf::Image inputImg, hidden1Img, hidden2Img, hidden3Img, outputImg;
    sf::Texture inputTex, hidden1Tex, hidden2Tex, hidden3Tex, outputTex;
    sf::Sprite inputSpr, hidden1Spr, hidden2Spr, hidden3Spr, outputSpr;
    sf::Texture alphabetTex;
    alphabetTex.loadFromFile("res/alphabet.png");
    sf::Sprite alphabetSpr;
    alphabetSpr.setTexture(alphabetTex);
    alphabetSpr.setPosition(613.0f, 0.0f);

    //render texture
    sf::RenderTexture canvas;
    sf::Texture canvasTex;
    sf::Sprite canvasSpr;
    canvas.create(320, 320);
    canvas.clear(sf::Color(0, 0, 0));
    canvasTex = canvas.getTexture();
    canvasSpr.setTexture(canvasTex);
    canvasSpr.setPosition(0.0f, 320.0f);
    canvasSpr.setScale(1.0f, -1.0f);

    //conversion to network input
    sf::Image fullImg, downsample;
    fullImg = canvasTex.copyToImage();
    downsample = sfImgUtil::downsample16(fullImg);
    std::vector<float> input;

    //drawing variables
    bool leftDraw(false), rightDraw(false);
    int lastMouseX(0), lastMouseY(0);

    //shoop da loop
    while (window.isOpen()) {
        sf::Event event;

        while (window.pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                window.close();
            break;

            case sf::Event::KeyPressed:
            {
                switch (event.key.code) {
                case sf::Keyboard::Escape:
                {
                    window.close();
                }
                break;
                case sf::Keyboard::C:
                {
                    canvas.clear(sf::Color(0, 0, 0));

                    canvasTex = canvas.getTexture();
                    fullImg = canvasTex.copyToImage();
                    downsample = sfImgUtil::downsample16(fullImg);
                }
                break;

                default:
                break;
                }
            }
            break;

            case sf::Event::MouseButtonPressed:
            {
                switch (event.mouseButton.button) {
                case sf::Mouse::Button::Left:
                {
                    leftDraw = true;
                }
                break;

                default:
                break;
                }
            }
            break;

            case sf::Event::MouseButtonReleased:
            {
                switch (event.mouseButton.button) {
                case sf::Mouse::Button::Left:
                     leftDraw = false;
                break;

                default:
                break;
                }
            }
            break;

            case sf::Event::MouseMoved:
            {
                if (leftDraw) {
                    sf::CircleShape circle(10);
                    circle.setFillColor(sf::Color(255, 255, 255));

                    sf::Vector2f
                        u(event.mouseMove.x-lastMouseX, event.mouseMove.y-lastMouseY),
                        v(lastMouseX, lastMouseY);
                    float len(sqrtf(u.x*u.x + u.y*u.y));
                    u /= len;
                    for (float ext=0.0f; ext<len; ext+=1.0f) {
                        v+=u;
                        circle.setPosition(v.x-10.0f, v.y-10.0f);
                        canvas.draw(circle);
                    }
                }

                lastMouseX = event.mouseMove.x;
                lastMouseY = event.mouseMove.y;

                canvasTex = canvas.getTexture();
                fullImg = canvasTex.copyToImage();
                downsample = sfImgUtil::downsample16(fullImg);
            }
            break;

            default:
            break;
            }
        }

        //feedforward
        input.clear();
        for (unsigned y=0; y<20; ++y) {
            for (unsigned x=0; x<20; ++x) {
                input.push_back(downsample.getPixel(x, y).r/255.0f);
            }
        }
        nn.setInputValues(input);
        nn.feedForward();

        window.clear();

        IMGSPRITEMACRO(input, input, window, 20, 20, 0.0f, 320.0f, 16.0f, 16.0f);
        IMGSPRITEMACRO(hidden1, nn.getLayerNodeValues(1), window, 20, 10, 330.0f, 0.0f, 10.0f, 10.0f);
        IMGSPRITEMACRO(hidden2, nn.getLayerNodeValues(2), window, 15, 10, 330.0f, 150.0f, 10.0f, 10.0f);
        IMGSPRITEMACRO(hidden3, nn.getLayerNodeValues(3), window, 10, 10, 330.0f, 300.0f, 10.0f, 10.0f);
        IMGSPRITEMACRO(output, nn.getLayerNodeValues(4), window, 1, 26, 600.0f, 0.0f, 13.0f, 13.0f);

        window.draw(canvasSpr);
        window.draw(alphabetSpr);

        window.display();
    }
}

#elif TESTID == 5

void test(void) {
    const unsigned int
        numHorLetters(100),
        numVertLetters(100),
        numValidationEntries(10000);

    LetterDataLoader loader("res/letter/dataset1", 20, 20, 100, 100);

    // load the network
    std::cout << "loading network.. ";
    NeuralNetwork nn("letters.ffnn");
    std::cout << "done." << std::endl;

    //set up the window
    sf::RenderWindow window;
    window.create(sf::VideoMode(1024, 768), "Genetic Neural Networks");

    //training data handlers
    unsigned l(0), validationEntryId(0), numCorrectClassifications(0), numIncorrectClassifications(0);

    //visualisation
    sf::Image inputImg, hidden1Img, hidden2Img, hidden3Img, outputImg, desiredOutputImg;
    sf::Texture inputTex, hidden1Tex, hidden2Tex, hidden3Tex, outputTex, desiredOutputTex;
    sf::Sprite inputSpr, hidden1Spr, hidden2Spr, hidden3Spr, outputSpr, desiredOutputSpr;

    double valMSE = 0.0;

    //shoop da loop
    while (window.isOpen()) {
        sf::Event event;

        while (window.pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                window.close();
            break;

            case sf::Event::KeyPressed:
            {
                switch (event.key.code) {
                case sf::Keyboard::Escape:
                {
                    window.close();
                }
                break;

                default:
                break;
                }
            }
            break;

            default:
            break;
            }
        }

        const std::vector<float>&
            input(loader.getInput(numHorLetters*numVertLetters*l + validationEntryId)),
            output(loader.getOutput(numHorLetters*numVertLetters*l + validationEntryId));
        nn.setInputValues(input);
        nn.feedForward();
        valMSE += nn.getMeanSquareError();

        int classification(-1), desiredClassification(-1);
        double maxVal(0.0), desiredMaxVal(0.0);
        for (unsigned i=0; i<26; ++i) {
            if (nn.getNodeValue(nn.getOutputNodeId(i)) > maxVal) {
                maxVal = nn.getNodeValue(nn.getOutputNodeId(i));
                classification = i;
            }
            if (static_cast<double>(output[i]) > desiredMaxVal) {
                desiredMaxVal = static_cast<double>(output[i]);
                desiredClassification = i;
            }
        }
        if (classification == desiredClassification)
            ++numCorrectClassifications;
        else
            ++numIncorrectClassifications;

        //draw
        window.clear(sf::Color(0, 0, 0));

        IMGSPRITEMACRO(input, input, window, 20, 20, 0.0f, 0.0f, 10.0f, 10.0f);
        IMGSPRITEMACRO(hidden1, nn.getLayerNodeValues(1), window, 20, 10, 250.0f, 0.0f, 10.0f, 10.0f);
        IMGSPRITEMACRO(hidden2, nn.getLayerNodeValues(2), window, 15, 10, 250.0f, 150.0f, 10.0f, 10.0f);
        IMGSPRITEMACRO(hidden3, nn.getLayerNodeValues(3), window, 10, 10, 250.0f, 300.0f, 10.0f, 10.0f);
        IMGSPRITEMACRO(output, nn.getLayerNodeValues(4), window, 1, 26, 600.0f, 0.0f, 10.0f, 10.0f);
        IMGSPRITEMACRO(desiredOutput, output, window, 1, 26, 615.0f, 0.0f, 10.0f, 10.0f);

        window.display();

        if (++l >= 26) {
            if (++validationEntryId >= numValidationEntries)
                window.close();
            l=0;
        }
    }

    valMSE /= numValidationEntries * 26.0f;
    std::cout << "valMSE: " << valMSE << std::endl;

    std::cout << "correct classifications: " << numCorrectClassifications << std::endl;
    std::cout << "incorrect classifications: " << numIncorrectClassifications << std::endl;

    float correctClassificationPercentage = static_cast<float>(numCorrectClassifications)/(numCorrectClassifications + numIncorrectClassifications);
    std::cout << "correct classification percentage: " << correctClassificationPercentage << std::endl;
}

#elif TESTID == 4

void test(void) { // letter network training/saving test
    const unsigned int
        numHorLetters(100),
        numVertLetters(100),
        numTrainingEntries(6000),
        trainingBatchSize(1000),
        numEvaluationEntries(2000),
        numValidationEntries(2000);

    LetterDataLoader loader("res/letter/dataset1", 20, 20, 100, 100);

    // build the network
    std::vector<size_t> layers;
    layers.push_back(199);
    layers.push_back(149);
    layers.push_back(99);
    NGNN nn(0.01, 0.8, 400, 26, layers, -0.5, 0.5);

    //set up the window
    sf::RenderWindow window;
    window.create(sf::VideoMode(1024, 768), "Genetic Neural Networks");

    //training data handlers
    unsigned int trainingEntryId(0);

    //visualisation
    sf::Image inputImg, hidden1Img, hidden2Img, hidden3Img, outputImg, desiredOutputImg;
    sf::Texture inputTex, hidden1Tex, hidden2Tex, hidden3Tex, outputTex, desiredOutputTex;
    sf::Sprite inputSpr, hidden1Spr, hidden2Spr, hidden3Spr, outputSpr, desiredOutputSpr;

    //shoop da loop
    while (window.isOpen()) {
        sf::Event event;

        while (window.pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                window.close();
            break;

            case sf::Event::KeyPressed:
            {
                switch (event.key.code) {
                case sf::Keyboard::Escape:
                {
                    window.close();
                }
                break;

                default:
                break;
                }
            }
            break;

            default:
            break;
            }
        }

        //training
        double preMSE(0.0), postMSE(0.0), evalMSE(0.0);

        for (unsigned int l=0; l<26; ++l) {
            const std::vector<float>&
                input(loader.getInput(numHorLetters*numVertLetters*l + trainingEntryId)),
                output(loader.getOutput(numHorLetters*numVertLetters*l + trainingEntryId));
            nn.setInputValues(input);
            nn.setDesiredOutputValues(output);
            nn.feedForward();
            preMSE += nn.getMeanSquareError();
            nn.backpropagate();
            nn.feedForward();
            postMSE += nn.getMeanSquareError();

            //draw
            window.clear(sf::Color(0, 0, 0));

            IMGSPRITEMACRO(input, input, window, 20, 20, 0.0f, 0.0f, 10.0f, 10.0f);
            IMGSPRITEMACRO(hidden1, nn.getLayerNodeValues(1), window, 20, 10, 250.0f, 0.0f, 10.0f, 10.0f);
            IMGSPRITEMACRO(hidden2, nn.getLayerNodeValues(2), window, 15, 10, 250.0f, 150.0f, 10.0f, 10.0f);
            IMGSPRITEMACRO(hidden3, nn.getLayerNodeValues(3), window, 10, 10, 250.0f, 300.0f, 10.0f, 10.0f);
            IMGSPRITEMACRO(output, nn.getLayerNodeValues(4), window, 1, 26, 600.0f, 0.0f, 10.0f, 10.0f);
            IMGSPRITEMACRO(desiredOutput, output, window, 1, 26, 615.0f, 0.0f, 10.0f, 10.0f);

            window.display();
        }

        preMSE /= 26.0f;
        postMSE /= 26.0f;

        /*std::cout
            << "ID: " << trainingEntryId << std::endl
            << "preMSE: " << preMSE << " postMSE: " << postMSE << std::endl
            << "diff: " << postMSE - preMSE << std::endl;
        */

        ++trainingEntryId;

        //evaluation
        if (trainingEntryId % trainingBatchSize == 0) {
            std::cout << "starting evaluation.. " << std::endl;
            for (unsigned int i=0; i<numEvaluationEntries; ++i) {
                for (unsigned int l=0; l<26; ++l) {
                    nn.setInputValues(loader.getInput(numHorLetters*numVertLetters*l + numTrainingEntries + i));
                    nn.setDesiredOutputValues(loader.getOutput(numHorLetters*numVertLetters*l + numTrainingEntries + i));
                    nn.feedForward();
                    evalMSE += nn.getMeanSquareError();
                }
            }
            evalMSE /= numEvaluationEntries * 26.0f;

            std::cout << "evalMSE: " << evalMSE << std::endl;

            if (evalMSE < 0.005)
                window.close();
        }

        if (trainingEntryId == numTrainingEntries)
            trainingEntryId = 0;
    }

    double valMSE = 0.0;

    std::cout << "starting validation.. " << std::endl;
    for (unsigned int i=0; i<numValidationEntries; ++i) {
        for (unsigned int l=0; l<26; ++l) {
            nn.setInputValues(loader.getInput(numHorLetters*numVertLetters*l + numTrainingEntries + numEvaluationEntries + i));
            nn.setDesiredOutputValues(loader.getOutput(numHorLetters*numVertLetters*l + numTrainingEntries + numEvaluationEntries + i));
            nn.feedForward();
            valMSE += nn.getMeanSquareError();
        }
    }
    valMSE /= numValidationEntries * 26.0f;

    std::cout << "valMSE: " << valMSE << std::endl;

    nn.saveToFile("letters.ffnn");
}

#elif TESTID == 3

void test(void) {
    const unsigned int
        nTrainingRounds(1000),
        nEvaluationRounds(200),
        MSELogMaxSize(512);


    //random number device
    ut::Rand rand;

    //set up the window
    sf::RenderWindow window;
    window.create(sf::VideoMode(1280, 1024), "Genetic Neural Networks");

    // build the network
    std::vector<size_t> layers;
    layers.push_back(15);
    layers.push_back(15);
    layers.push_back(15);
    NGNN nn(0.01, 0.2, 2, 3, layers, -2.0, 2.0);

    //load the image
    sf::Image img;
    img.loadFromFile("res/img/car-rgb.png");
    auto imgSize = img.getSize();

    //some drawing stuff
    sf::Image resultImg, diffImg;
    sf::RenderTexture resultTex;
    sf::Texture diffTex;
    sf::Sprite resultSprite, diffSprite;

    resultImg.create(512, 512, sf::Color(0, 0, 0));
    diffImg.create(512, 512, sf::Color(128, 128, 128));
    resultTex.create(512, 512, false);
    diffTex.create(512, 512);
    resultSprite.setPosition(768.0f, 0.0f);
    diffSprite.setPosition(768.0f, 512.0f);

    //data structs
    std::vector<double> MSELog;
    MSELog.resize(MSELogMaxSize, 0.0);
    std::vector<sf::Vector2u> points = sfImgUtil::getRandomPointsFromBrightness(diffImg, nTrainingRounds);


    //flags
    bool useFullQuality = false;

    //shoop da loop
    while (window.isOpen()) {
        sf::Event event;

        while (window.pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                window.close();
            break;

            case sf::Event::KeyPressed:
            {
                switch (event.key.code) {
                case sf::Keyboard::Escape:
                {
                    window.close();
                }
                break;

                case sf::Keyboard::Q: // toggle full quality
                    useFullQuality = !useFullQuality;
                break;

                default:
                break;
                }
            }
            break;

            default:
            break;
            }
        }

        //training

        double preMSE(0.0), postMSE(0.0), evalMSE(0.0);

        for (unsigned int i=0; i<nTrainingRounds; ++i) {
            float
                x(((float)points[i].x)/imgSize.x),
                y(((float)points[i].y)/imgSize.y);

            sf::Color pix = img.getPixel((unsigned int)(x*imgSize.x), (unsigned int)(y*imgSize.y));

            nn.setInputValue(nn.getInputNodeId(0), x);
            nn.setInputValue(nn.getInputNodeId(1), y);
            nn.setDesiredOutputValue(nn.getOutputNodeId(0), pix.r/255.0f);
            nn.setDesiredOutputValue(nn.getOutputNodeId(1), pix.g/255.0f);
            nn.setDesiredOutputValue(nn.getOutputNodeId(2), pix.b/255.0f);
            nn.feedForward();
            preMSE += nn.getMeanSquareError();
            nn.backpropagate();
            nn.feedForward();
            postMSE += nn.getMeanSquareError();
        }

        preMSE /= nTrainingRounds;
        postMSE /= nTrainingRounds;

        std::cout
            << "preMSE: " << preMSE << " postMSE: " << postMSE << std::endl
            << "diff: " << postMSE - preMSE << std::endl;

        //evaluation
        for (unsigned int i=0; i<nEvaluationRounds; ++i) {
            float
                x(rand.fRand(0.0f, (float)imgSize.x)/imgSize.x),
                y(rand.fRand(0.0f, (float)imgSize.y)/imgSize.y);

            sf::Color pix = img.getPixel((unsigned int)(x*imgSize.x), (unsigned int)(y*imgSize.y));

            nn.setInputValue(nn.getInputNodeId(0), x);
            nn.setInputValue(nn.getInputNodeId(1), y);
            nn.setDesiredOutputValue(nn.getOutputNodeId(0), pix.r/255.0f);
            nn.setDesiredOutputValue(nn.getOutputNodeId(1), pix.g/255.0f);
            nn.setDesiredOutputValue(nn.getOutputNodeId(2), pix.b/255.0f);
            nn.feedForward();
            evalMSE += nn.getMeanSquareError();
        }

        evalMSE /= nEvaluationRounds;

        //  change learning rate according to MSE
        nn.setLearningRate(evalMSE/5.0f);

        MSELog.push_back(evalMSE);
        while (MSELog.size() > MSELogMaxSize)
            MSELog.erase(MSELog.begin());

        std::cout << "Evaluation MSE: " << *(--MSELog.end()) << std::endl;

        sf::VertexArray MSELogGraph(sf::LinesStrip, MSELogMaxSize);
        for (unsigned int i=0; i<MSELogMaxSize; ++i) {
            MSELogGraph[i].position = sf::Vector2f(0.0f + 1.0f*i, 960.0f - 256.0f*MSELog[i]);
            MSELogGraph[i].color = sf::Color(255, 255, 255);
        }

        sf::VertexArray pointsArray(sf::Points, nTrainingRounds);
        for (unsigned int i=0; i<nTrainingRounds; ++i) {
            pointsArray[i].position = sf::Vector2f(768.0f + points[i].x, points[i].y);
            pointsArray[i].color = sf::Color(255, 0, 0);
        }

        //result and difference images
        unsigned int nDivs;
        if (useFullQuality)
            nDivs = 256;
        else
            nDivs = pow(2, abs(log10(MSELog[MSELogMaxSize-1]))+4);
        std::cout << "nDivs: " << nDivs << std::endl;
        nn.drawImage(resultTex, 0.0f, 512.0f, 512.0f, -512.0f, nDivs, nDivs);
        resultSprite.setTexture(resultTex.getTexture());
        resultImg = resultTex.getTexture().copyToImage();
        diffImg = sfImgUtil::blendImages(img, resultImg, sfImgUtil::BLEND_DIFF);
        diffTex.update(diffImg);
        diffSprite.setTexture(diffTex);

        points = sfImgUtil::getRandomPointsFromBrightness(diffImg, nTrainingRounds);

        //draw
        window.clear(sf::Color(0, 0, 0));

        nn.draw(window);

        window.draw(resultSprite);
        window.draw(diffSprite);

        window.draw(MSELogGraph);
        window.draw(pointsArray);

        window.display();
    }
}

#elif TESTID == 2

void test(void) {
    NGNN n(0.01, 0.8, 2, 1, 3, 6, 10, -2.0, 2.0, -0.5, 0.5);
    NGNN n2(n);
    std::cout << "Initial genome: ";
    n2.printGenome();
    XorDataLoader dl(true, time(NULL));

    NGNNEvolver ev(&n2, &dl);
    ev.evolve(10, 100, 0.001, 0.2, 0.000001);
    //ev.mutate(2000, 500, 10, 100, 0.01, 1000);
}

#elif TESTID == 1

void test(void) {
    NGNN nn(0.01, 0.8, 2, 1, 5, 5, 5, -2.0, 2.0, -0.5, 0.5);
    nn.printGenome();
    XorDataLoader loader(true, time(NULL));
    NGNNEvolver ev(&nn, &loader);

    ev.mutate(100, 20, 1000, 1000, 0.1, 1000000);
}

#endif

int main(void) {
    srand(time(NULL));
    test();

    return 0;
}
