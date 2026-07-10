#include "serial.h"
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "LBMsys.h"

void serial(int STEPS, int REFINE) {
    try {
        LBMsys sys;
        sys.loadConfig("config.txt");
        sys.refineConfig(50);
        sys.initialize(0.05f, 0.0f);

        int NX = sys.getWidth();
        int NY = sys.getHeight();
        std::cout << NX * NY << "\n";

        float omega = 0.2f;
        float u_in_x = 0.5f;
        float u_in_y = 0.0f;

        for (int t = 0; t < STEPS; t++) {
            sys.step(omega, u_in_x, u_in_y);
            if (t % 50 == 0) {
                cv::Mat img(NY, NX, CV_8UC3);

                for (int y = 0; y < NY; y++) {
                    for (int x = 0; x < NX; x++) {
                        int node = sys.node_index(x, y);

                        if (sys.flag_at(node) == 1) {
                            img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                            continue;
                        }
                        if (sys.flag_at(node) == 2) {
                            img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
                            continue;
                        }
                        if (sys.flag_at(node) == 3) {
                            img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
                            continue;
                        }

                        float ux = sys.ux_at(node);
                        float uy = sys.uy_at(node);
                        float speed = std::sqrt(ux * ux + uy * uy);

                        int val = std::min(255, (int)(speed * 800));
                        cv::Vec3b color = cv::Vec3b(255 - val, 0, val);

                        img.at<cv::Vec3b>(y, x) = color;
                    }
                }

                cv::imshow("Fluid speed", img);

                if (cv::waitKey(1) == 27) break;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}
