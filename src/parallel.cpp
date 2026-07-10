#define CL_TARGET_OPENCL_VERSION 300
#include "parallel.h"
#include <CL/cl.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include "LBMsys.h"
using namespace std;

#define SIMPLE_CHECK_ERRORS(ERR) \
    if(ERR != CL_SUCCESS){ \
        cerr << "OpenCL error " << ERR << " at line " << __LINE__ << endl; exit(1); \
    }

// Helper function for build errors
void checkBuildErrors(cl_program program, cl_device_id device) {
    cl_build_status status;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL);
    if (status != CL_BUILD_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        cerr << "Build error: " << log << endl;
        free(log);
        exit(1);
    }
}

std::string readKernelFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open kernel file: " + filename);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int parallel(int STEPS, int REFINE) {
    // ---- LBM parameters ----
    LBMsys lbm;
    lbm.loadConfig("config.txt");
    lbm.refineConfig(REFINE);
    lbm.initialize(0.05f, 0.0f);  // example inlet velocity

    const int N = lbm.getSize();
    const int NX = lbm.getWidth();
    const int NY = lbm.getHeight();

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    // ---- Platform and device selection ----
    err = clGetPlatformIDs(1, &platform, NULL); SIMPLE_CHECK_ERRORS(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); SIMPLE_CHECK_ERRORS(err);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); SIMPLE_CHECK_ERRORS(err);
    queue = clCreateCommandQueue(context, device, 0, &err); SIMPLE_CHECK_ERRORS(err);

    // ---- Buffers ----
    cl_mem f_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 9 * N, NULL, &err); SIMPLE_CHECK_ERRORS(err);
    cl_mem f_new_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 9 * N, NULL, &err); SIMPLE_CHECK_ERRORS(err);
    cl_mem rho_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &err); SIMPLE_CHECK_ERRORS(err);
    cl_mem ux_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &err); SIMPLE_CHECK_ERRORS(err);
    cl_mem uy_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &err); SIMPLE_CHECK_ERRORS(err);
    cl_mem flags_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * N, lbm.getFlags().data(), &err); SIMPLE_CHECK_ERRORS(err);

    // ---- Initialize f on device ----
    err = clEnqueueWriteBuffer(queue, f_buf, CL_TRUE, 0, sizeof(float) * 9 * N, lbm.getF().data(), 0, NULL, NULL); SIMPLE_CHECK_ERRORS(err);
    err = clEnqueueWriteBuffer(queue, ux_buf, CL_TRUE, 0, sizeof(float) * N, lbm.getUx().data(), 0, NULL, NULL); SIMPLE_CHECK_ERRORS(err);
    err = clEnqueueWriteBuffer(queue, uy_buf, CL_TRUE, 0, sizeof(float) * N, lbm.getUy().data(), 0, NULL, NULL); SIMPLE_CHECK_ERRORS(err);

    // ---- Ucitavanje kernela ----
    std::string collide_src = readKernelFromFile("collide_kernel.cl");
    std::string stream_src = readKernelFromFile("stream_kernel.cl");
    std::string macro_src = readKernelFromFile("computeMacros_kernel.cl");
    std::string bb_src = readKernelFromFile("bounceBack_kernel.cl");
    std::string bc_src = readKernelFromFile("boundaryConditions_kernel.cl");

    const char* collide_kernel_src = collide_src.c_str();
    const char* stream_kernel_src = stream_src.c_str();
    const char* macro_kernel_src = macro_src.c_str();
    const char* bb_kernel_src = bb_src.c_str();
    const char* bc_kernel_src = bc_src.c_str();

    // ---- Compile kernels ----
    cl_program collide_program = clCreateProgramWithSource(context, 1, &collide_kernel_src, NULL, &err); SIMPLE_CHECK_ERRORS(err);
    err = clBuildProgram(collide_program, 1, &device, NULL, NULL, NULL);
    checkBuildErrors(collide_program, device);
    cl_kernel collide_kernel = clCreateKernel(collide_program, "collide", &err); SIMPLE_CHECK_ERRORS(err);

    cl_program stream_program = clCreateProgramWithSource(context, 1, &stream_kernel_src, NULL, &err); SIMPLE_CHECK_ERRORS(err);
    err = clBuildProgram(stream_program, 1, &device, NULL, NULL, NULL);
    checkBuildErrors(stream_program, device);
    cl_kernel stream_kernel = clCreateKernel(stream_program, "stream", &err); SIMPLE_CHECK_ERRORS(err);

    cl_program macro_program = clCreateProgramWithSource(context, 1, &macro_kernel_src, NULL, &err); SIMPLE_CHECK_ERRORS(err);
    err = clBuildProgram(macro_program, 1, &device, NULL, NULL, NULL);
    checkBuildErrors(macro_program, device);
    cl_kernel macro_kernel = clCreateKernel(macro_program, "computeMacros", &err); SIMPLE_CHECK_ERRORS(err);

    cl_program bb_program = clCreateProgramWithSource(context, 1, &bb_kernel_src, NULL, &err); SIMPLE_CHECK_ERRORS(err);
    err = clBuildProgram(bb_program, 1, &device, NULL, NULL, NULL);
    checkBuildErrors(bb_program, device);
    cl_kernel bb_kernel = clCreateKernel(bb_program, "bounceBack", &err); SIMPLE_CHECK_ERRORS(err);

    cl_program bc_program = clCreateProgramWithSource(context, 1, &bc_kernel_src, NULL, &err); SIMPLE_CHECK_ERRORS(err);
    err = clBuildProgram(bc_program, 1, &device, NULL, NULL, NULL);
    checkBuildErrors(bc_program, device);
    cl_kernel bc_kernel = clCreateKernel(bc_program, "boundaryConditions", &err); SIMPLE_CHECK_ERRORS(err);

    // ---- Simulation loop ----
    size_t global_size_2D[2] = { (size_t)NX, (size_t)NY };
    const float omega = 0.6f;
    float rho_out = 1.0f;
    float u_in_x = 0.2f;
    float u_in_y = 0.0f;

    // Proper double buffering initialization
    cl_mem f_current = f_buf;      // Current state
    cl_mem f_next = f_new_buf;     // Next state

    vector<float> host_ux(N);
    vector<float> host_uy(N);


    // Initialize distributions to equilibrium
    err = clEnqueueWriteBuffer(queue, f_current, CL_TRUE, 0, sizeof(float) * 9 * N, lbm.getF().data(), 0, NULL, NULL);
    SIMPLE_CHECK_ERRORS(err);

    for (int step = 0; step < STEPS; step++) {
        // 1. Streaming step
        err = clSetKernelArg(stream_kernel, 0, sizeof(cl_mem), &f_current);
        err = clSetKernelArg(stream_kernel, 1, sizeof(cl_mem), &f_next);
        err = clSetKernelArg(stream_kernel, 2, sizeof(cl_mem), &flags_buf);
        err = clSetKernelArg(stream_kernel, 3, sizeof(int), &NX);
        err = clSetKernelArg(stream_kernel, 4, sizeof(int), &NY);
        err = clEnqueueNDRangeKernel(queue, stream_kernel, 2, NULL, global_size_2D, NULL, 0, NULL, NULL);
        SIMPLE_CHECK_ERRORS(err);

        // 2. Boundary conditions (inlet/outlet) - PRVO!
        err = clSetKernelArg(bc_kernel, 0, sizeof(cl_mem), &f_next);
        err = clSetKernelArg(bc_kernel, 1, sizeof(cl_mem), &ux_buf);
        err = clSetKernelArg(bc_kernel, 2, sizeof(cl_mem), &uy_buf);
        err = clSetKernelArg(bc_kernel, 3, sizeof(cl_mem), &flags_buf);
        err = clSetKernelArg(bc_kernel, 4, sizeof(float), &u_in_x);
        err = clSetKernelArg(bc_kernel, 5, sizeof(float), &u_in_y);
        err = clSetKernelArg(bc_kernel, 6, sizeof(float), &rho_out);
        err = clSetKernelArg(bc_kernel, 7, sizeof(int), &NX);
        err = clSetKernelArg(bc_kernel, 8, sizeof(int), &NY);
        err = clEnqueueNDRangeKernel(queue, bc_kernel, 2, NULL, global_size_2D, NULL, 0, NULL, NULL);
        SIMPLE_CHECK_ERRORS(err);

        // 3. Bounce-back (solid boundaries)
        err = clSetKernelArg(bb_kernel, 0, sizeof(cl_mem), &f_current);
        err = clSetKernelArg(bb_kernel, 1, sizeof(cl_mem), &f_next);
        err = clSetKernelArg(bb_kernel, 2, sizeof(cl_mem), &flags_buf);
        err = clSetKernelArg(bb_kernel, 3, sizeof(int), &NX);
        err = clSetKernelArg(bb_kernel, 4, sizeof(int), &NY);
        err = clEnqueueNDRangeKernel(queue, bb_kernel, 2, NULL, global_size_2D, NULL, 0, NULL, NULL);
        SIMPLE_CHECK_ERRORS(err);

        // 4. Compute macroscopic variables (rho, ux, uy)
        err = clSetKernelArg(macro_kernel, 0, sizeof(cl_mem), &f_next);
        err = clSetKernelArg(macro_kernel, 1, sizeof(cl_mem), &rho_buf);
        err = clSetKernelArg(macro_kernel, 2, sizeof(cl_mem), &ux_buf);
        err = clSetKernelArg(macro_kernel, 3, sizeof(cl_mem), &uy_buf);
        err = clSetKernelArg(macro_kernel, 4, sizeof(cl_mem), &flags_buf);
        err = clSetKernelArg(macro_kernel, 5, sizeof(int), &NX);
        err = clSetKernelArg(macro_kernel, 6, sizeof(int), &NY);
        err = clEnqueueNDRangeKernel(queue, macro_kernel, 2, NULL, global_size_2D, NULL, 0, NULL, NULL);
        SIMPLE_CHECK_ERRORS(err);

        // 5. Collision step - ISPRAVNO!
        err = clSetKernelArg(collide_kernel, 0, sizeof(cl_mem), &f_current);  // f kao IZLAZ
        err = clSetKernelArg(collide_kernel, 1, sizeof(cl_mem), &f_next);     // f_new kao ULAZ
        err = clSetKernelArg(collide_kernel, 2, sizeof(cl_mem), &rho_buf);
        err = clSetKernelArg(collide_kernel, 3, sizeof(cl_mem), &ux_buf);
        err = clSetKernelArg(collide_kernel, 4, sizeof(cl_mem), &uy_buf);
        err = clSetKernelArg(collide_kernel, 5, sizeof(cl_mem), &flags_buf);
        err = clSetKernelArg(collide_kernel, 6, sizeof(float), &omega);
        err = clSetKernelArg(collide_kernel, 7, sizeof(int), &NX);
        err = clSetKernelArg(collide_kernel, 8, sizeof(int), &NY);
        err = clEnqueueNDRangeKernel(queue, collide_kernel, 2, NULL, global_size_2D, NULL, 0, NULL, NULL);
        SIMPLE_CHECK_ERRORS(err);

        // Synchronize
        clFinish(queue);

        // Vizualizacija
        if (step % 50 == 0) {
            err = clEnqueueReadBuffer(queue, ux_buf, CL_TRUE, 0, sizeof(float) * N, host_ux.data(), 0, NULL, NULL);
            SIMPLE_CHECK_ERRORS(err);
            err = clEnqueueReadBuffer(queue, uy_buf, CL_TRUE, 0, sizeof(float) * N, host_uy.data(), 0, NULL, NULL);
            SIMPLE_CHECK_ERRORS(err);

            cv::Mat img(NY, NX, CV_8UC3, cv::Scalar(255, 255, 255));

            for (int y = 0; y < NY; y++) {
                for (int x = 0; x < NX; x++) {
                    int node = lbm.node_index(x, y);

                    if (lbm.flag_at(node) == 1) {
                        img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                        continue;
                    }
                    if (lbm.flag_at(node) == 2) {
                        img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
                        continue;
                    }
                    if (lbm.flag_at(node) == 3) {
                        img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
                        continue;
                    }

                    float ux = host_ux[node];
                    float uy = host_uy[node];
                    float speed = std::sqrt(ux * ux + uy * uy);

                    int val = std::min(255, (int)(speed * 2000)); 
                    cv::Vec3b color = cv::Vec3b(255 - val, 0, val);

                    img.at<cv::Vec3b>(y, x) = color;
                }
            }

            imshow("Fluid Flow - OpenCL", img);

            if (cv::waitKey(1) == 27) {
                std::cout << "Simulation stopped by user" << endl;
                break;
            }
        }
    }

    clEnqueueCopyBuffer(queue, f_next, f_buf, 0, 0, sizeof(float) * 9 * N, 0, NULL, NULL);
    clFinish(queue);

    std::cout << "Simulation finished." << endl;

    // ---- Complete Cleanup ----
    clReleaseMemObject(f_buf);
    clReleaseMemObject(f_new_buf);
    clReleaseMemObject(rho_buf);
    clReleaseMemObject(ux_buf);
    clReleaseMemObject(uy_buf);
    clReleaseMemObject(flags_buf);

    clReleaseKernel(collide_kernel);
    clReleaseKernel(stream_kernel);
    clReleaseKernel(macro_kernel);
    clReleaseKernel(bb_kernel);
    clReleaseKernel(bc_kernel);

    clReleaseProgram(collide_program);
    clReleaseProgram(stream_program);
    clReleaseProgram(macro_program);
    clReleaseProgram(bb_program);
    clReleaseProgram(bc_program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}