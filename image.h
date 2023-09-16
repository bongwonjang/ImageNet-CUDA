/*
    Base on https://github.com/pjreddie/darknet/blob/a3714d0a2bf92c3dea84c4bea65b2b0c64dbc6b1/src/image.h,
    we customize some functions.
*/
#ifndef IMAGE_H_
#define IMAGE_H_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <random>

typedef struct {
    int w;
    int h;
    int c;
    float *data;
    int label;
} image;

/* Utility functions */
int rand_int(int min, int max);
int constrain_int(int a, int min, int max);
static float get_pixel(image m, int x, int y, int c);
static float get_pixel_extend(image m, int x, int y, int c);
static void set_pixel(image m, int x, int y, int c, float val);
static void add_pixel(image m, int x, int y, int c, float val);
static float bilinear_interpolate(image im, float x, float y, int c);

/* Image data create/free functions */
image make_empty_image(int w, int h, int c);
image make_image(int w, int h, int c);
image load_image(const char* file_name, int w = 0, int h = 0, int c = 3);
image load_image_stb(const char *filename);
void free_image(image m);

/* Image processing functions */
image center_crop_image(image im, int w, int h);
image resize_image(image im, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image normalize_imagenet(image im, int dx, int dy, int w, int h);
void flip_image(image a);
image random_crop_image(image im, int w, int h);

#endif /* IMAGE_H_ */
