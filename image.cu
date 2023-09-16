/*
    Base on https://github.com/pjreddie/darknet/blob/a3714d0a2bf92c3dea84c4bea65b2b0c64dbc6b1/src/image.h,
    we customize some functions.
*/
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "header/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "header/stb_image_write.h"

int rand_int(int min, int max)
{
	if (max < min)
	{
		int s = min;
		min = max;
		max = s;
	}

	float rand_r = std::rand();
	int r =	(std::rand() % (max - min + 1)) + min;

	return r;
}

int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

static float get_pixel_extend(image m, int x, int y, int c)
{
    if(x < 0 || x >= m.w || y < 0 || y >= m.h) 
        return 0;

    if(c < 0 || c >= m.c) return 0;
        return get_pixel(m, x, y, c);
}
static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}
static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

static float bilinear_interpolate(image im, float x, float y, int c)
{
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * get_pixel_extend(im, ix, iy, c) +
        dy     * (1-dx) * get_pixel_extend(im, ix, iy+1, c) +
        (1-dy) *   dx   * get_pixel_extend(im, ix+1, iy, c) +
        dy     *   dx   * get_pixel_extend(im, ix+1, iy+1, c);
    return val;
}

image make_empty_image(int w, int h, int c)
{
	image out;
	out.data = nullptr;
	out.h = h;
	out.w = w;
	out.c = c;
	out.label = 0;
	return out;
}

image make_image(int w, int h, int c)
{
	image out = make_empty_image(w, h, c);
	out.data = (float *)calloc(h * w * c, sizeof(float));
	return out;
}

image load_image_stb(const char *filename, int req_channel)
{
	int width, height, channels;
	unsigned char * img = stbi_load(filename, &width, &height, &channels, req_channel);

	if(!img)
	{
		printf("Error in loading the image : %s\n", filename);
		exit(0);
	}

	image im = make_image(width, height, req_channel); // image should be same

	if(channels == 1) // GRAY
	{
		for(int k = 0; k < req_channel; k++) // repeat to require channel
		{
			for(int j = 0; j < height; j++)
			{
				for(int i = 0; i < width; i++)
				{
					int src_index = i + width * j;                              // k + channels * i + channels * width * j;	// 1 2 3 4 5
					int dst_index = i + width * j + width * height * k;			// RRR(1 2 3 4 5) GGG(1 2 3 4 5)  BBB(1 2 3 4 5)

                    im.data[dst_index] = (float)img[src_index];
				}
			}
		}
	}
	else if(channels == 3) // RGB
	{
		for(int k = 0; k < channels; k++)
		{
			for(int j = 0; j < height; j++)
			{
				for(int i = 0; i < width; i++)
				{
					int src_index = k + channels * i + channels * width * j;	// RGBRGBRGB image
					int dst_index = i + width * j + width * height * k;			// RRRGGGBBB image

                    im.data[dst_index] = (float)img[src_index];
				}
			}
		}
	}
	else
	{
		printf("Not allow the channel format %d %s\n", channels, filename);
		exit(-1);
	}

	free(img);

	return im;
}

void free_image(image m)
{
	if(m.data)
	{
		free(m.data);
	}
}

image center_crop_image(image im, int w, int h)
{
    int m = (im.w < im.h) ? im.w : im.h;
    image c = crop_image(im, (im.w - m) / 2, (im.h - m)/2, m, m);
    image r = resize_image(c, w, h);
    free_image(c);
    return r;
}

image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);

    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }

    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}

image crop_image(image im, int dx, int dy, int w, int h)
{
    image cropped = make_image(w, h, im.c);
    int i, j, k;

    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int r = j + dy;
                int c = i + dx;
                float val = 0;
                r = constrain_int(r, 0, im.h-1);
                c = constrain_int(c, 0, im.w-1);
                val = get_pixel(im, c, r, k);
                set_pixel(cropped, i, j, k, val);
            }
        }
    }
    return cropped;
}

image normalize_imagenet(image im, int dx, int dy, int w, int h)
{
    image normalized = make_image(w, h, im.c);
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                float val = get_pixel(im, i, j, k);

                // ToTensor : in Pytorch
                val = val / 255.0f;

                // Normalize : in Pytorch
                if(k == 0) // R
                	set_pixel(normalized, i, j, k, (val - 0.485f) / 0.229f);
                else if(k == 1) // G
                	set_pixel(normalized, i, j, k, (val - 0.456f) / 0.224f);
                else if(k == 2) // B
                	set_pixel(normalized, i, j, k, (val - 0.406f) / 0.225f);
            }
        }
    }
    return normalized;
}

image load_image(const char* file_name, int w, int h, int c) // w = 0, h = 0, c = 3
{
	image out = load_image_stb(file_name, c);

	if((h && w) && (h != out.h || w != out.w))
	{
		image resized = resize_image(out, w, h);
		free_image(out);
		out = resized;
	}

	return out;
}

void flip_image(image a)
{
    int i,j,k;

    for(k = 0; k < a.c; ++k){
        for(i = 0; i < a.h; ++i){
            for(j = 0; j < a.w/2; ++j){
                int index = j + a.w*(i + a.h*(k));
                int flip = (a.w - j - 1) + a.w*(i + a.h*(k));
                float swap = a.data[flip];
                a.data[flip] = a.data[index];
                a.data[index] = swap;
            }
        }
    }
}

image random_crop_image(image im, int w, int h)
{
	int dx = rand_int(0, im.w - w);
	int dy = rand_int(0, im.h - h);

	image crop = crop_image(im, dx, dy, w, h);

	return crop;
}
