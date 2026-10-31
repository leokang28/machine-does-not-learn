#include "time.h"
#define __KNN_IMPLEMENTATION__
#include "knn.h"
#define OLIVEC_IMPLEMENTATION
#include "olive.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMG_WIDTH  800
#define IMG_HEIGHT 600

uint32_t img_pixels[IMG_WIDTH * IMG_HEIGHT];

void knn_render(Olivec_Canvas img, KNN nn) {
    size_t neuron_radius = 25;
    uint32_t background_color = 0xFF181818;
    uint32_t low_color = 0x00FF00FF;
    uint32_t high_color = 0x0000FF00;
    
    olivec_fill(img, background_color);
    int layer_border_pady = 50;
    int layer_border_padx = 50;
    int knn_width = img.width - layer_border_padx * 2;
    size_t arch_count = nn.count + 1;
    int layer_padx = knn_width / arch_count;
    int knn_height = img.height - layer_border_pady * 2;
    int layer_x = img.width / 2 - knn_width / 2 ;
    int layer_y = img.height / 2 - knn_height / 2;
    for(size_t l = 0; l < 4; l += 1) {

        int layer_pady = knn_height/nn.activations[l].cols;
        for(size_t i = 0; i < nn.activations[l].cols; i += 1) {
            size_t cx = layer_x + l * layer_padx + layer_padx / 2;
            size_t cy = layer_y + i * layer_pady + layer_pady / 2;
            
            if (l + 1 < arch_count) {
                int layer_pady2 = knn_height/ nn.activations[l+1].cols;
                for(size_t j = 0; j < nn.activations[l + 1].cols; j += 1) {
                    size_t cx_n = layer_x + (l + 1) * layer_padx + layer_padx / 2;
                    size_t cy_n = layer_y + j * layer_pady2 + layer_pady2 / 2;
                    uint32_t alpha = floorf(255.f*sigmoidf(MATRIX_AT(nn.weights[l], i,j)));
                    uint32_t line_color = 0xFF000000|low_color;
                    olivec_blend_color(&line_color, (alpha<<(8*3))|high_color);
                    olivec_line(img, cx, cy, cx_n, cy_n, line_color);
                }
            }

            if (l > 0) {
                uint32_t alpha = floorf(255.f*sigmoidf(MATRIX_AT(nn.biases[l - 1], 0, i)));
                uint32_t neuron_color = 0xFF000000|low_color;
                olivec_blend_color(&neuron_color, (alpha<<(8*3))|high_color);
                olivec_circle(img, cx, cy, neuron_radius, neuron_color);
            } else {
                olivec_circle(img, cx, cy, neuron_radius, 0xFFAAAAAA);
                
            }
        }


    }
}

int main() {
    srand(time(0));
    size_t arch[] = {4,4,2,1};
    size_t arch_count = ARR_LEN(arch);
    KNN nn = knn_alloc(arch, arch_count);
    knn_rand(nn, -10, 10);
    
    const char *img_file_path = "knn.png";
    
    Olivec_Canvas img = olivec_canvas(img_pixels, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);

    knn_render(img, nn);
    size_t frame_thikness = 10;
    uint32_t frame_color = 0xFF282828;
    olivec_frame(img, 0, 0, img.width - 1, img.height - 1, frame_thikness, frame_color);

    if (
        !stbi_write_png(img_file_path, img.width, img.height, 4, img.pixels, img.stride*sizeof(uint32_t))
    ) {
        printf("ERROR: could not save png file %s\n", img_file_path);
        return 1;

    }
    return 0;
}