#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>

#ifndef IMAGE_SIZE
#define IMAGE_SIZE 784
#endif

static int predict_single(const float *pixels, const float *weights, const float *bias) {
    float max_score = -1e9f;
    int best_class = 0;

    for (int digit = 0; digit < 10; digit++) {
        float score = (bias != NULL) ? bias[digit] : 0.0f;
        for (int p = 0; p < IMAGE_SIZE; p++) {
            score += weights[digit * IMAGE_SIZE + p] * pixels[p];
        }
        if (score > max_score) {
            max_score = score;
            best_class = digit;
        }
    }
    return best_class;
}

__attribute__((visibility("default"))) void run_inference(
    const float *pixels,
    const float *weights,
    const float *bias,
    const char *image_path
) {
    int predicted_digit = predict_single(pixels, weights, bias);
    printf("\n[C - Inferência] Classe predita para a amostra: %d\n", predicted_digit);

    pid_t pid = fork();
    if (pid == 0) {
        if (image_path && image_path[0] != '\0') {
            execlp("python3", "python3", "display_emat.py", image_path, (char *)NULL);
        }
        exit(0);
    } else if (pid > 0) {
        waitpid(pid, NULL, 0);
    }
}

__attribute__((visibility("default"))) float run_test_dataset(
    const float *weights,
    const float *bias,
    const float *test_images,
    const int *test_labels,
    int num_samples
) {
    if (num_samples <= 0) return 0.0f;

    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        const float *current_pixels = &test_images[i * IMAGE_SIZE];
        int predicted = predict_single(current_pixels, weights, bias);
        if (predicted == test_labels[i]) {
            correct++;
        }
    }

    float accuracy = (float)correct / num_samples * 100.0f;
    printf("\n[C - Matmul Test] Acurácia no conjunto de teste (%d amostras): %.2f%%\n", num_samples, accuracy);
    return accuracy;
}