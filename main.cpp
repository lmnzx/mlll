#include "timer.h"
#include <iostream>
#include <mlx/array.h>
#include <mlx/mlx.h>
#include <mlx/ops.h>
#include <mlx/random.h>
#include <mlx/transforms.h>

int main() {
    int num_features = 100;
    int num_examples = 1'000;
    int num_iters = 10'000;
    float learning_rate = 0.1;

    auto w_star = mlx::core::random::normal({num_features});

    auto X = mlx::core::random::normal({num_examples, num_features});

    auto y = mlx::core::matmul(X, w_star) > 0;

    mlx::core::array w = 1e-2 * mlx::core::random::normal({num_features});

    auto loss_fn = [&](mlx::core::array w) {
        auto logits = mlx::core::matmul(X, w);
        auto scale = (1.0f / num_examples);
        return scale *
               mlx::core::sum(mlx::core::logaddexp(mlx::core::array(0.0f), logits) - y * logits);
    };

    auto grad_fn = mlx::core::grad(loss_fn);

    auto tic = timer::time();
    for (int it = 0; it < num_iters; ++it) {
        auto grad = grad_fn(w);
        w = w - learning_rate * grad;
        eval(w);
    }
    auto toc = timer::time();

    auto loss = loss_fn(w);
    auto acc = sum((matmul(X, w) > 0) == y) / num_examples;
    auto throughput = num_iters / timer::seconds(toc - tic);
    std::cout << "Loss " << loss << ", Accuracy, " << acc << ", Throughput " << throughput
              << " (it/s)." << std::endl;
}
