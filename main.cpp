#include "timer.h"
#include <iostream>
#include <matplot/matplot.h>
#include <mlx/array.h>
#include <mlx/mlx.h>
#include <mlx/ops.h>
#include <mlx/random.h>
#include <mlx/transforms.h>
#include <vector>

int main() {
    int num_features = 100;
    int num_examples = 1'000;
    int num_iters = 10'000;
    float learning_rate = 0.1;

    std::vector<double> iterations;
    std::vector<double> losses;
    std::vector<double> accuracies;

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

        if (it % 100 == 0) {
            auto loss = loss_fn(w);
            auto acc = sum((matmul(X, w) > 0) == y) / num_examples;

            iterations.push_back(it);
            losses.push_back(loss.item<float>());
            accuracies.push_back(acc.item<float>());
        }
    }
    auto toc = timer::time();

    auto loss = loss_fn(w);
    auto acc = sum((matmul(X, w) > 0) == y) / num_examples;
    auto throughput = num_iters / timer::seconds(toc - tic);
    std::cout << "Loss " << loss << ", Accuracy, " << acc << ", Throughput " << throughput
              << " (it/s)." << std::endl;

    auto f = matplot::figure();
    f->width(1200);
    f->height(500);
    f->title("Logistic Regression Training Metrics");

    auto ax1 = f->add_subplot(1, 2, 0);
    ax1->plot(iterations, losses)->line_width(2);
    ax1->title("Training Loss");
    ax1->xlabel("Iteration");
    ax1->ylabel("Loss");
    ax1->grid(true);

    auto ax2 = f->add_subplot(1, 2, 1);
    ax2->plot(iterations, accuracies)->line_width(2);
    ax2->title("Training Accuracy");
    ax2->xlabel("Iteration");
    ax2->ylabel("Accuracy");
    ax2->grid(true);

    f->save("training_metrics.png");

    matplot::show();

    return 0;
}
