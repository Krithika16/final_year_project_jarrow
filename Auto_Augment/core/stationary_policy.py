from Auto_Augment.core.mnist import get_mnist, data_generator, SimpleModel, get_and_compile_model
from Auto_Augment.core.util.supervised_loop import supervised_train_loop
from ray import tune


def run_trial(lr=0.001, batch_size=128, epochs=20):
    print("--------------------")
    print(lr, batch_size, epochs)
    print("--------------------")
    train, val, test = get_mnist()
    model = get_and_compile_model(SimpleModel, lr=lr)
    loss, val_loss, acc, val_acc = supervised_train_loop(model, train, test, data_generator,
                                                         batch_size=batch_size, epochs=epochs, debug=False)
    return max(val_acc)


def training_func(config):
    lr, batch_size, epochs = config['lr'], config['batch_size'], config['epochs']
    val_acc = run_trial(lr, batch_size, epochs)
    tune.report(mean_loss=val_acc)


if __name__ == "__main__":
    from Auto_Augment.core.util import set_memory_growth

    analysis = tune.run(
        training_func,
        config={
            "lr": tune.choice([1e-5, 1e-4, 1e-3, 1e-2]),
            "batch_size": tune.choice([16, 32, 64, 128]),
            "epochs": tune.choice([5, 10, 20])
        },
        resources_per_trial={'gpu': 1}
    )
