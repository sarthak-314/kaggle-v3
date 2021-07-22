def get_lr_callback(lr_min=1e-6, lr_warmup_epochs=10, lr_max=1e-3, epochs=25):
    def lrfn(epoch): 
        EXP_DECAY = 0.9
        if epoch < lr_warmup_epochs: 
            lr = (lr_max-lr_min) / lr_warmup_epochs * epoch + lr_min
        else: 
            lr = (lr_max-lr_min) * EXP_DECAY ** (epoch-lr_warmup_epochs) + lr_min
        return lr
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
    rng = [i for i in range(epochs)]
    y = [lrfn(x) for x in rng]
    plt.plot(rng, y)
    print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
    return lr_callback