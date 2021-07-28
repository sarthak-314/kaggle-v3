def build_model(dropout=0.5, num_dense=4): 
    model = tf.keras.Sequential([
        hub.KerasLayer(TFHUB_URL, trainable=True), 
        tf.keras.layers.Dropout(dropout), 
        tf.keras.layers.Dense(num_dense, kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
    ])
    return model

def load_model(weights_path, num_dense, dropout): 
    start_time = time()
    with STRATEGY.scope(): 
        model = build_model(dropout, num_dense)
        model.build((None, IMG_SIZE, IMG_SIZE, 3)); model.summary()
        model.load_weights(weights) 
        model.layers[0].trainable = True
        for layer in model.layers: layer.trainable = True
        # model.pop(); model.add(tf.keras.layers.Dense(4))
    print(f'{time()-start_time} seconds to load the model')
    return model

def compile_model(model, loss, metrics): 
    model.compile(
        loss=loss,
        metrics=metrics, 
        optimizer=get_ranger(1e-2), 
        steps_per_execution=32,
    )
    print('Model Compiled')