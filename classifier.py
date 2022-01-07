import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.metrics import Precision, Recall, Accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
from detector import get_digit_images
from meta import *


def prepare_dataset(class_ds_size):
    for idx, label in enumerate(labels):
        ds = tf.data.Dataset.list_files(os.path.join(DATA_PATH, label, "*.png")).take(class_ds_size)
        labelling = tf.data.Dataset.from_tensor_slices(tf.ones(len(ds)) * idx, name='labels')
        data = tf.data.Dataset.zip((ds, labelling))

        data = data.map(training_preprocess)
        data = data.shuffle(buffer_size=1024)
        if idx == 0:
            train_data = data.take(int(len(data) * .7))
            test_data = data.skip(int(len(data) * .7))
            test_data = test_data.take(int(len(data) * .3))
        else:
            train_data = train_data.concatenate(data.take(int(len(data) * .7)))
            test = data.skip(int(len(data) * .7))
            test_data = test_data.concatenate(test.take(int(len(data) * .3)))
        train_data.cache()
        test_data.cache()
    tf.data.experimental.save(train_data, TRAIN_DATA_PATH)
    tf.data.experimental.save(test_data, TEST_DATA_PATH)
    print(f'Train data length:{len(train_data)}, test data length: {len(test_data)}')
    return train_data, test_data


# dataset : https://www.kaggle.com/michelheusser/handwritten-digits-and-operators
def training_preprocess(image_path, label):
    byte_image = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(byte_image)
    img = img / 255
    img = tf.image.resize(img, INPUT_IMAGE_SIZE)
    return img, label


def create_digit_classifier():
    in_size = INPUT_IMAGE_SIZE
    input_layer = Input(shape=(in_size[0], in_size[1], 1), name='input')

    b_norm = BatchNormalization(momentum=0.8)(input_layer)

    conv_1 = Conv2D(64, (4, 4), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(b_norm)
    max_1 = MaxPooling2D(64, (2, 2), padding='same')(conv_1)

    conv_2 = Conv2D(64, (3, 3), activation='relu')(max_1)
    max_2 = MaxPooling2D(32, (2, 2), padding='same')(conv_2)
    do_1 = Dropout(0.2)(max_2)

    conv_3 = Conv2D(128, (4, 4), activation='relu', bias_regularizer=tf.keras.regularizers.l2(5e-5),
                    kernel_regularizer=tf.keras.regularizers.l2(1e-5))(do_1)
    max_3 = MaxPooling2D(16, (2, 2), padding='same')(conv_3)
    do_2 = Dropout(0.2)(max_3)

    conv_4 = Conv2D(128, (2, 2), activation='relu')(do_2)
    max_4 = MaxPooling2D(16, (2, 2), padding='same')(conv_4)
    do_3 = Dropout(0.2)(max_4)

    flat_1 = Flatten()(do_3)
    dense_1 = Dense(16, activation='softmax',
                    bias_regularizer=tf.keras.regularizers.l2(1e-4),
                    activity_regularizer=tf.keras.regularizers.l2(1e-5))(flat_1)

    return Model(inputs=[input_layer], outputs=[dense_1], name='digit_classifier')


def create_deeper_digit_classifier():
    in_size = INPUT_IMAGE_SIZE
    input = Input(shape=(in_size[0], in_size[1], 1), name='input')

    conv_1 = Conv2D(64, (4, 4), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(
        input)
    max_1 = MaxPooling2D(64, (2, 2), padding='same')(conv_1)

    b_norm_1 = BatchNormalization(momentum=0.8)(max_1)

    conv_2 = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(b_norm_1)
    max_2 = MaxPooling2D(32, (2, 2), padding='same')(conv_2)
    do_1 = Dropout(0.12)(max_2)

    conv_3 = Conv2D(128, (4, 4), activation='relu', padding='same', bias_regularizer=tf.keras.regularizers.l2(5e-5),
                    kernel_regularizer=tf.keras.regularizers.l2(1e-5))(do_1)
    max_3 = MaxPooling2D(32, (2, 2), padding='same')(conv_3)
    do_2 = Dropout(0.2)(max_3)

    conv_4 = Conv2D(128, (2, 2), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(do_2)
    max_4 = MaxPooling2D(32, (2, 2), padding='same')(conv_4)
    do_3 = Dropout(0.2)(max_4)

    flat = Flatten()(do_3)
    dense_1 = Dense(512, activation='relu',
                    bias_regularizer=tf.keras.regularizers.l2(1e-4),
                    activity_regularizer=tf.keras.regularizers.l2(1e-5))(flat)

    dense_2 = Dense(128, activation='relu',
                    bias_regularizer=tf.keras.regularizers.l2(1e-5))(dense_1)
    b_norm_2 = BatchNormalization(momentum=0.8)(dense_2)

    dense_3 = Dense(64, activation='relu',
                    bias_regularizer=tf.keras.regularizers.l2(1e-5))(b_norm_2)
    do_4 = Dropout(0.2)(dense_3)

    dense_4 = Dense(N_CLASSES, activation='softmax',
                    bias_regularizer=tf.keras.regularizers.l2(1e-5),
                    activity_regularizer=tf.keras.regularizers.l2(1e-5))(do_4)

    return Model(inputs=[input], outputs=[dense_4], name='digit_classifier')


@tf.function
def train_step(model, batch, loss, optimizer):
    with tf.GradientTape() as tape:
        X = batch[0]
        y_true = batch[1]
        y_pred = model(X, training=True)
        loss_v = loss(y_true, y_pred)
        grad = tape.gradient(loss_v, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return loss_v


def train(data, epochs, batch_size=200, prefetch=40, lr=4e-5, alt=False, restore=False):
    if alt:
        model = create_deeper_digit_classifier()
    else:
        model = create_digit_classifier()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=30000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    checkpoint_dir = "./checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, 'chkpt')
    checkpoint = tf.train.Checkpoint(opt=optimizer, model=model)
    chkpt_save_path = tf.train.latest_checkpoint(checkpoint_dir)
    if restore and chkpt_save_path is not None:
        checkpoint.restore(chkpt_save_path)
        print('Restored!')
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        data = data.shuffle(buffer_size=data.cardinality().numpy())
        batched = data.batch(batch_size)
        batched = batched.prefetch(prefetch)  # to prevent bottlenecking
        pb = Progbar(len(batched), stateful_metrics=['loss'])
        for batch in batched:
            loss_v = train_step(model, batch, loss, optimizer)
            pb.add(1, values=[('loss', loss_v.numpy())])
        if epoch % 10 == 0:
            print("New checkpoint: ", checkpoint.save(file_prefix=checkpoint_prefix))
    if alt:
        model.save(os.path.join(MODELS_PATH, 'digit_classifier_alt.h5'))
    else:
        model.save(os.path.join(MODELS_PATH, 'digit_classifier.h5'))
    return model


def test(model, data):
    print(data.cardinality().numpy())
    data = data.batch(16)
    data = data.prefetch(8)
    recall = Recall()
    precision = Precision()
    accuracy = Accuracy()
    for idx, batch in enumerate(data):
        X = batch[0]
        y_true = tf.cast(batch[1], dtype=tf.int32)
        y_out = model.predict(X)
        y_pred = tf.math.argmax(y_out, axis=1)
        yt_oh = tf.one_hot(y_true, N_CLASSES)
        yp_oh = tf.one_hot(y_pred, N_CLASSES)
        recall.update_state(yt_oh, yp_oh)
        precision.update_state(yt_oh, yp_oh)
        accuracy.update_state(yt_oh, yp_oh)
        print(f'Batch {idx}')
        print(f'True: {y_true}\nPred: {y_pred}')
        print(f'Accuracy: {accuracy.result().numpy()}')
        print(f'Precision: {precision.result().numpy()}, recall: {recall.result().numpy()}\n')


def load_best_model():
    return tf.keras.models.load_model(os.path.join(MODELS_PATH, 'digit_classifier_best.h5'))


def load_latest_model(alt=False):
    if alt:
        return tf.keras.models.load_model(os.path.join(MODELS_PATH, 'digit_classifier_alt.h5'))
    return tf.keras.models.load_model(os.path.join(MODELS_PATH, 'digit_classifier.h5'))


def get_expression(model, image_path, kernel=(5, 5)):
    digit_images = get_digit_images(image_path, kernel)
    y_out = model.predict(digit_images)
    y_pred = tf.math.argmax(y_out, axis=1)
    expression = ""
    for y in y_pred:
        expression += labels[y]
    return expression
