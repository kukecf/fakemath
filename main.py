import tensorflow as tf

import classifier
from solver import solve_and_draw
from meta import TEST_DATA_PATH, TRAIN_DATA_PATH

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(gpu)

# when the data is not prepared...
train_data, test_data = classifier.prepare_dataset(20000)

# when you have prepared the data beforehand...
# train_data = tf.data.experimental.load(TRAIN_DATA_PATH)
# test_data = tf.data.experimental.load(TEST_DATA_PATH)

# when you want to train your model
# model_alt = classifier.train(train_data, 30, batch_size=200, lr=5e-5, alt=True)

# if you need a recovery...
# model_alt = classifier.train(train_data, 10,batch_size=200,lr=6e-5,alt=True,restore=True)

# load a pretrained model
model_alt = classifier.load_best_model()

# test it all out
classifier.test(model_alt, test_data)

# detect expression from image and draw it
solve_and_draw(model_alt, 'notebooks/data/handwritten_ex/20211231_191020.jpg')
