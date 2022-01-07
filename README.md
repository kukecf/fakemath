**BEFORE YOU USE!**

Download the dataset zip file from https://www.kaggle.com/michelheusser/handwritten-digits-and-operators and unpack it in 'notebooks/data/dataset' directory. Create 'notebooks/data/test' and 'notebooks/data/train' directories. 


GENERAL INFO

The models are stored in notebook/models. The notebook is a WIP,
but the modules are functional. The most important functions of each module can be tested by uncommenting them within
the module.

meta.py stores all constants, such as path constants, image sizes etc.

The detector.py module's main function is `detect_characters(image_path)` which is used to get bounding boxes of all the
digits and operators for a given image path.

classifier.py contains functions for training, testing or getting a pretrained model. To train a model, first call the
function `prepare_dataset(n)`, where n is an integer representing the number of representative examples for each class.
This function returns train and test data with evenly distributed examples for each class. 

Then, call `train(data,epochs)` with the given train data (alt is an optional parameter and is used to train the alternate model). To test, call `test(model,data)` with the
model and test data. `get_best_model()` gets the best pretrained model acquired so far. `get_latest_model(alt=True)` returns last
models trained with alt as the optional parameter for the alternate model (defaults to True due to performing better on
the test set).

Finally, the solver has its own eval-like function implementation evaluate_expression(exp). solve_and_draw(model,
image_path) draws a bounding box around an expression in the image, prints predictions of the model in the top left
corner and a solution on the right (if it is parsable).

The results of the network could be better if we were more careful about the way we write (e.g. when we write x for multiplication, but the dataset says *, or () instead of []).
Of course, we should also write prettier :)

Possible improvements (~99% accuracy, ~90% precision and recall):
- Better dataset
- Take even more data samples for training
- Try to create a deeper model which would perform better
- Experiment with regularizations and learning rate
- Enlarging images to detect more details
- Add webcam support with cv.VideoCapture
- Add web interface
- Dockerize solution

