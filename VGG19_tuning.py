import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense,BatchNormalization, Activation
from keras.models import Sequential
from keras import optimizers
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
import operator
from utils import load_images,find_fbetascore_threshold_class,find_f2score_threshold_global

def generate_predictions_file(probabilities, thresholds,label_list):


    predictions_labels = []
    for prob in probabilities:
        labels = [label_list[i] for i, value in enumerate(prob) if value > thresholds[i]]
        predictions_labels.append(' '.join(labels))

    # Prepare to write predictions to file
    image_files = os.listdir('C:/planet/test-jpg/')
    if 'Thumbs.db' in image_files:
        image_files.remove('Thumbs.db')

    predictions_df_dict = {'image_name': [w.replace('.jpg', '') for w in image_files],
                       'tags': ['' for i in range(len(image_files))],
                       }
    predictions_df = pd.DataFrame.from_dict(predictions_df_dict)
    predictions_df.set_index('image_name', drop=True, inplace=True)
    predictions_df.loc[[w.replace('.jpg', '') for w in image_files], 'tags'] = predictions_labels
    predictions_df.to_csv('predictions_single_class.csv', encoding='utf-8', index=True)

    return

def generate_predictions_file_global_thre(probabilities, threshold_alternative,label_list):


    predictions_labels = []
    for prob in probabilities:
        labels = [label_list[i] for i, value in enumerate(prob) if value > threshold_alternative]
        predictions_labels.append(' '.join(labels))

    # Prepare to write predictions to file
    image_files = os.listdir('C:/planet/test-jpg/')
    if 'Thumbs.db' in image_files:
        image_files.remove('Thumbs.db')

    predictions_df_dict = {'image_name': [w.replace('.jpg', '') for w in image_files],
                       'tags': ['' for i in range(len(image_files))],
                       }
    predictions_df = pd.DataFrame.from_dict(predictions_df_dict)
    predictions_df.set_index('image_name', drop=True, inplace=True)
    print(predictions_df[:5])
    predictions_df.loc[[w.replace('.jpg', '') for w in image_files], 'tags'] = predictions_labels
    predictions_df.to_csv('predictions_single.csv', encoding='utf-8', index=True)

    predictions_labels = []
    for prob in probabilities:
        labels = [label_list[i] for i, value in enumerate(prob) if value > 0.15]
        predictions_labels.append(' '.join(labels))

    # Prepare to write predictions to file
    image_files = os.listdir('C:/planet/test-jpg/')
    if 'Thumbs.db' in image_files:
        image_files.remove('Thumbs.db')

    predictions_df_dict = {'image_name': [w.replace('.jpg', '') for w in image_files],
                       'tags': ['' for i in range(len(image_files))],
                       }
    predictions_df = pd.DataFrame.from_dict(predictions_df_dict)
    predictions_df.set_index('image_name', drop=True, inplace=True)
    print(predictions_df[:5])
    predictions_df.loc[[w.replace('.jpg', '') for w in image_files], 'tags'] = predictions_labels
    predictions_df.to_csv('predictions_single_0.2.csv', encoding='utf-8', index=True)

    return


def run_model_tuning(df,arch,tag,augmented,image_folder,norm,filetype,size,channels,label_list):

    data_set, data_labels = load_images('all', df, 'all', arch,augmented,image_folder,norm,filetype,size,channels)
    print(data_set.shape)
    print(data_labels.shape)

    nsplit=5

    kf = KFold(n_splits=nsplit)
    fold = 0
    for train_index, test_index in kf.split(data_set):
        train_set, train_labels = data_set[train_index], data_labels[train_index]
        valid_set, valid_labels = data_set[test_index], data_labels[test_index]

        weight_file = 'C:/planet/saved_models/weights.best_' + tag + '_' + 'VGG19_tune' + '_' + str(0) + '.hdf5'

        # Create the model
        from keras.applications.vgg19 import VGG19

        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(size,size,channels))

        # VGG19 tuning is done in two stages, in stage one
        # we just tune the top, and in stage two we tune
        # all the layers together
        if arch=='VGG19_tune':
            for layer in base_model.layers[:22]:
                print(layer.name)
                layer.trainable = False

        model = Sequential()
        model.add(Flatten(input_shape=(4,4,512),name='one'))
        model.add(Dense(1024, activation='relu', name='three'))
        model.add(Dense(17, name='seven'))
        model.add(Activation('sigmoid', name='nine'))
        model.summary()

        final_model = Sequential()
        final_model.add(BatchNormalization(input_shape=data_set.shape[1:]))
        final_model.add(base_model)
        final_model.add(model)
        final_model.summary()

        #Very fine tuning of all the layers
        final_model.load_weights(weight_file)

        # sample_weights = np.matmul(train_labels, tag_weights)
        ### TODO: Train the model.
        weight_file = 'C:/planet/saved_models/weights.best_' + tag + '_' + arch + '_' + str(fold) + '.hdf5'

        checkpointer = ModelCheckpoint(filepath=weight_file,
                                       verbose=1, save_best_only=True)

        # Very 1st epoch
        batch_size = 96
        final_model.compile(loss='binary_crossentropy',
                            optimizer=optimizers.RMSprop(lr=0.00005),
                            metrics=['accuracy'])
        datagen = ImageDataGenerator(featurewise_std_normalization=False,
                                     rotation_range=1,
                                     width_shift_range=0,
                                     height_shift_range=0,
                                     zca_whitening=False,
                                     shear_range=0,
                                     zoom_range=0,
                                     horizontal_flip=False,
                                     vertical_flip=False,
                                     fill_mode='reflect')

        final_model.fit_generator(datagen.flow(train_set, train_labels, batch_size=batch_size),
                                  validation_data=(valid_set, valid_labels),
                                  steps_per_epoch=train_set.shape[0] / batch_size, epochs=1, callbacks=[checkpointer],
                                  verbose=2)

        if arch == 'VGG19_tune_square':
            # Let's accelarate training a little bit
            final_model.compile(loss='binary_crossentropy',
                                optimizer=optimizers.RMSprop(lr=0.00005),
                                metrics=['accuracy'])
            datagen = ImageDataGenerator(featurewise_std_normalization=False,
                                         rotation_range=360,
                                         width_shift_range=0,
                                         height_shift_range=0,
                                         zca_whitening=False,
                                         shear_range=0,
                                         zoom_range=0,
                                         horizontal_flip=False,
                                         vertical_flip=False,
                                         fill_mode='reflect')

            # fits the model on batches with real-time data augmentation:

            final_model.fit_generator(datagen.flow(train_set, train_labels, batch_size=batch_size),
                                      validation_data=(valid_set, valid_labels),
                                      steps_per_epoch=train_set.shape[0] / batch_size, epochs=1, callbacks=[checkpointer],
                                      verbose=2)

            # Okay now let's make it even harder but slow down learning rate
            final_model.compile(loss='binary_crossentropy',
                                optimizer=optimizers.RMSprop(lr=0.00002),
                                metrics=['accuracy'])
            datagen = ImageDataGenerator(featurewise_std_normalization=False,
                                         rotation_range=360,
                                         width_shift_range=0.05,
                                         height_shift_range=0.05,
                                         zca_whitening=False,
                                         shear_range=0.05,
                                         zoom_range=0.05,
                                         horizontal_flip=False,
                                         vertical_flip=False,
                                         fill_mode='reflect')

            #fits the model on batches with real-time data augmentation:
            final_model.fit_generator(datagen.flow(train_set, train_labels, batch_size=batch_size),
                                      validation_data=(valid_set, valid_labels),
                                      steps_per_epoch=train_set.shape[0] / batch_size, epochs=1, callbacks=[checkpointer],
                                      verbose=2)
            # Let's really slow down
            final_model.compile(loss='binary_crossentropy',
                                optimizer=optimizers.RMSprop(lr=0.000005),
                                metrics=['accuracy'])
            datagen = ImageDataGenerator(featurewise_std_normalization=False,
                                         rotation_range=360,
                                         width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         zca_whitening=False,
                                         shear_range=0.15,
                                         zoom_range=0.1,
                                         horizontal_flip=False,
                                         vertical_flip=False,
                                         fill_mode='nearest')

            #fits the model on batches with real-time data augmentation:
            final_model.fit_generator(datagen.flow(train_set, train_labels, batch_size=batch_size),
                                      validation_data=(valid_set, valid_labels),
                                      steps_per_epoch=train_set.shape[0] / batch_size, epochs=1, callbacks=[checkpointer],
                                      verbose=2)



        final_model.load_weights(weight_file)
        data_probs = final_model.predict(valid_set)


        fold+=1
        #For optimization loop only once
        if fold==1:
            break


    threshold_global = find_f2score_threshold_global(data_probs, valid_labels)
    thresholds_class = find_fbetascore_threshold_class(data_probs, valid_labels)

    predictions = []
    for Resnet_prob in data_probs:
        one_hot_labels = [1 if value > threshold_global else 0 for i, value in enumerate(Resnet_prob) ]
        predictions.append(one_hot_labels)

    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(valid_labels), np.array(predictions)

    scores = {}
    for i in range(y_true.shape[1]):
        scores[label_list[i]]=(accuracy_score(y_true[:,i],y_pred[:,i]),fbeta_score(y_true[:,i], y_pred[:,i], beta=2, average='binary'))
    print(sorted(scores.items(), key=operator.itemgetter(0)))


    # We need to use average='samples' here, any other average method will generate bogus results
    train_score = fbeta_score(y_true, y_pred, beta=2, average='samples')
    print('Training score with global threshold:(samples)',train_score)
    # We need to use average='samples' here, any other average method will generate bogus results
    train_score = fbeta_score(y_true, y_pred, beta=2, average='micro')
    print('Training score(micro) with global threshold(micro):',train_score)


    predictions = []
    for Resnet_prob in data_probs:
        one_hot_labels = [1 if value > thresholds_class[i] else 0 for i, value in enumerate(Resnet_prob) ]
        predictions.append(one_hot_labels)

    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(valid_labels), np.array(predictions)

    scores = {}
    for i in range(y_true.shape[1]):
        scores[label_list[i]]=(accuracy_score(y_true[:,i],y_pred[:,i]),fbeta_score(y_true[:,i], y_pred[:,i], beta=2, average='binary'))
    print(sorted(scores.items(), key=operator.itemgetter(0)))


    # We need to use average='samples' here, any other average method will generate bogus results
    train_score = fbeta_score(y_true, y_pred, beta=2, average='samples')
    print('Training score with class threshold:(samples)',train_score)

    train_score = fbeta_score(y_true, y_pred, beta=2, average='micro')
    print('Training score with class threshold:(micro)',train_score)


    if arch=='VGG19_tune_square':
        ## PREPARE PREDICTIONS HERE
        del data_set
        del train_set
        del valid_set
        prediction_files_df = pd.read_csv('empty_predictions.csv')

        predict_probs = np.ndarray((nsplit, prediction_files_df.shape[0], 17), dtype=np.float32)

        prediction_set, prediction_labels = load_images('all', prediction_files_df, 'predictions', arch, augmented,
                                                        'C:/planet/test-jpg/', norm, filetype, size, channels)

        final_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        for fold in range(nsplit):
            weight_file = 'C:/planet/saved_models/weights.best_' + tag + '_' + arch + '_' + str(fold) + '.hdf5'
            ### TODO: Load the model weights with the best validation loss.

            final_model.load_weights(weight_file)

            # Predict the dataset for this fold
            predict_probs[fold, :, :] = final_model.predict(prediction_set)
            print(fold)

        predict_probs = np.mean(predict_probs, 0)
        generate_predictions_file(predict_probs, thresholds_class, label_list)
        generate_predictions_file_global_thre(predict_probs, threshold_global, label_list)
        np.savez('Prediction_probs/Prediction_probs_' + tag + '_' + arch + filetype + '_' + str(size) + '_' + str(
            channels) + str(norm) + '.npz', predict_probs=predict_probs, scores=scores, thresholds=threshold_global)

        ## END PREPARE PREDICTIONS

    #Save the results:
    np.savez('Results_'+tag+'_'+arch+'.npz', data_probs=data_probs, data_labels=data_labels)

    #Save the calculated thresholds
    np.savez('thresholds_class.npz', thresholds=thresholds_class)

    #Save the calculated thresholds
    np.savez('thresholds_global.npz', thresholds=threshold_global)

    model_json =  final_model.to_json()
    model_file = 'C:/planet/saved_models/model_'+tag+'_'+arch+'.json'
    with open(model_file, 'w+') as json_file:
        json_file.write(model_json)


    return




