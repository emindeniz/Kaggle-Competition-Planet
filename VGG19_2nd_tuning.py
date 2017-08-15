import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import  Flatten, Dense,BatchNormalization, GaussianNoise, Input,Activation
from keras.models import Sequential
from keras import backend as K
import gc
from pathlib import Path
from six.moves import cPickle as pickle
from keras import optimizers
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import io, transform
import operator


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


def f2_score(y_true, y_pred):
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def find_f2score_threshold_global(p_valid, y_valid, try_all=False, verbose=False):
    best = 0
    best_score = -1
    totry = np.arange(0,1,0.05) if try_all is False else np.unique(p_valid)
    for t in totry:
        score = f2_score(y_valid, p_valid > t)
        if score > best_score:
            best_score = score
            best = t
    if verbose is True:
        print('Best score: ', round(best_score, 5), ' @ threshold =', best)
    return best

def find_fbetascore_threshold_class(p_valid, y_valid, try_all=False):
    best = [0]*p_valid.shape[1]
    best_score = [-1]*p_valid.shape[1]

    totry = np.arange(0,1,0.05) if try_all is False else np.unique(p_valid)

    for i in range(p_valid.shape[1]):
        for t in totry:
            score = fbeta_score(y_valid[:,i], p_valid[:,i] > t, beta=2, average='binary')
            if score > best_score[i]:
                best_score[i] = score
                best[i] = t

    best = [round(t,3) for t in best]
    return best

def load_images(tag,allimage_names,type,arch,augmented,image_folder,norm,filetype,image_size,channels):
    sess = tf.Session()
    K.set_session(sess)

    pickle_file = 'C:/planet/Pickle_files/' + tag + '-'+type+'_' +arch+ '_augmented_'+str(augmented)+'_norm_'+str(norm)+filetype+'_size'+str(image_size)+'_chan'+str(channels)+'.pickle'
    if Path(pickle_file).is_file():
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            dataset = save['dataset']
            labels = save['labels']
            del save  # hint to help gc free up memory
    else:
        if arch == 'Inception_V3':
            dataset = np.ndarray((allimage_names.shape[0], 2048), dtype=np.float32)
            labels = np.ndarray((allimage_names.shape[0], 17), dtype=np.int32)
        elif arch == 'Resnet':
            dataset = np.ndarray((allimage_names.shape[0], 2,2, 2048), dtype=np.float32)
            labels = np.ndarray((allimage_names.shape[0], 17), dtype=np.int32)
        elif arch == 'VGG19':
            dataset = np.ndarray((allimage_names.shape[0], 4,4,512), dtype=np.float32)
            labels = np.ndarray((allimage_names.shape[0], 17), dtype=np.int32)
        elif arch == 'VGG16':
            dataset = np.ndarray((allimage_names.shape[0], 8, 8, 512), dtype=np.float32)
            labels = np.ndarray((allimage_names.shape[0], 17), dtype=np.int32)
        elif arch == 'Xception':
            dataset = np.ndarray((allimage_names.shape[0],2048), dtype=np.float32)
            labels = np.ndarray((allimage_names.shape[0], 17), dtype=np.int32)
        else:
            dataset = np.ndarray((allimage_names.shape[0], image_size, image_size, channels), dtype=np.float32)
            labels = np.ndarray((allimage_names.shape[0], 17), dtype=np.int32)

        batch_size = 70000
        for i in range(0,allimage_names.shape[0],batch_size):
            df = allimage_names[i:i+batch_size]
            dataset_batch = np.ndarray((df.shape[0], image_size, image_size, channels), dtype=np.float32)
            labels_batch = np.ndarray((df.shape[0], 17), dtype=np.int32)
            num_images = 0
            for row in df.itertuples():
                image_file = image_folder + dict(row._asdict())['image_name'] + filetype
                try:
                    image_data = io.imread(image_file).astype(float)
                    image_data = transform.resize(image_data, (image_size, image_size))
                    if filetype=='.jpg' and norm=='divide':
                        dataset_batch[num_images, :, :, :] = image_data/255.0
                    elif filetype == '.jpg' and norm == 'globalmm':
                        image_min = np.min(image_data[:,:,0:channels])
                        image_max = np.max(image_data[:,:,0:channels])
                        dataset_batch[num_images, :, :, :] = (image_data[:,:,0:channels]-image_min)/(image_max-image_min)
                    elif filetype == '.jpg' and norm == 'globalmm255':
                        image_min = np.min(image_data[:,:,0:channels])
                        image_max = np.max(image_data[:,:,0:channels])
                        dataset_batch[num_images, :, :, :] = 255*(image_data[:,:,0:channels]-image_min)/(image_max-image_min)
                    else:
                        dataset_batch[num_images, :, :, :] = image_data
                    if 'predictions' in type:
                        labels_batch[num_images]=1
                    else:
                        labels_batch[num_images] = row[3:22]
                    num_images = num_images + 1
                    del image_data
                except IOError as e:
                    print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

            if arch =='Resnet':
                from keras.applications.resnet50 import ResNet50, preprocess_input
                dataset_batch = ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(dataset_batch))
            if arch =='Inception_V3':
                from keras.applications.inception_v3 import InceptionV3, preprocess_input
                dataset_batch = InceptionV3(weights='imagenet', include_top=False,pooling='avg').predict(preprocess_input(dataset_batch))
            if arch =='VGG19':
                from keras.applications.vgg19 import VGG19, preprocess_input
                dataset_batch = VGG19(weights='imagenet', include_top=False).predict(preprocess_input(dataset_batch))
            if arch =='VGG16':
                from keras.applications.vgg16 import VGG16, preprocess_input
                dataset_batch = VGG16(weights='imagenet', include_top=False).predict(preprocess_input(dataset_batch))
            if arch == 'Xception':
                from keras.applications.xception import Xception, preprocess_input
                dataset_batch = Xception(weights='imagenet', include_top=False,pooling='avg').predict(preprocess_input(dataset_batch))



            dataset[i:i+batch_size] = dataset_batch
            labels[i:i + batch_size] = labels_batch
            del dataset_batch
            print('Processed',i)

        try:
            f = open(pickle_file, 'wb')
            save = {
                'dataset': dataset,
                'labels': labels,
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise


    print('Full tensor for '+type+':', dataset.shape)
    print('Label shape:',labels.shape)
    print('label distribution:', np.mean(labels))

    return dataset,labels


def run_model(df,arch,tag,augmented,image_folder,norm,filetype,size,channels,label_list):

    data_set, data_labels = load_images('all', df, 'all', arch,augmented,image_folder,norm,filetype,size,channels)
    print(data_set.shape)
    print(data_labels.shape)

    nsplit=5

    threshold_global = np.ndarray((nsplit), dtype=np.float32)
    thresholds_class = np.ndarray((nsplit,17), dtype=np.float32)

    kf = KFold(n_splits=nsplit)
    fold = 0
    for train_index, test_index in kf.split(data_set):
        train_set, train_labels = data_set[train_index], data_labels[train_index]
        valid_set, valid_labels = data_set[test_index], data_labels[test_index]

        weight_file = 'C:/planet/saved_models/weights.best_' + tag + '_' + 'VGG19_tune' + '_' + str(0) + '.hdf5'

        # Create the model
        from keras.applications.vgg19 import VGG19

        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(size,size,channels))


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
                                  steps_per_epoch=train_set.shape[0] / batch_size, epochs=2, callbacks=[checkpointer],
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
                                  steps_per_epoch=train_set.shape[0] / batch_size, epochs=2, callbacks=[checkpointer],
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
                                  steps_per_epoch=train_set.shape[0] / batch_size, epochs=2, callbacks=[checkpointer],
                                  verbose=2)



        final_model.load_weights(weight_file)
        data_probs = final_model.predict(valid_set)

        thresholds_class[fold,:] = find_fbetascore_threshold_class(data_probs, valid_labels)
        threshold_global[fold] = find_f2score_threshold_global(data_probs, valid_labels)

        np.savez('thresholds_class.npz', thresholds=thresholds_class[fold])

        # Save the calculated thresholds
        np.savez('thresholds_global.npz', thresholds=threshold_global[fold])
        print(threshold_global[fold])
        print(thresholds_class[fold])

        fold+=1


    threshold_global = np.mean(threshold_global, 0)
    thresholds_class = np.mean(thresholds_class, 0)

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



    #Save the results:
    np.savez('Results_'+tag+'_'+arch+'.npz', data_probs=data_probs, data_labels=data_labels)

    #Save the calculated thresholds
    np.savez('thresholds_class.npz', thresholds=thresholds_class)

    #Save the calculated thresholds
    np.savez('thresholds_global.npz', thresholds=threshold_global)


    ## PREPARE PREDICTIONS HERE
    del data_set
    del train_set
    del valid_set
    prediction_files_df = pd.read_csv('empty_predictions.csv')

    predict_probs = np.ndarray((nsplit, prediction_files_df.shape[0], 17), dtype=np.float32)


    prediction_set, prediction_labels = load_images('all', prediction_files_df, 'predictions', arch,augmented,'C:/planet/test-jpg/',norm,filetype,size,channels)


    final_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    for fold in range(nsplit):


        weight_file = 'C:/planet/saved_models/weights.best_' + tag + '_' + arch + '_' + str(fold) + '.hdf5'
        ### TODO: Load the model weights with the best validation loss.

        final_model.load_weights(weight_file)

        #Predict the dataset for this fold
        predict_probs[fold,:,:] =  final_model.predict(prediction_set)
        print(fold)


    predict_probs = np.mean(predict_probs,0)
    generate_predictions_file(predict_probs, thresholds_class, label_list)
    generate_predictions_file_global_thre(predict_probs,threshold_global,label_list)
    np.savez('Prediction_probs/Prediction_probs_'+tag+'_'+arch+filetype+'_'+str(size)+'_'+str(channels)+str(norm)+'.npz', predict_probs=predict_probs,scores=scores,thresholds=threshold_global)

    ## END PREPARE PREDICTIONS

    model_json =  final_model.to_json()
    model_file = 'C:/planet/saved_models/model_'+tag+'_'+arch+'.json'
    with open(model_file, 'w+') as json_file:
        json_file.write(model_json)

    sess = K.get_session()
    if sess:
        sess.close()
    gc.collect()

    return


def main():
    #np.random.seed(98)
    tag = 'all'
    arch = 'VGG19_tune_square'
    augmented = 0
    image_folder = 'C:/planet/train-jpg/'
    norm = 'none'
    size = 128
    channels = 3
    filetype = '.jpg'

    df = pd.read_csv('train_v2.csv')

    # Build list with unique labels
    label_list = []
    for tag_str in df.tags.values:
        labels = tag_str.split(' ')
        for label in labels:
            if label not in label_list:
                label_list.append(label)
    print(label_list)

    # Add onehot features for every label
    for label in label_list:
        df[label] = df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)


    label_list = ['haze','primary','agriculture','clear', 'water', 'habitation', 'road', 'cultivation','slash_burn','cloudy','partly_cloudy', 'conventional_mine',
               'bare_ground','artisinal_mine', 'blooming', 'selective_logging', 'blow_down', ]
    df = df.sample(frac=1)


    all_tags = [item for sublist in list(df['tags'].apply(lambda row: row.split(" ")).values) for item in sublist]
    print('total of {} non-unique tags in all training images'.format(len(all_tags)))
    print('average number of labels per image {}'.format(1.0 * len(all_tags) / df.shape[0]))


    print(df.info())
    print(df[:5])
    run_model(df,arch,tag,augmented,image_folder,norm,filetype,size,channels,label_list)

    return



if __name__ == "__main__":
    main()



