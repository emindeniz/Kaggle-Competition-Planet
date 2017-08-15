import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Flatten, Dense,BatchNormalization, GaussianNoise, Activation
from keras.models import Sequential
from keras import optimizers
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import KFold
import operator
from utils import load_images,find_f2score_threshold_global,find_fbetascore_threshold_class
import numpy as np # linear algebra
from VGG19_tuning import run_model_tuning


def run_model_initial(df,arch,tag,augmented,image_folder,norm,filetype,size,channels,label_list):


    data_set, data_labels = load_images('all', df, 'all', arch,augmented,image_folder,norm,filetype,size,channels)
    print(data_set.shape)
    print(data_labels.shape)


    nsplit=5
    kf = KFold(n_splits=nsplit)
    fold = 0
    for train_index, test_index in kf.split(data_set):
        train_set, train_labels = data_set[train_index], data_labels[train_index]
        valid_set, valid_labels = data_set[test_index], data_labels[test_index]


        # Create the model
        model = Sequential()
        model.add(Flatten(input_shape=data_set.shape[1:],name='one'))
        model.add(Dense(1024, activation='relu', name='three'))
        model.add(Dense(17, name='seven'))
        model.add(Activation('sigmoid', name='nine'))
        model.summary()

        rms_optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss='binary_crossentropy', optimizer=rms_optimizer, metrics=['accuracy'])

        weight_file = 'C:/planet/saved_models/weights.best_' + tag + '_' + arch + '_' + str(fold) + '.hdf5'

        checkpointer = ModelCheckpoint(filepath=weight_file,
                                       verbose=1, save_best_only=True)

        model.fit(train_set, train_labels,
                         validation_data=(valid_set, valid_labels),
                         epochs=5, batch_size=1024,
                         callbacks=[checkpointer], verbose=2)


        fold+=1
        if fold==1:
            break

    # nsplit equals 1 because we assessing the performance assessment
    nsplit = 1
    data_probs = np.ndarray((nsplit, valid_set.shape[0], valid_labels.shape[1]), dtype=np.float32)

    for fold in range(nsplit):

        weight_file = 'C:/planet/saved_models/weights.best_' + tag + '_' + arch + '_' + str(fold) + '.hdf5'

        ### TODO: Load the model weights with the best validation loss.
        model.load_weights(weight_file)

        #Predict the dataset for this fold
        data_probs[fold,:,:] =  model.predict(valid_set)

    data_probs = np.mean(data_probs,0)

    thresholds_class = find_fbetascore_threshold_class(data_probs,valid_labels)
    threshold_global = find_f2score_threshold_global(data_probs,valid_labels)

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


    model_json =  model.to_json()
    model_file = 'C:/planet/saved_models/model_'+tag+'_'+arch+'.json'
    with open(model_file, 'w+') as json_file:
        json_file.write(model_json)

    return


def main():

    tag = 'all'
    arch = 'VGG19'
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
    print(df[1:5])

    run_model_initial(df,arch,tag,augmented,image_folder,norm,filetype,size,channels,label_list)
    run_model_tuning(df, 'VGG19_tune', tag, augmented, image_folder, norm, filetype, size, channels, label_list)
    run_model_tuning(df, 'VGG19_tune_square', tag, augmented, image_folder, norm, filetype, size, channels, label_list)

    return

if __name__ == "__main__":
    main()



