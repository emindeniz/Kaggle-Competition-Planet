import numpy as np # linear algebra
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import tensorflow as tf
from keras import backend as K
from pathlib import Path
from six.moves import cPickle as pickle
from skimage import io, transform
from sklearn.metrics import fbeta_score

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
            dataset = np.ndarray((allimage_names.shape[0], 4, 4, 512), dtype=np.float32)
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
                    if filetype == '.jpg' and norm == 'divide':
                        dataset_batch[num_images, :, :, :] = image_data / 255.0
                    elif filetype == '.jpg' and norm == 'globalmm':
                        image_min = np.min(image_data[:,:,0:channels])
                        image_max = np.max(image_data[:,:,0:channels])
                        dataset_batch[num_images, :, :, :] = (image_data-image_min)/(image_max-image_min)
                    elif filetype == '.jpg' and norm == 'globalmm255':
                        image_min = np.min(image_data[:,:,0:channels])
                        image_max = np.max(image_data[:,:,0:channels])
                        dataset_batch[num_images, :, :, :] = 255*(image_data[:,:,0:channels]-image_min)/(image_max-image_min)
                    elif filetype == '.jpg' and norm == 'global255':
                        image_data[:, :, 0] = MinMaxScaler(feature_range=(0, 255)).fit_transform(image_data[:, :, 0])
                        image_data[:, :, 1] = MinMaxScaler(feature_range=(0, 255)).fit_transform(image_data[:, :, 1])
                        image_data[:, :, 2] = MinMaxScaler(feature_range=(0, 255)).fit_transform(image_data[:, :, 2])
                        dataset_batch[num_images, :, :, :] = image_data
                    else:
                        dataset_batch[num_images, :, :, :] = image_data

                    if type=='predictions':
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