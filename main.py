import stow
import logging
import tensorflow as tf
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CWERMetric
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from configs import ModelConfigs
from mltu.dataProvider import DataProvider
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate

from training import train_model


def download_and_unzip(url, extract_to='Datasets'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


if not stow.exists(stow.join('Datasets', 'captcha_images_v2')):
    download_and_unzip('https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip',
                       extract_to='Datasets')

dataset, vocab, max_len = [], set(), 0
for file in stow.ls(stow.join('Datasets', 'captcha_images_v2')):
    dataset.append([stow.relpath(file), file.name])
    vocab.update(list(file.name))
    max_len = max(max_len, len(file.name))

configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()

data_provider = DataProvider(
        dataset=dataset,
        skip_validation=True,
        batch_size=configs.batch_size,
        data_preprocessors=[ImageReader()],
        transformers=[
            ImageResizer(configs.width, configs.height),
            LabelIndexer(configs.vocab),
            LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
)

train_data_provider, val_data_provider = data_provider.split(split=0.9)
train_data_provider.augmentors = [RandomBrightness(), RandomRotate(), RandomErodeDilate()]

model = train_model(
        input_dim=(configs.height, configs.width, 3),
        output_dim=len(configs.vocab),
)

model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
        loss=CTCloss(),
        metrics=[CWERMetric(padding_token=len(configs.vocab))],
)

# Define callbacks
logging.info(f"Saving model to {configs.model_path}")
earlystopper = EarlyStopping(monitor='val_CER', patience=40, verbose=1)
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor='val_CER', verbose=1, save_best_only=True,
                             mode='min')
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f'{configs.model_path}/logs', update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_CER', factor=0.9, min_delta=1e-10, patience=20, verbose=1, mode='auto')
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Train the model
logging.info(f"Training model for {configs.train_epochs} epochs")
# Convert train_data_provider to a tf.data.Dataset object
train_data_provider = tf.data.Dataset.from_generator(lambda: train_data_provider, output_types=(tf.float32, tf.int32))
# Convert val_data_provider to a tf.data.Dataset object
val_data_provider = tf.data.Dataset.from_generator(lambda: val_data_provider, output_types=(tf.float32, tf.int32))
model.fit(
        train_data_provider,
        validation_data=val_data_provider,
        epochs=configs.train_epochs,
        callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
        workers=configs.train_workers
)

# Save training and validation datasets as csv files
logging.info(f"Saving training and validation datasets to {configs.model_path}")
train_data_provider.to_csv(stow.join(configs.model_path, 'train.csv'))
val_data_provider.to_csv(stow.join(configs.model_path, 'val.csv'))
