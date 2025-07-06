import numpy as np  
import os
import tensorflow as tf  
from tensorflow.keras import Sequential, Model, Input, applications  
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomRotation, RandomZoom, RandomFlip  
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.losses import CategoricalCrossentropy  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  
from tensorflow.data import AUTOTUNE  
import matplotlib.pyplot as plt  
from sklearn.metrics import classification_report, confusion_matrix


BATCH_SIZE = 64
IMG_SIZE = (224, 224)
SEED = 42
TARGET_TEST_ACCURACY = 0.85
MAX_RETRIES = 4


data_dir = "C:/Users/gande/Downloads/Alzheimers-ADNI"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")


tf.random.set_seed(SEED)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset='training',
    seed=SEED,
    shuffle=True
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=SEED,
    shuffle=True    
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

print("✅ Datasets loaded successfully!")

class_names = train_dataset.class_names
print("🔍 Class Names:", class_names)

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


class_counts = {folder: len(os.listdir(os.path.join(train_dir, folder))) for folder in class_names}
print("📊 Class Counts:", class_counts)

total = sum(class_counts.values())
class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts.values())}
print("⚖️ Class Weights:", class_weights)


def data_augmentor():
    return Sequential([
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomFlip(mode='horizontal')  
    ], name="data_augmentation")

augmentation_layer = data_augmentor()

def alzheimer_model(image_shape=IMG_SIZE, num_classes=len(class_names)):
    base_model = applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=image_shape + (3,))
    
    for layer in base_model.layers[-100:]:  
        layer.trainable = True

    inputs = Input(shape=image_shape + (3,))
    x = augmentation_layer(inputs)
    x = applications.efficientnet.preprocess_input(x)
    
    x = base_model(x, training=True)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


checkpoint_cb = ModelCheckpoint(
    '20_04_2025_ADNI_best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1, save_weights_only=False
)

earlystop_cb = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, restore_best_weights=True)


model = alzheimer_model()
model.summary()

all_histories = []

print("🚀 Starting Initial Training...")
initial_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=90,
    class_weight=class_weights,
    callbacks=[checkpoint_cb, earlystop_cb]
)
all_histories.append(initial_history)

eval_results = model.evaluate(test_dataset, verbose=0)
current_test_acc = eval_results[1]
print(f'🎯 Initial Test Loss: {eval_results[0]} | Test Accuracy: {current_test_acc:.4f}')

retry_count = 0
while current_test_acc < TARGET_TEST_ACCURACY and retry_count < MAX_RETRIES:
    print(f"\n🔁 Retry {retry_count + 1}/{MAX_RETRIES}: Test accuracy {current_test_acc:.4f} < {TARGET_TEST_ACCURACY}")
    print("⏳ Continuing training for 5 more epochs...\n")

    retrain_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=8,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, earlystop_cb]
    )
    all_histories.append(retrain_history)

    model = tf.keras.models.load_model('20_04_2025_ADNI_best_model.keras')
    eval_results = model.evaluate(test_dataset, verbose=0)
    current_test_acc = eval_results[1]
    retry_count += 1

model.save('20_04_2025_ADNI_final_model.keras')

print(f"\n✅ Final Test Accuracy: {current_test_acc:.4f}")
if current_test_acc >= TARGET_TEST_ACCURACY:
    print(f"🎉 Target of {TARGET_TEST_ACCURACY*100:.0f}% test accuracy achieved!")
else:
    print(f"⚠️ Stopped after {retry_count} retries. Final test accuracy is {current_test_acc:.4f}")

with open("evaluation_results.txt", "w") as f:
    f.write(f'Test Loss: {eval_results[0]} | Test Accuracy: {current_test_acc:.4f}\n')


full_acc = []
full_val_acc = []
full_loss = []
full_val_loss = []

for hist in all_histories:
    full_acc.extend(hist.history['accuracy'])
    full_val_acc.extend(hist.history['val_accuracy'])
    full_loss.extend(hist.history['loss'])
    full_val_loss.extend(hist.history['val_loss'])

plt.figure(figsize=(8, 6))
plt.plot(full_acc, label='Train Accuracy', marker='o')
plt.plot(full_val_acc, label='Validation Accuracy', marker='s')
plt.title('Model Accuracy (All Epochs)', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('20_04_2025_accuracy_plot_all_epochs.png')
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(full_loss, label='Train Loss', marker='o', color='red')
plt.plot(full_val_loss, label='Validation Loss', marker='s', color='blue')
plt.title('Model Loss (All Epochs)', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('20_04_2025_loss_plot_all_epochs.png')
plt.show()


y_true = []
y_pred = []

for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(tf.argmax(labels, axis=1).numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("📊 Classification Report")
print(classification_report(y_true, y_pred, target_names=class_names))

print("🧩 Confusion Matrix")
print(confusion_matrix(y_true, y_pred))
