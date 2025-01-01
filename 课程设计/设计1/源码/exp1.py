import os
import tensorflow as tf
import matplotlib.pyplot as plt

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#tf.config.set_visible_devices([], 'GPU')


# 设置路径和超参数
train_directory = 'data/train'
val_directory = 'data/val'
result_directory = 'result'
batch_size = 32
img_size = (160, 160)
base_learning_rate = 0.0001
initial_epochs = 15
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs


train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    shuffle=True,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_directory,
    shuffle=True,
    image_size=img_size,
    batch_size=batch_size
)

# 提取类别名
class_names = train_dataset.class_names
print(f"Class Names: {class_names}")

# 数据集性能优化
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.0005),
])
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

'''for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
  plt.show()'''

# 加载预训练模型（MobileNetV2）
img_shape = img_size + (3,)  # 添加通道数
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_shape,
    include_top=False,  # 不加载顶层分类器
    weights='imagenet'  # 加载 ImageNet 权重
)
base_model.trainable = False  # 冻结预训练模型

# 构建模型
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(class_names), activation='softmax')  # 输出类别数

inputs = tf.keras.Input(shape=img_shape)
x = data_augmentation(inputs)
x = preprocess_input(x)  # 数据预处理
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)  # 防止过拟合
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

#model.summary()

# 训练模型
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=initial_epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,3.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig(os.path.join(result_directory, f"学习曲线.png"))



# 模型解冻并微调
base_model.trainable = True
fine_tune_at = 90  # 微调的层数
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# 重新编译模型（更低的学习率）
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate / 10),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

# 微调训练
history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1]
)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.5])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.savefig(os.path.join(result_directory, f"微调学习曲线.png"))

model.save('car_classification.keras')