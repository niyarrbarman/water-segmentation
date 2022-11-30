from imports import *
from data import DataGen

if __name__ == "__main__":
    
    ROOT_DIR = "dataset"
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    
    data = DataGen(ROOT_DIR, IMAGE_SIZE, BATCH_SIZE)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data._loadData()
    
    print(X_train[0])
    
    train_dataset = data._createDataset(X_train, y_train)
    val_dataset = data._createDataset(X_val, y_val)
    
    model = sm.Unet('resnet34', input_shape=(256, 256, 3), encoder_weights="imagenet", classes = 1, activation = "sigmoid")
    
    model.compile(optimizer = tf.keras.optimizers.Adam(1e-3), 
              loss = sm.losses.DiceLoss(), 
              metrics = sm.metrics.iou_score)
    
    # print(model.summary())
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),
    ]
    
    h = model.fit(train_dataset,
          epochs = 5,
          validation_data = val_dataset,
          callbacks = callbacks)
    
    test_dataset = data._createDataset(X_test, y_test)
    _, iou = model.evaluate(test_dataset)
    print(f"Testing IOU score is { iou*100 } %")