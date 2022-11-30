from imports import *

class DataGen():
    
    def __init__(self, path, img_size, batch_size) -> None:
        
        self.path = path
        self.img_size = img_size
        self.batch_size = batch_size
        
    def _loadData(self, split = 0.1):
        
        images = sorted(glob(os.path.join(self.path, "images/*")))
        masks = sorted(glob(os.path.join(self.path, "masks/*")))

        total_size = len(images)
        valid_size = int(split * total_size)
        test_size = int(split * total_size)

        train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
        train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

        train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
        train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    
    def _readImage(self, path):
        
        path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (self.img_size, self.img_size))
        x = x/255.0
        x = x.astype(np.float32)
        return x
    
    def _readMask(self, path):
        
        path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (self.img_size, self.img_size))
        x = x/255.0
        x = np.expand_dims(x, axis=-1)
        x = x.astype(np.float32)
        return x 
    
    def tf_parse(self, x, y):
        def _parse(x, y):
            x = self._readImage(x)
            y = self._readMask(y)
            return x, y

        x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
        x.set_shape([256, 256, 3])
        y.set_shape([256, 256, 1])
        return x, y
    
    def _createDataset(self, x, y):
        
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(self.tf_parse)
        dataset = dataset.batch(self.batch_size)
        return dataset
    
    def main():
        
        pass
                

if __name__ == "__main__":
    
    DataGen.main()