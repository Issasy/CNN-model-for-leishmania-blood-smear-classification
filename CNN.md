```python
#Preprocessing images |27/06/2025|@issa yusuf
#Positive_images
import cv2
import os
from tqdm import tqdm  
def decompose_and_save_images(input_folder="F:\\BIS_thesis\\Positive",
                              output_folder="F:\\BIS_thesis\\Positive_output"):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not read {image_file}. Skipping.")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(rgb_image)
        grayscale = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_image)

        image_name = os.path.splitext(image_file)[0]
        image_output_folder = os.path.join(output_folder, image_name)
        os.makedirs(image_output_folder, exist_ok=True)

        cv2.imwrite(os.path.join(image_output_folder, "R.png"), r)
        cv2.imwrite(os.path.join(image_output_folder, "G.png"), g)
        cv2.imwrite(os.path.join(image_output_folder, "B.png"), b)
        cv2.imwrite(os.path.join(image_output_folder, "Grayscale.png"), grayscale)
        cv2.imwrite(os.path.join(image_output_folder, "H.png"), h)
        cv2.imwrite(os.path.join(image_output_folder, "S.png"), s)
        cv2.imwrite(os.path.join(image_output_folder, "V.png"), v)

    print(f"Decomposition complete. Sub-images saved in: {output_folder}")

decompose_and_save_images("F:\\BIS_thesis\\Positive", "F:\\BIS_thesis\\Positive_output")
```


```python
#Preprocessing images |27/06/2025|@issa yusuf
#Negative_images
import cv2
import os
from tqdm import tqdm  
def decompose_and_save_images(input_folder="F:\\BIS_thesis\\Negative",
                              output_folder="F:\\BIS_thesis\\Negative_output"):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not read {image_file}. Skipping.")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(rgb_image)
        grayscale = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_image)

        image_name = os.path.splitext(image_file)[0]
        image_output_folder = os.path.join(output_folder, image_name)
        os.makedirs(image_output_folder, exist_ok=True)

        cv2.imwrite(os.path.join(image_output_folder, "R.png"), r)
        cv2.imwrite(os.path.join(image_output_folder, "G.png"), g)
        cv2.imwrite(os.path.join(image_output_folder, "B.png"), b)
        cv2.imwrite(os.path.join(image_output_folder, "Grayscale.png"), grayscale)
        cv2.imwrite(os.path.join(image_output_folder, "H.png"), h)
        cv2.imwrite(os.path.join(image_output_folder, "S.png"), s)
        cv2.imwrite(os.path.join(image_output_folder, "V.png"), v)

    print(f"Decomposition complete. Sub-images saved in: {output_folder}")

decompose_and_save_images("F:\\BIS_thesis\\Negative", "F:\\BIS_thesis\\Negative_output")
```


```python
#Noise reduction techniques | from the article: /10.3390/s24248180
import cv2
import numpy as np
import os
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt

def apply_filters(channel):
    """Apply noise reduction filters to a single channel"""
    # Median Filter (3x3 kernel) - removes salt-and-pepper noise
    median = cv2.medianBlur(channel, 3)
    
    # Gaussian Filter (5x5 kernel) - smooths general noise
    gaussian = cv2.GaussianBlur(median, (5, 5), 0)
    
    # Morphological Closing (3x3 ellipse kernel) - fills small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(gaussian, cv2.MORPH_CLOSE, kernel)
    
    return closed

def enhance_contrast(channel):
    """Apply contrast enhancement techniques"""
    # Laplacian Edge Enhancement
    laplacian = cv2.Laplacian(channel, cv2.CV_16S, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian)
    enhanced = cv2.addWeighted(channel, 1.5, laplacian, -0.5, 0)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(enhanced)
    
    return clahe_applied

def process_combinations(base_image, channel_name):
    """Test all combinations of preprocessing techniques"""
    results = {}
    
    # Individual techniques
    median = cv2.medianBlur(base_image, 3)
    gaussian = cv2.GaussianBlur(base_image, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(base_image, cv2.MORPH_CLOSE, kernel)
    laplacian = cv2.Laplacian(base_image, cv2.CV_16S, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Store individual results
    results['raw'] = base_image
    results['median'] = median
    results['gaussian'] = gaussian
    results['closed'] = closed
    results['laplacian'] = laplacian
    results['clahe'] = clahe.apply(base_image)
    
    # Combination 1: Median + Gaussian + Closing
    temp = cv2.medianBlur(base_image, 3)
    temp = cv2.GaussianBlur(temp, (5, 5), 0)
    results['combo1'] = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
    
    # Combination 2: All filters + CLAHE
    temp = apply_filters(base_image)
    results['combo2'] = enhance_contrast(temp)
    
    return results

def visualize_results(results, channel_name):
    """Display all processing combinations"""
    plt.figure(figsize=(15, 10))
    for i, (name, img) in enumerate(results.items(), 1):
        plt.subplot(3, 3, i)
        plt.imshow(img, cmap='gray')
        plt.title(f"{channel_name} - {name}")
    plt.tight_layout()
    plt.show()

def process_decomposed_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for image_folder in tqdm(os.listdir(input_folder)):
        channel_path = os.path.join(input_folder, image_folder)
        if not os.path.isdir(channel_path):
            continue
            
        # Process each channel independently
        for channel_file in os.listdir(channel_path):
            if not channel_file.endswith('.png'):
                continue
                
            channel_name = channel_file.split('.')[0]
            img_path = os.path.join(channel_path, channel_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply processing pipeline
            filtered = apply_filters(img)
            enhanced = enhance_contrast(filtered)
            
            # Test all combinations
            combinations_results = process_combinations(img, channel_name)
            
            # Save and visualize
            output_path = os.path.join(output_folder, image_folder)
            os.makedirs(output_path, exist_ok=True)
            
            cv2.imwrite(os.path.join(output_path, f"{channel_name}_filtered.png"), filtered)
            cv2.imwrite(os.path.join(output_path, f"{channel_name}_enhanced.png"), enhanced)
            
            # Save all combinations for analysis
            for combo_name, combo_img in combinations_results.items():
                cv2.imwrite(os.path.join(output_path, f"{channel_name}_{combo_name}.png"), combo_img)
            
            # Visualize one sample per folder (optional)
            if channel_file == "S.png":  # Typically most important for parasites
                visualize_results(combinations_results, channel_name)
process_decomposed_images("F:\\BIS_thesis\\Positive_output", "F:\\BIS_thesis\\Positive_output_noise_reduced")
```


```python


```


```python
##04/07/2025 positive
import os
import cv2
from tqdm import tqdm

# Paths configuration
input_base_folder = r"F:\BIS_thesis\Positive_output_noise_reduced"
output_base_folder = r"F:\BIS_thesis\Processed_S_Variants_p"

# Create main output folder
os.makedirs(output_base_folder, exist_ok=True)

# Process each original image folder
for img_folder in tqdm(os.listdir(input_base_folder)):
    input_folder_path = os.path.join(input_base_folder, img_folder)
    
    if not os.path.isdir(input_folder_path):
        continue
    
    # Create matching output subfolder
    output_subfolder = os.path.join(output_base_folder, img_folder)
    os.makedirs(output_subfolder, exist_ok=True)
    
    # Find all processed S-channel variants
    s_variants = [f for f in os.listdir(input_folder_path) 
                 if f.startswith('S_') and f.endswith('.png')]
    
    # Copy each variant to new location
    for variant in s_variants:
        variant_path = os.path.join(input_folder_path, variant)
        img = cv2.imread(variant_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # Save with original variant name
            output_path = os.path.join(output_subfolder, variant)
            cv2.imwrite(output_path, img)

print(f"\nOrganized {len(s_variants)} S-channel variants per image in: {output_base_folder}")
```


```python
##04/07/2025 negative
import os
import cv2
from tqdm import tqdm

# Paths configuration
input_base_folder = r"F:\BIS_thesis\Negative_output_noise_reduced"
output_base_folder = r"F:\BIS_thesis\Processed_S_Variants_n"

# Create main output folder
os.makedirs(output_base_folder, exist_ok=True)

# Process each original image folder
for img_folder in tqdm(os.listdir(input_base_folder)):
    input_folder_path = os.path.join(input_base_folder, img_folder)
    
    if not os.path.isdir(input_folder_path):
        continue
    
    # Create matching output subfolder
    output_subfolder = os.path.join(output_base_folder, img_folder)
    os.makedirs(output_subfolder, exist_ok=True)
    
    # Find all processed S-channel variants
    s_variants = [f for f in os.listdir(input_folder_path) 
                 if f.startswith('S_') and f.endswith('.png')]
    
    # Copy each variant to new location
    for variant in s_variants:
        variant_path = os.path.join(input_folder_path, variant)
        img = cv2.imread(variant_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # Save with original variant name
            output_path = os.path.join(output_subfolder, variant)
            cv2.imwrite(output_path, img)

print(f"\nOrganized {len(s_variants)} S-channel variants per image in: {output_base_folder}")
```


```python

```


```python
import os
import cv2
import shutil
from tqdm import tqdm

# ===== USER CONFIGURATION =====
INPUT_DIR = r"F:\BIS_thesis\data1"  # Folder with class subfolders
OUTPUT_DIR = r"F:\BIS_thesis\S_median_data1"         # Where to save organized S-images
S_VARIANT = "S_median.png"                                 # Which S-channel file to use
TARGET_SIZE = (256, 256)                                   # Resize dimensions
CLEAN_OUTPUT = True                                        # Delete output dir before start

# ===== PREPARE OUTPUT =====
if CLEAN_OUTPUT and os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== PROCESS IMAGES =====
total_copied = 0
class_folders = [f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))]

for class_name in class_folders:
    class_input_dir = os.path.join(INPUT_DIR, class_name)
    class_output_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    print(f"\nProcessing class: {class_name}")
    image_folders = [f for f in os.listdir(class_input_dir) if os.path.isdir(os.path.join(class_input_dir, f))]
    
    for img_folder in tqdm(image_folders):
        # Path to the S-channel image
        s_img_path = os.path.join(class_input_dir, img_folder, S_VARIANT)
        
        if os.path.exists(s_img_path):
            # Load, resize, and save
            img = cv2.imread(s_img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                output_path = os.path.join(class_output_dir, f"{img_folder}_{S_VARIANT}")
                cv2.imwrite(output_path, img_resized)
                total_copied += 1

# ===== VERIFICATION =====
print(f"\nSuccessfully processed {total_copied} images")
print(f"Output structure:\n{OUTPUT_DIR}")
print("\nSample output paths:")
for root, _, files in os.walk(OUTPUT_DIR):
    if files:
        print(f"• {os.path.relpath(os.path.join(root, files[0]), OUTPUT_DIR)}")
```


```python
#25/07/2025 
#Loading the data
#F:\BIS_thesis\Positive_output_noise_reduced
data = r"F:\BIS_thesis\S_median_data1"
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
data=tf.keras.utils.image_dataset_from_directory(data) ##done :)
data_iterator=data.as_nu mpy_iterator() #convereter to arrays
data_iterator
batch=data_iterator.next()
batch
len(batch) # 2 for images and labels
batch[0].shape
batch[1] 

```


```python
#class 1= Positive
#class 0=Negative
#which number (1,0) is assigned to each image
fig, ax=plt.subplots(ncols=4,figsize=(20,20))
for idx,  img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
```


```python
# Preprocessing the data
#scale data
scaled= batch[0]/250.0
batch[0].min()
scaled.max()
data=data.map(lambda x,y: (x/250,y))
scaled_iterator=data.as_numpy_iterator()
batch=scaled_iterator.next()
```


```python
batch[0].min()
```


```python
fig, ax=plt.subplots(ncols=4,figsize=(20,20))
for idx,  img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
##done at 12:10 PM on 25/07/2025
```


```python
len(data)
#split the data
train_size= int(len(data)*.7)
val_size= int(len(data)*.2)
test_size= int(len(data)*.1)
```


```python
train= data.take(train_size)
val= data.skip(train_size).take(val_size)
test= data.skip(train_size+val_size).take(test_size)
#done_010825
# len(train) should be 7 #done
# len(val) should be 2 #done
# len(test) should be 1 #done
```


```python
# Now we build the Deep Learning model 10:12 PM 010825
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

```


```python
model= Sequential()
```


```python
#
model.add(Conv2D(16, (3,3), 1, activation='relu', Input(shape=(256,256,3)))) 
model.add(MaxPooling2D())
#
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
#
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
#
model.add(Flatten())
#
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
          
```


```python
model = Sequential([
    Conv2D(16, (3,3), strides=1, activation='relu', input_shape=(256,256,3)), 
    MaxPooling2D(),
    Conv2D(32, (3,3), strides=1, activation='relu'),  
    MaxPooling2D(),
    Conv2D(16, (3,3), strides=1, activation='relu'), 
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])
```


```python
model.compile('adam', loss= tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
```


```python
model.summary()
```


```python
#3.2 train
logdir = r"F:\BIS_thesis\logs"
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
tensorboard_callback= tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist= model.fit(train,epochs=20,validation_data=val, callbacks=[tensorboard_callback])
```


```python
hist.history
```


```python
fig=plt.figure()
plt.plot(hist.history['loss'],color='teal',label='loss')
plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
fig.suptitle('Loss',fontsize=20)
plt.legend(loc="upper left")
plt.show()
```


```python
fig=plt.figure()
plt.plot(hist.history['loss'],color='teal',label='loss')
plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
fig.suptitle('Loss',fontsize=20)
plt.legend(loc="upper left")
plt.show()
```


```python
#4. Evaluate performance 29/08/2025\ 12:03 PM
#evaluate
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre= Precision()
re= Recall()
acc= BinaryAccuracy()

```


```python
len(test)
```


```python
for batch in test.as_numpy_iterator():
    X, y= batch
    yhat= model.predict(X)
    pre.update_state(y,yhat)
    re.update_state(y,yhat)
    acc.update_state(y,yhat) 
```


```python
print(f"Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}")
```


```python
#Test
#not for use
#test an image that the model has never seen before
import cv2
img= cv2.imread(r'F:\BIS_thesis\test_Leishmania_smear.jpg')
plt.imshow(cv2.medianBlur(img, 3))
plt.show()
```


```python
#just s without filter median
import cv2
img= cv2.imread(r'F:\BIS_thesis\test_Leishmania_smear.jpg')
#
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s_channel = hsv_img[:, :, 1]
# Split into H, S, V channels
h, s, v = cv2.split(hsv_img)
plt.figure(figsize=(12, 4))
plt.subplot(132)
plt.imshow(s, cmap='gray')
plt.title('Saturation Channel (S)')
plt.axis('off')
plt.show()

```


```python
# s_median 
filtered_s = cv2.medianBlur(s_channel, 3)
plt.imshow(filtered_s, cmap='gray')
plt.show()

```


```python
#resized_s_median
filtered_s_expanded = tf.expand_dims(filtered_s, axis=-1)
resize=tf.image.resize(filtered_s_expanded,(256,256))
resize_2d = tf.squeeze(resize)  # Remove channel dimension
plt.imshow(resize_2d.numpy(),cmap='gray')                             
plt.show()
```


```python
yhat = predict_single_channel(resize_2d/250,0)
```


```python
yhat=model.predict(resize)
yhat
```


```python
import numpy as np
import cv2

# Read image
img = cv2.imread(r'F:\BIS_thesis\test_Leishmania_smear.jpg')

# Convert to HSV and take Saturation channel
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s_channel = hsv_img[:, :, 1]   # shape (H,W)

# Resize to match model input
s_channel = cv2.resize(s_channel, (256, 256))

# Convert grayscale → RGB by repeating channels
s_channel_rgb = np.repeat(s_channel[..., np.newaxis], 3, axis=-1)  # (256,256,3)

# Add batch dimension
resize_p = np.expand_dims(s_channel_rgb, axis=0)  # (1,256,256,3)

# Normalize if required (most CNNs trained with values 0–1)
resize_p = resize_p.astype("float32") / 250.0

print(resize_p.shape)  # (1,256,256,3)

yhat = model.predict(resize_p)
print(yhat)
```


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load original image (BGR by default in OpenCV)
img_bgr = cv2.imread(r'F:\BIS_thesis\test_Leishmania_smear.jpg')

# Convert to RGB for plotting
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Convert to HSV and extract S channel
hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
s_channel = hsv_img[:, :, 1]

# 3. Resize to 256x256
s_resized = cv2.resize(s_channel, (256, 256))

# 4. Convert grayscale to 3-channel (since model expects 3 channels)
s_rgb = np.repeat(s_resized[..., np.newaxis], 3, axis=-1)

# 5. Add batch dimension + normalize
resize_p = np.expand_dims(s_rgb, axis=0).astype("float32") / 255.0

print("Final input shape for model:", resize_p.shape)

# --- Plot all steps ---
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# HSV → S channel
plt.subplot(1, 4, 2)
plt.imshow(s_channel, cmap="gray")
plt.title("Saturation Channel (S)")
plt.axis("off")

# Resized S
plt.subplot(1, 4, 3)
plt.imshow(s_resized, cmap="gray")
plt.title("Resized S (256x256)")
plt.axis("off")

# Final preprocessed (as RGB for model)
plt.subplot(1, 4, 4)
plt.imshow(resize_p[0])  # take first batch
plt.title("Final resize_p")
plt.axis("off")

plt.show()

```


```python
yhat = model.predict(resize_p)
print(yhat)
```


```python
model.summary()
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
```


```python
import cv2
import numpy as np

# 1. Load image (BGR)
img = cv2.imread(r'F:\BIS_thesis\test_Leishmania_smear.jpg')

# 2. Resize to 256x256
img_resized = cv2.resize(img, (256, 256))

# 3. Convert BGR → RGB
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# 4. Normalize pixel values
img_rgb = img_rgb.astype("float32") / 255.0

# 5. Add batch dimension
input_model = np.expand_dims(img_rgb, axis=0)  # shape: (1,256,256,3)

# 6. Predict
yhat = model.predict(input_model)
prediction = "Positive" if yhat[0][0] >= 0.5 else "Negative"
print("Prediction:", prediction)
print("Probability:", yhat[0][0])
 
# --- Plot all steps ---
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# HSV → S channel
plt.subplot(1, 4, 2)
plt.imshow(s_channel, cmap="gray")
plt.title("Saturation Channel (S)")
plt.axis("off")

# Resized S
plt.subplot(1, 4, 3)
plt.imshow(s_resized, cmap="gray")
plt.title("Resized S (256x256)")
plt.axis("off")

# Final preprocessed (as RGB for model)
plt.subplot(1, 4, 4)
plt.imshow(resize_p[0])  # take first batch
plt.title("Final resize_p")
plt.axis("off")

plt.show()


```


```python
# https://journals.sagepub.com/cms/10.1177/1040638719862599/asset/c05996c9-f3b8-42b5-b109-06d942e02033/assets/images/large/10.1177_1040638719862599-fig2.jpg
import cv2
import numpy as np

# 1. Load image (BGR)
img = cv2.imread(r'F:\BIS_thesis\test_leishmania_002.jpg')

# 2. Resize to 256x256
img_resized = cv2.resize(img, (256, 256))

# 3. Convert BGR → RGB
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# 4. Normalize pixel values
img_rgb = img_rgb.astype("float32") / 255.0

# 5. Add batch dimension
input_model = np.expand_dims(img_rgb, axis=0)  # shape: (1,256,256,3)

# 6. Predict
yhat = model.predict(input_model)
prediction = "Positive" if yhat[0][0] >= 0.5 else "Negative"
print("Prediction:", prediction)
print("Probability:", yhat[0][0])
 
# --- Plot all steps ---
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# HSV → S channel
plt.subplot(1, 4, 2)
plt.imshow(s_channel, cmap="gray")
plt.title("Saturation Channel (S)")
plt.axis("off")

# Resized S
plt.subplot(1, 4, 3)
plt.imshow(s_resized, cmap="gray")
plt.title("Resized S (256x256)")
plt.axis("off")

# Final preprocessed (as RGB for model)
plt.subplot(1, 4, 4)
plt.imshow(resize_p[0])  # take first batch
plt.title("Final resize_p")
plt.axis("off")
plt.show()



#save the model
import os
from tensorflow.keras.models import load_model
model.save(os.path.join(r"F:\BIS_thesis\models","my_model.h5"))
```


```python
#load the model
new_model= load_model(os.path.join(r"F:\BIS_thesis\models","my_model.h5")) 
```


```python
new_model
##  yhat_new= new_model.predict()
```


```python
# last modification for Dr Zakaria 28/09/25
#0_1
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load original image (BGR by default in OpenCV)
img_bgr = cv2.imread(r'F:\BIS_thesis\test\0_1.jpg')

# Convert to RGB for plotting
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Convert to HSV and extract S channel
hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
s_channel = hsv_img[:, :, 1]

# 3. Resize to 256x256
s_resized = cv2.resize(s_channel, (256, 256))

# 4. Convert grayscale to 3-channel (since model expects 3 channels)
s_rgb = np.repeat(s_resized[..., np.newaxis], 3, axis=-1)

# 5. Add batch dimension + normalize
resize_p = np.expand_dims(s_rgb, axis=0).astype("float32") / 255.0

print("Final input shape for model:", resize_p.shape)

# --- Plot all steps ---
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# HSV → S channel
plt.subplot(1, 4, 2)
plt.imshow(s_channel, cmap="gray")
plt.title("Saturation Channel (S)")
plt.axis("off")

# Resized S
plt.subplot(1, 4, 3)
plt.imshow(s_resized, cmap="gray")
plt.title("Resized S (256x256)")
plt.axis("off")

# Final preprocessed (as RGB for model)
plt.subplot(1, 4, 4)
plt.imshow(resize_p[0])  # take first batch
plt.title("Final resize_p")
plt.axis("off")

plt.show()
# 6. Predict
yhat = model.predict(resize_p)
prediction = "Positive" if yhat[0][0] >= 0.5 else "Negative"
print("Prediction:", prediction)
print("Probability:", yhat[0][0])
 
```


```python

```


```python
# last modification for Dr Zakaria 28/09/25
#0_2
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load original image (BGR by default in OpenCV)
img_bgr = cv2.imread(r'F:\BIS_thesis\test\0_2.jpg')

# Convert to RGB for plotting
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Convert to HSV and extract S channel
hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
s_channel = hsv_img[:, :, 1]

# 3. Resize to 256x256
s_resized = cv2.resize(s_channel, (256, 256))

# 4. Convert grayscale to 3-channel (since model expects 3 channels)
s_rgb = np.repeat(s_resized[..., np.newaxis], 3, axis=-1)

# 5. Add batch dimension + normalize
resize_p = np.expand_dims(s_rgb, axis=0).astype("float32") / 255.0

print("Final input shape for model:", resize_p.shape)

# --- Plot all steps ---
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# HSV → S channel
plt.subplot(1, 4, 2)
plt.imshow(s_channel, cmap="gray")
plt.title("Saturation Channel (S)")
plt.axis("off")

# Resized S
plt.subplot(1, 4, 3)
plt.imshow(s_resized, cmap="gray")
plt.title("Resized S (256x256)")
plt.axis("off")

# Final preprocessed (as RGB for model)
plt.subplot(1, 4, 4)
plt.imshow(resize_p[0])  # take first batch
plt.title("Final resize_p")
plt.axis("off")

plt.show()
# 6. Predict
yhat = model.predict(resize_p)
prediction = "Positive" if yhat[0][0] >= 0.5 else "Negative"
print("Prediction:", prediction)
print("Probability:", yhat[0][0])
 
```


```python

```


```python
# last modification for Dr Zakaria 28/09/25
#0_3
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load original image (BGR by default in OpenCV)
img_bgr = cv2.imread(r'F:\BIS_thesis\test\0_3.jpg')

# Convert to RGB for plotting
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Convert to HSV and extract S channel
hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
s_channel = hsv_img[:, :, 1]

# 3. Resize to 256x256
s_resized = cv2.resize(s_channel, (256, 256))

# 4. Convert grayscale to 3-channel (since model expects 3 channels)
s_rgb = np.repeat(s_resized[..., np.newaxis], 3, axis=-1)

# 5. Add batch dimension + normalize
resize_p = np.expand_dims(s_rgb, axis=0).astype("float32") / 255.0

print("Final input shape for model:", resize_p.shape)

# --- Plot all steps ---
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# HSV → S channel
plt.subplot(1, 4, 2)
plt.imshow(s_channel, cmap="gray")
plt.title("Saturation Channel (S)")
plt.axis("off")

# Resized S
plt.subplot(1, 4, 3)
plt.imshow(s_resized, cmap="gray")
plt.title("Resized S (256x256)")
plt.axis("off")

# Final preprocessed (as RGB for model)
plt.subplot(1, 4, 4)
plt.imshow(resize_p[0])  # take first batch
plt.title("Final resize_p")
plt.axis("off")

plt.show()
# 6. Predict
yhat = model.predict(resize_p)
prediction = "Positive" if yhat[0][0] >= 0.5 else "Negative"
print("Prediction:", prediction)
print("Probability:", yhat[0][0])
 
```


```python

```


```python
# last modification for Dr Zakaria 28/09/25
#0_4
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load original image (BGR by default in OpenCV)
img_bgr = cv2.imread(r'F:\BIS_thesis\test\0_4.jpg')

# Convert to RGB for plotting
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Convert to HSV and extract S channel
hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
s_channel = hsv_img[:, :, 1]

# 3. Resize to 256x256
s_resized = cv2.resize(s_channel, (256, 256))

# 4. Convert grayscale to 3-channel (since model expects 3 channels)
s_rgb = np.repeat(s_resized[..., np.newaxis], 3, axis=-1)

# 5. Add batch dimension + normalize
resize_p = np.expand_dims(s_rgb, axis=0).astype("float32") / 255.0

print("Final input shape for model:", resize_p.shape)

# --- Plot all steps ---
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# HSV → S channel
plt.subplot(1, 4, 2)
plt.imshow(s_channel, cmap="gray")
plt.title("Saturation Channel (S)")
plt.axis("off")

# Resized S
plt.subplot(1, 4, 3)
plt.imshow(s_resized, cmap="gray")
plt.title("Resized S (256x256)")
plt.axis("off")

# Final preprocessed (as RGB for model)
plt.subplot(1, 4, 4)
plt.imshow(resize_p[0])  # take first batch
plt.title("Final resize_p")
plt.axis("off")

plt.show()
# 6. Predict
yhat = model.predict(resize_p)
prediction = "Positive" if yhat[0][0] >= 0.5 else "Negative"
print("Prediction:", prediction)
print("Probability:", yhat[0][0])
 
```


```python

```


```python
# last modification for Dr Zakaria 28/09/25
#0_5
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load original image (BGR by default in OpenCV)
img_bgr = cv2.imread(r'F:\BIS_thesis\test\0_5.jpg')

# Convert to RGB for plotting
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Convert to HSV and extract S channel
hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
s_channel = hsv_img[:, :, 1]

# 3. Resize to 256x256
s_resized = cv2.resize(s_channel, (256, 256))

# 4. Convert grayscale to 3-channel (since model expects 3 channels)
s_rgb = np.repeat(s_resized[..., np.newaxis], 3, axis=-1)

# 5. Add batch dimension + normalize
resize_p = np.expand_dims(s_rgb, axis=0).astype("float32") / 255.0

print("Final input shape for model:", resize_p.shape)

# --- Plot all steps ---
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# HSV → S channel
plt.subplot(1, 4, 2)
plt.imshow(s_channel, cmap="gray")
plt.title("Saturation Channel (S)")
plt.axis("off")

# Resized S
plt.subplot(1, 4, 3)
plt.imshow(s_resized, cmap="gray")
plt.title("Resized S (256x256)")
plt.axis("off")

# Final preprocessed (as RGB for model)
plt.subplot(1, 4, 4)
plt.imshow(resize_p[0])  # take first batch
plt.title("Final resize_p")
plt.axis("off")

plt.show()
# 6. Predict
yhat = model.predict(resize_p)
prediction = "Positive" if yhat[0][0] >= 0.5 else "Negative"
print("Prediction:", prediction)
print("Probability:", yhat[0][0])
 
```


```python

```


```python
# last modification for Dr Zakaria 28/09/25
#13
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load original image (BGR by default in OpenCV)
img_bgr = cv2.imread(r'F:\BIS_thesis\test\13.jpg')

# Convert to RGB for plotting
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Convert to HSV and extract S channel
hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
s_channel = hsv_img[:, :, 1]

# 3. Resize to 256x256
s_resized = cv2.resize(s_channel, (256, 256))

# 4. Convert grayscale to 3-channel (since model expects 3 channels)
s_rgb = np.repeat(s_resized[..., np.newaxis], 3, axis=-1)

# 5. Add batch dimension + normalize
resize_p = np.expand_dims(s_rgb, axis=0).astype("float32") / 255.0

print("Final input shape for model:", resize_p.shape)

# --- Plot all steps ---
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# HSV → S channel
plt.subplot(1, 4, 2)
plt.imshow(s_channel, cmap="gray")
plt.title("Saturation Channel (S)")
plt.axis("off")

# Resized S
plt.subplot(1, 4, 3)
plt.imshow(s_resized, cmap="gray")
plt.title("Resized S (256x256)")
plt.axis("off")

# Final preprocessed (as RGB for model)
plt.subplot(1, 4, 4)
plt.imshow(resize_p[0])  # take first batch
plt.title("Final resize_p")
plt.axis("off")

plt.show()
# 6. Predict
yhat = model.predict(resize_p)
prediction = "Positive" if yhat[0][0] >= 0.5 else "Negative"
print("Prediction:", prediction)
print("Probability:", yhat[0][0])
 
```


```python

```


```python
# last modification for Dr Zakaria 28/09/25
#14
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load original image (BGR by default in OpenCV)
img_bgr = cv2.imread(r'F:\BIS_thesis\test\141.jpg')

# Convert to RGB for plotting
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Convert to HSV and extract S channel
hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
s_channel = hsv_img[:, :, 1]

# 3. Resize to 256x256
s_resized = cv2.resize(s_channel, (256, 256))

# 4. Convert grayscale to 3-channel (since model expects 3 channels)
s_rgb = np.repeat(s_resized[..., np.newaxis], 3, axis=-1)

# 5. Add batch dimension + normalize
resize_p = np.expand_dims(s_rgb, axis=0).astype("float32") / 255.0

print("Final input shape for model:", resize_p.shape)

# --- Plot all steps ---
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# HSV → S channel
plt.subplot(1, 4, 2)
plt.imshow(s_channel, cmap="gray")
plt.title("Saturation Channel (S)")
plt.axis("off")

# Resized S
plt.subplot(1, 4, 3)
plt.imshow(s_resized, cmap="gray")
plt.title("Resized S (256x256)")
plt.axis("off")

# Final preprocessed (as RGB for model)
plt.subplot(1, 4, 4)
plt.imshow(resize_p[0])  # take first batch
plt.title("Final resize_p")
plt.axis("off")

plt.show()
# 6. Predict
yhat = model.predict(resize_p)
prediction = "Positive" if yhat[0][0] >= 0.5 else "Negative"
print("Prediction:", prediction)
print("Probability:", yhat[0][0])
 
```


```python

```


```python
# last modification for Dr Zakaria 28/09/25
#11
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load original image (BGR by default in OpenCV)
img_bgr = cv2.imread(r'F:\BIS_thesis\test\11.jpg')

# Convert to RGB for plotting
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Convert to HSV and extract S channel
hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
s_channel = hsv_img[:, :, 1]

# 3. Resize to 256x256
s_resized = cv2.resize(s_channel, (256, 256))

# 4. Convert grayscale to 3-channel (since model expects 3 channels)
s_rgb = np.repeat(s_resized[..., np.newaxis], 3, axis=-1)

# 5. Add batch dimension + normalize
resize_p = np.expand_dims(s_rgb, axis=0).astype("float32") / 255.0

print("Final input shape for model:", resize_p.shape)

# --- Plot all steps ---
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# HSV → S channel
plt.subplot(1, 4, 2)
plt.imshow(s_channel, cmap="gray")
plt.title("Saturation Channel (S)")
plt.axis("off")

# Resized S
plt.subplot(1, 4, 3)
plt.imshow(s_resized, cmap="gray")
plt.title("Resized S (256x256)")
plt.axis("off")

# Final preprocessed (as RGB for model)
plt.subplot(1, 4, 4)
plt.imshow(resize_p[0])  # take first batch
plt.title("Final resize_p")
plt.axis("off")

plt.show()
# 6. Predict
yhat = model.predict(resize_p)
prediction = "Positive" if yhat[0][0] >= 0.5 else "Negative"
print("Prediction:", prediction)
print("Probability:", yhat[0][0])
 
```


```python

```


```python
# last modification for Dr Zakaria 28/09/25
#12
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load original image (BGR by default in OpenCV)
img_bgr = cv2.imread(r'F:\BIS_thesis\test\12.jpg')

# Convert to RGB for plotting
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Convert to HSV and extract S channel
hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
s_channel = hsv_img[:, :, 1]

# 3. Resize to 256x256
s_resized = cv2.resize(s_channel, (256, 256))

# 4. Convert grayscale to 3-channel (since model expects 3 channels)
s_rgb = np.repeat(s_resized[..., np.newaxis], 3, axis=-1)

# 5. Add batch dimension + normalize
resize_p = np.expand_dims(s_rgb, axis=0).astype("float32") / 255.0

print("Final input shape for model:", resize_p.shape)

# --- Plot all steps ---
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# HSV → S channel
plt.subplot(1, 4, 2)
plt.imshow(s_channel, cmap="gray")
plt.title("Saturation Channel (S)")
plt.axis("off")

# Resized S
plt.subplot(1, 4, 3)
plt.imshow(s_resized, cmap="gray")
plt.title("Resized S (256x256)")
plt.axis("off")

# Final preprocessed (as RGB for model)
plt.subplot(1, 4, 4)
plt.imshow(resize_p[0])  # take first batch
plt.title("Final resize_p")
plt.axis("off")

plt.show()
# 6. Predict
yhat = model.predict(resize_p)
prediction = "Positive" if yhat[0][0] >= 0.5 else "Negative"
print("Prediction:", prediction)
print("Probability:", yhat[0][0])
 
```
