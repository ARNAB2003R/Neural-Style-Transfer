# transfer/views.py

from django.shortcuts import render
from .forms import ImageUploadForm
from .models import Image
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image as PilImage
import io
import cv2
import os

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_image(img_array):
    img = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img = img / 255.0  # Normalize the image to [0, 1] range
    img = img[tf.newaxis, :]
    return img

def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            content_image = form.cleaned_data['content_image']
            style_image = form.cleaned_data['style_image']

            content_image_np = np.frombuffer(content_image.read(), np.uint8)
            content_image_np = cv2.imdecode(content_image_np, cv2.IMREAD_COLOR)
            content_image_np = cv2.cvtColor(content_image_np, cv2.COLOR_BGR2RGB)
            content_image_tf = load_image(content_image_np)

            style_image_np = np.frombuffer(style_image.read(), np.uint8)
            style_image_np = cv2.imdecode(style_image_np, cv2.IMREAD_COLOR)
            style_image_np = cv2.cvtColor(style_image_np, cv2.COLOR_BGR2RGB)
            style_image_tf = load_image(style_image_np)

            # Debugging prints
            print(f"Content image shape: {content_image_np.shape}")
            print(f"Style image shape: {style_image_np.shape}")
            print(f"Content tensor shape: {content_image_tf.shape}")
            print(f"Style tensor shape: {style_image_tf.shape}")

            stylized_image = model(tf.constant(content_image_tf), tf.constant(style_image_tf))[0]
            stylized_image = np.squeeze(stylized_image)

            pil_img = PilImage.fromarray((stylized_image * 255).astype(np.uint8))
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG')
            buffer.seek(0)

            image_instance = Image(content_image=content_image, style_image=style_image)
            image_instance.result_image.save('stylized_image.jpg', buffer, save=False)
            image_instance.save()

            return render(request, 'result.html', {'image': image_instance})

    else:
        form = ImageUploadForm()
    return render(request, 'index.html', {'form': form})
