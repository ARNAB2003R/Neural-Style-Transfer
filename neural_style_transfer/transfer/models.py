from django.db import models

# Create your models here.

class Image(models.Model):
    content_image = models.ImageField(upload_to='images/')
    style_image = models.ImageField(upload_to='images/')
    result_image = models.ImageField(upload_to='images/', blank=True, null=True)
