from django import forms

class ImageUploadForm(forms.Form):
    content_image = forms.ImageField()
    style_image = forms.ImageField()