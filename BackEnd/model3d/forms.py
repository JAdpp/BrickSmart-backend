from django import forms

class PromptForm(forms.Form):
    prompt = forms.CharField(label="输入你想搭建的任何事物", max_length=100, widget=forms.TextInput(attrs={
        'placeholder': '例如: 一架蓝色的飞机...',
        'autocomplete': 'off'
    }))

class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='上传参考照片')