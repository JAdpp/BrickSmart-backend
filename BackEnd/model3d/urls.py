from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import generate_model, generate_model_image, generate_lego_tutorial, lego_storytelling

urlpatterns = [
    path("model/", generate_model, name='generate_model'),
    path("model/image/", generate_model_image, name='generate_model_image'),
    path('lego_tutorial/', generate_lego_tutorial, name='generate_lego_tutorial'),
    path('lego_storytelling/', lego_storytelling, name='lego_storytelling')
]

# 在开发环境下添加媒体文件的服务
#if settings.DEBUG:
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)