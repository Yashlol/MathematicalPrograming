from django.contrib import admin
from django.urls import path,include
from visualizer import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',include('visualizer.urls')),
]
