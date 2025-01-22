from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('visualizer.urls')),  # Replace 'myapp' with your app's name
]
