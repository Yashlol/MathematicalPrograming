from django.urls import path
from . import views

urlpatterns = [
    path('', views.solve_linear_program, name='solve_linear_program'),
]
