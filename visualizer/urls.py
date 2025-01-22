from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home route for the empty path
    path('solve/', views.solve_linear_program, name='solve_linear_program'),  # Solver route
]
