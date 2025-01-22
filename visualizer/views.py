from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
import io
import base64
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

def home(request):
    """Render the home page to select the method."""
    return render(request, 'index.html')

def plot_constraints(constraints, bounds, feasible_region=None, optimal_vertex=None):
    """Plots the constraints, feasible region, and optimal solution."""
    x = np.linspace(bounds[0], bounds[1], 400)
    plt.figure(figsize=(10, 8))

    # Plot constraints as lines
    for coeff, b in constraints:
        if coeff[1] != 0:  # Plot lines with a slope
            y = (b - coeff[0] * x) / coeff[1]
            plt.plot(x, y, label=f"{coeff[0]}x1 + {coeff[1]}x2 ≤ {b}")
        else:  # Vertical line
            x_val = b / coeff[0]
            plt.axvline(x_val, color='r', linestyle='--', label=f"x1 = {x_val}")

    # Highlight feasible region
    if feasible_region is not None and len(feasible_region) > 0:
        hull = ConvexHull(feasible_region)
        polygon = Polygon(feasible_region[hull.vertices], closed=True, color='lightgreen', alpha=0.5, label='Feasible Region')
        plt.gca().add_patch(polygon)

    # Highlight corner points
    if feasible_region is not None:
        for point in feasible_region:
            plt.plot(point[0], point[1], 'bo')  # Mark corners

    # Highlight the optimal solution
    if optimal_vertex is not None:
        plt.plot(optimal_vertex[0], optimal_vertex[1], 'ro', label='Optimal Solution')

    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Linear Programming: Graphical Method")
    plt.legend()
    plt.grid()

    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return image_base64

@csrf_exempt
def solve_linear_program(request):
    if request.method == 'POST':
        try:
            method = request.POST.get('method')
            c = list(map(float, request.POST.get('objective_function').split(',')))
            A = [list(map(float, x.split(','))) for x in request.POST.getlist('constraints_left')]
            b = list(map(float, request.POST.getlist('constraints_right')))

            bounds = [0, max(b)]
            constraints = list(zip(A, b))

            # Solve using vertices of the feasible region
            vertices = []
            num_constraints = len(A)
            for i in range(num_constraints):
                for j in range(i + 1, num_constraints):
                    A_ = np.array([A[i], A[j]])
                    b_ = np.array([b[i], b[j]])
                    try:
                        vertex = np.linalg.solve(A_, b_)
                        if all(np.dot(A, vertex) <= b) and all(vertex >= 0):
                            vertices.append(vertex)
                    except np.linalg.LinAlgError:
                        continue

            feasible_vertices = np.unique(vertices, axis=0)
            
            if len(feasible_vertices) > 0:
                z_values = [np.dot(c, v) for v in feasible_vertices]
                optimal_value = max(z_values) if method == 'max' else min(z_values)
                optimal_vertex = feasible_vertices[np.argmax(z_values)] if method == 'max' else feasible_vertices[np.argmin(z_values)]

                graph_image = plot_constraints(constraints, bounds, feasible_region=feasible_vertices, optimal_vertex=optimal_vertex)
                return JsonResponse({
                    'optimal_point': optimal_vertex.tolist(),
                    'optimal_value': optimal_value,
                    'graph': graph_image
                })
            else:
                return JsonResponse({'error': 'No feasible region found.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return render(request, 'solve_linear_program.html')