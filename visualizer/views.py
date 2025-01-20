from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import io
import base64
import json
import re
import matplotlib
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Use non-interactive backend for matplotlib
matplotlib.use('Agg')

@ensure_csrf_cookie
def home(request):
    """Render the home page with CSRF token."""
    return render(request, 'home.html')

def parse_expression(expr):
    """Parse linear expression into coefficients."""
    try:
        expr = expr.replace(' ', '')
        terms = re.findall(r'[+-]?\d*\.?\d*[xy]|[+-]?\d+\.?\d*', expr)
        coefficients = [0, 0]
        for term in terms:
            if 'x' in term:
                coeff = term.replace('x', '')
                coeff = 1 if coeff in ['+', ''] else -1 if coeff == '-' else float(coeff)
                coefficients[0] = coeff
            elif 'y' in term:
                coeff = term.replace('y', '')
                coeff = 1 if coeff in ['+', ''] else -1 if coeff == '-' else float(coeff)
                coefficients[1] = coeff
        return coefficients
    except Exception as e:
        logger.error(f"Error parsing expression '{expr}': {str(e)}")
        raise ValueError(f"Invalid expression format: {expr}")

def parse_constraint(expr):
    """Parse constraint into coefficients and right-hand side."""
    try:
        parts = re.split(r'<=|>=|=', expr)
        if len(parts) != 2:
            raise ValueError(f"Invalid constraint format: {expr}")
        left_side = parts[0]
        right_side = float(parts[1])
        coefficients = parse_expression(left_side)
        return coefficients + [right_side]
    except Exception as e:
        logger.error(f"Error parsing constraint '{expr}': {str(e)}")
        raise ValueError(f"Invalid constraint format: {expr}")

def get_vertices(A, b):
    """Calculate vertices of the feasible region."""
    try:
        vertices = []
        num_constraints = len(b)
        for i in range(num_constraints):
            for j in range(i + 1, num_constraints):
                A_sub = np.array([A[i], A[j]])
                b_sub = np.array([b[i], b[j]])
                if np.linalg.det(A_sub) != 0:  # Ensure not parallel
                    vertex = np.linalg.solve(A_sub, b_sub)
                    if all(np.dot(A, vertex) <= b + 1e-10) and all(vertex >= -1e-10):  # Feasibility check
                        vertices.append(vertex)
        # Add origin if feasible
        if all(np.dot(A, [0, 0]) <= b):
            vertices.append([0, 0])
        return np.array(vertices)
    except Exception as e:
        logger.error(f"Error calculating vertices: {str(e)}")
        raise ValueError("Failed to calculate feasible region vertices")

@require_http_methods(["POST"])
def solve_lp(request):
    """Handle the LP solving request."""
    try:
        # Parse JSON data
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format in request'}, status=400)

        # Validate required fields
        required_fields = ['optimization_type', 'objective', 'constraints']
        if not all(field in data for field in required_fields):
            return JsonResponse({'error': 'Missing required fields'}, status=400)

        # Parse input data
        objective = parse_expression(data['objective'])
        constraints = [parse_constraint(c) for c in data['constraints'].split(';') if c.strip()]
        if not constraints:
            return JsonResponse({'error': 'No valid constraints provided'}, status=400)

        constraints_matrix = np.array(constraints)

        # Set up the optimization problem
        c = np.array(objective)
        if data['optimization_type'] == 'maximize':
            c = -c

        A = constraints_matrix[:, :2]
        b = constraints_matrix[:, 2]
        bounds = [(0, None), (0, None)]

        # Solve the LP problem
        res = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        if not res.success:
            return JsonResponse({'error': f'Optimization failed: {res.message}'}, status=400)

        # Create visualization
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(16, 12))

        colors = ['#10B981', '#3B82F6', '#8B5CF6', '#F59E0B', '#EC4899']
        max_x = max(20, np.max(b) / 2)
        x = np.linspace(0, max_x, 400)

        # Plot constraints
        for i, (a_i, b_i) in enumerate(zip(A, b)):
            if abs(a_i[1]) > 1e-10:  # Non-vertical line
                y = (b_i - a_i[0] * x) / a_i[1]
                plt.plot(x, y, color=colors[i % len(colors)], label=f'{a_i[0]:g}x + {a_i[1]:g}y ≤ {b_i:g}', linewidth=2.5)
            else:  # Vertical line
                plt.axvline(x=b_i / a_i[0], color=colors[i % len(colors)], label=f'x = {b_i / a_i[0]:g}', linewidth=2.5)

        # Plot feasible region
        vertices = get_vertices(A, b)
        if len(vertices) > 2:
            hull = ConvexHull(vertices)
            feasible_polygon = Polygon(vertices[hull.vertices], alpha=0.2, color='#94A3B8', label='Feasible Region')
            ax.add_patch(feasible_polygon)

        # Plot optimal point
        plt.plot(res.x[0], res.x[1], 'o', color='#F472B6', markersize=15, label='Optimal Point')
        plt.annotate(f'({res.x[0]:.2f}, {res.x[1]:.2f})', (res.x[0], res.x[1]), xytext=(10, 10), textcoords='offset points',
                     color='white', fontsize=12, fontweight='bold',
                     bbox=dict(facecolor='#1F2937', edgecolor='#4B5563', alpha=0.8))

        # Customize plot appearance
        ax.grid(True, linestyle='--', alpha=0.2, color='#4B5563')
        ax.set_facecolor('#111827')
        fig.patch.set_facecolor('#111827')

        plt.xlabel('x', color='white', fontsize=14, fontweight='medium')
        plt.ylabel('y', color='white', fontsize=14, fontweight='medium')
        plt.title('Linear Programming Solution', color='white', fontsize=16, pad=20, fontweight='bold')

        legend = plt.legend(loc='upper left', frameon=True, facecolor='#1F2937', edgecolor='#4B5563', fontsize=12)
        plt.setp(legend.get_texts(), color='white')

        for spine in ax.spines.values():
            spine.set_color('#4B5563')

        ax.tick_params(colors='white', labelsize=12)

        # Set plot limits
        max_val = max(np.max(b), np.max(res.x)) * 1.2
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.tight_layout()

        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor='#111827', edgecolor='none')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Calculate optimal value
        optimal_value = float(-res.fun if data['optimization_type'] == 'maximize' else res.fun)

        return JsonResponse({
            'solution': res.x.tolist(),
            'optimal_value': optimal_value,
            'image': image_base64
        })

    except Exception as e:
        logger.error(f"Error solving LP: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)
