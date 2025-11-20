"""
Síntesis de Mecanismo de 4 Barras para Trayectoria del Pie
===========================================================

Este script funciona tanto en:
- Entorno local (Windows/Linux/Mac): Abre una ventana con la animación interactiva
- Google Colab: Genera una animación HTML embebida en el notebook

USO EN GOOGLE COLAB:
1. Subir este archivo o copiar el código en una celda
2. Ejecutar la celda
3. La optimización tomará varios minutos
4. La animación se mostrará como HTML interactivo en el notebook

USO EN LOCAL:
1. Ejecutar: python gemini1.py
2. La optimización tomará varios minutos
3. Se abrirá una ventana con la animación

REQUISITOS:
- numpy
- matplotlib
- scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
import sys

# Detectar si estamos en Google Colab
try:
    import google.colab
    from IPython.display import HTML, display
    IN_COLAB = True
    print("Ejecutando en Google Colab")
except:
    IN_COLAB = False
    print("Ejecutando en entorno local")

# --- 1. Definiciones Iniciales y Puntos Objetivo ---

# Puntos de precisión (Objetivo)
# (0, 0) — Initial contact
# (10, 0) — Mid-stance
# (40, 0) — Toe-off
# (50, 5) — Initial swing
# (60, 12) — Mid-swing
# (70, 0) — Contacto final (NUEVO PUNTO)
P_target = np.array([
    [0, 0],
    [10, 0],
    [40, 0],
    [50, 5],
    [60, 12],
    [70, 0]  # NUEVO PUNTO: Contacto final
])

# Número de puntos de precisión (actualizado a 6)
N_POINTS = len(P_target)

# --- 2. Funciones Auxiliares (Análisis Cinemático) ---

def solve_kinematics(mech_params, phi, mode=1):
    """
    Resuelve la cinemática de lazo cerrado para un ángulo de entrada (phi) dado.
    Encuentra los ángulos theta_3 y theta_4.
    
    Parámetros:
    mech_params = [L1, L2, L3, L4]
    phi = ángulo de entrada (theta_2)
    mode = modo de ensamblaje (+1 o -1)

    Retorna:
    (theta_3, theta_4) o (None, None) si es imposible
    """
    L1, L2, L3, L4 = mech_params

    # Resolvemos A*cos(theta_3) + B*sin(theta_3) = C
    A = 2 * L3 * (L2 * np.cos(phi) - L1)
    B = 2 * L3 * L2 * np.sin(phi)
    C = L4*2 - L12 - L22 - L3*2 + 2 * L1 * L2 * np.cos(phi)

    # Solución para theta_3
    R = np.sqrt(A*2 + B*2)
    if R == 0:
        return None, None
        
    cos_gamma = A / R
    sin_gamma = B / R
    gamma = np.arctan2(sin_gamma, cos_gamma)

    if C / R > 1.0 or C / R < -1.0:
        return None, None  # No hay solución real (el lazo no se cierra)

    theta_3 = gamma + mode * np.arccos(C / R)

    # Solución para theta_4
    # L4*cos(th_4) = L2*cos(phi) + L3*cos(th_3) - L1
    # L4*sin(th_4) = L2*sin(phi) + L3*sin(th_3)
    Y_4 = L2 * np.sin(phi) + L3 * np.sin(theta_3)
    X_4 = L2 * np.cos(phi) + L3 * np.cos(theta_3) - L1
    
    theta_4 = np.arctan2(Y_4, X_4)

    return theta_3, theta_4

def calculate_coupler_point(mech_params, angles):
    """
    Calcula la posición del punto acoplador (P).
    
    Parámetros:
    mech_params = [L2, L5, delta] (parámetros del acoplador)
    angles = [phi, theta_3] (ángulos de entrada y acoplador)
    """
    L2, L5, delta = mech_params
    phi, theta_3 = angles

    # Px = L2*cos(phi) + L5*cos(theta_3 + delta)
    # Py = L2*sin(phi) + L5*sin(theta_3 + delta)
    x = L2 * np.cos(phi) + L5 * np.cos(theta_3 + delta)
    y = L2 * np.sin(phi) + L5 * np.sin(theta_3 + delta)
    
    return np.array([x, y])

def check_crank_rocker(L1, L2, L3, L4):
    """
    Verifica si el mecanismo es un Grashof de tipo manivela-balancín
    (asumiendo L2 como la manivela de entrada).
    """
    links = np.array([L1, L2, L3, L4])
    L_S = np.min(links)  # Eslabón más corto
    L_L = np.max(links)  # Eslabón más largo
    
    # Condición de Grashof
    if (L_S + L_L) > (np.sum(links) - L_S - L_L):
        return False  # No es Grashof
    
    # Condición de Manivela-Balancín (L2 debe ser el más corto)
    if L_S == L2:
        return True
        
    return False


# --- 3. Función de Costo para Optimización ---

def cost_function(params):
    """
    Función de costo que la optimización intentará minimizar.
    'params' es un vector de 13 variables (antes 11):
    [L1, L2, L3, L4, L5, delta, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6]
    """
    
    # 1. Desempaquetar parámetros
    L1, L2, L3, L4, L5, delta = params[:6]
    phis = params[6:]
    
    link_params = [L1, L2, L3, L4]
    coupler_params = [L2, L5, delta]
    
    total_error = 0
    
    # 2. Penalizaciones (Restricciones)
    
    # Penalización 1: Debe ser un mecanismo Grashof manivela-balancín
    # para asegurar una rotación de entrada continua.
    if not check_crank_rocker(L1, L2, L3, L4):
        return 1e9  # Error muy alto si no es manivela-balancín

    # Penalización 2: Los ángulos de entrada deben estar en orden
    if np.any(np.diff(phis) <= 0):
        return 1e9  # Error muy alto si los ángulos no son monotónicos
        
    # 3. Calcular error en los puntos de precisión
    for i in range(N_POINTS):  # Ahora 6 puntos
        phi_j = phis[i]
        P_target_j = P_target[i]
        
        # a. Resolver cinemática
        theta_3, theta_4 = solve_kinematics(link_params, phi_j, mode=1)
        
        if theta_3 is None:
            return 1e9  # Penalización alta si el lazo no se cierra
            
        # b. Calcular posición del punto acoplador generado
        P_gen_j = calculate_coupler_point(coupler_params, [phi_j, theta_3])
        
        # c. Acumular el error cuadrático
        error_sq = np.sum((P_gen_j - P_target_j)**2)
        total_error += error_sq
        
    return total_error


# --- 4. Ejecución de la Síntesis (Optimización) ---

def run_synthesis():
    """
    Configura y ejecuta el optimizador de evolución diferencial.
    """
    print("Iniciando síntesis de mecanismo...")
    print(f"Optimizando para {N_POINTS} puntos de precisión")
    
    # Límites para las 13 variables (antes 11):
    # [L1, L2, L3, L4, L5, delta, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6]
    # Las longitudes se basan en la escala de los puntos (0-70)
    bounds = [
        (10, 100),  # L1 (Eslabón de tierra)
        (5, 50),    # L2 (Eslabón de entrada/manivela)
        (20, 150),  # L3 (Eslabón acoplador)
        (20, 150),  # L4 (Eslabón seguidor)
        (20, 150),  # L5 (Acoplador P-A)
        (-np.pi, np.pi), # delta (Ángulo del acoplador)
        (0, 2*np.pi),    # phi_1
        (0.1, 2*np.pi),  # phi_2
        (0.2, 2*np.pi),  # phi_3
        (0.3, 2*np.pi),  # phi_4
        (0.4, 2*np.pi),  # phi_5
        (0.5, 2*np.pi)   # phi_6 (NUEVO ÁNGULO)
    ]
    
    # Ejecutar el optimizador
    # Aumentamos ligeramente maxiter y popsize para mejor convergencia con más puntos
    result = differential_evolution(
        cost_function, 
        bounds, 
        strategy='best1bin', 
        maxiter=600,  # Aumentado de 500
        popsize=25,   # Aumentado de 20
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,
        workers=-1  # Usar todos los núcleos de CPU
    )
    
    if result.success:
        print("\nOptimización completada exitosamente.")
        print(f"Error (Suma de Cuadrados): {result.fun}")
        return result.x
    else:
        print("\nLa optimización no convergió a una solución.")
        print(f"Mensaje: {result.message}")
        return None

# --- 5. Post-Procesamiento y Visualización ---

def plot_results(optimal_params):
    """
    Toma los parámetros óptimos y genera los gráficos de resultados.
    """
    if optimal_params is None:
        print("No se encontraron parámetros óptimos para graficar.")
        return

    # 1. Desempaquetar parámetros
    L1, L2, L3, L4, L5, delta = optimal_params[:6]
    phis = optimal_params[6:]
    
    mech_link_params = [L1, L2, L3, L4]
    mech_coupler_params = [L2, L5, delta]
    
    # 2. Calcular los 6 puntos de precisión generados
    P_gen = []
    for phi in phis:
        theta_3, _ = solve_kinematics(mech_link_params, phi, mode=1)
        if theta_3 is not None:
            P_gen.append(calculate_coupler_point(mech_coupler_params, [phi, theta_3]))
    P_gen = np.array(P_gen)

    # 3. Calcular la curva completa del acoplador
    phi_range = np.linspace(0, 2 * np.pi, 360) # Rango completo de la manivela
    coupler_curve = []
    for phi in phi_range:
        theta_3, _ = solve_kinematics(mech_link_params, phi, mode=1)
        if theta_3 is not None:
            coupler_curve.append(calculate_coupler_point(mech_coupler_params, [phi, theta_3]))
    coupler_curve = np.array(coupler_curve)
    
    # Aplicar desplazamiento de -100 en Y a la curva del acoplador y P_gen
    y_offset = -100
    if coupler_curve.size > 0:
        coupler_curve[:, 1] += y_offset
    if P_gen.size > 0:
        P_gen[:, 1] += y_offset

    # 4. Calcular RMSE
    rmse = np.sqrt(np.mean(np.sum((P_gen - P_target)**2, axis=1)))
    
    print("\n--- Parámetros del Mecanismo Optimizado ---")
    print(f"L1 (Tierra): {L1:.3f}")
    print(f"L2 (Manivela): {L2:.3f}")
    print(f"L3 (Acoplador): {L3:.3f}")
    print(f"L4 (Balancín): {L4:.3f}")
    print(f"L5 (Punto P-A): {L5:.3f}")
    print(f"delta (Ángulo P): {np.degrees(delta):.3f}°")
    print("\n--- Ángulos de Entrada (phi) ---")
    print(f"Ángulos (grados): {np.degrees(phis)}")
    print(f"\n--- Error ---")
    print(f"RMSE (Root Mean Square Error): {rmse:.4f}")

    # 5. Preparar datos para animación - OPTIMIZADO PARA COLAB
    # Reducir el número de frames para evitar el límite de tamaño
    if IN_COLAB:
        n_frames = 120  # Reducido de 360 para Colab
        print(f"Usando {n_frames} frames para animación (optimizado para Colab)")
    else:
        n_frames = 360  # Completo para entorno local
        
    phi_range_anim = np.linspace(0, 2 * np.pi, n_frames)

    # Precomputar posiciones para cada frame (A, B, P)
    # Aplicar desplazamiento de -100 en Y
    y_offset = -100
    A0 = np.array([0.0, 0.0 + y_offset])
    B0 = np.array([L1, 0.0 + y_offset])

    A_list = []
    B_list = []
    P_list = []
    valid_idx = []

    for i, phi in enumerate(phi_range_anim):
        theta_3, theta_4 = solve_kinematics(mech_link_params, phi, mode=1)
        if theta_3 is None:
            # Mantener None para indicar frame inválido
            A_list.append(None)
            B_list.append(None)
            P_list.append(None)
            continue

        A_pt = np.array([L2 * np.cos(phi), L2 * np.sin(phi) + y_offset])
        B_pt = np.array([L1 + L4 * np.cos(theta_4), L4 * np.sin(theta_4) + y_offset])
        P_pt = calculate_coupler_point(mech_coupler_params, [phi, theta_3])
        P_pt[1] += y_offset  # Desplazar P en Y

        A_list.append(A_pt)
        B_list.append(B_pt)
        P_list.append(P_pt)
        valid_idx.append(i)

    # Si no hay frames válidos abortar
    if len(valid_idx) == 0:
        print("No hay configuraciones válidas para animar.")
        return

    # Convertir a arrays para facilidad (rellenar con NaN donde no válido)
    A_arr = np.full((n_frames, 2), np.nan)
    B_arr = np.full((n_frames, 2), np.nan)
    P_arr = np.full((n_frames, 2), np.nan)

    for i in valid_idx:
        A_arr[i] = A_list[i]
        B_arr[i] = B_list[i]
        P_arr[i] = P_list[i]

    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(12, 9))

    # Aplicar desplazamiento a los puntos objetivo para visualización
    P_target_display = P_target.copy()
    P_target_display[:, 1] += y_offset

    # Dibujar trayectoria completa (opcional, en fondo tenue)
    if coupler_curve.size > 0:
        ax.plot(coupler_curve[:, 0], coupler_curve[:, 1], color='lightgreen',
                linewidth=1.5, alpha=0.5, label='Trayectoria completa (fondo)')

    # Puntos objetivo y puntos generados
    ax.plot(P_target_display[:, 0], P_target_display[:, 1], 'bo', markersize=8, label='Puntos objetivo')
    if P_gen.size > 0:
        ax.plot(P_gen[:, 0], P_gen[:, 1], 'r*', markersize=12, label='Puntos generados')

    # Etiquetar los puntos objetivo para mejor identificación
    point_labels = ['Contacto inicial', 'Medio apoyo', 'Despegue talón', 
                   'Inicio oscilación', 'Medio oscilación', 'Contacto final']
    for i, (x, y) in enumerate(P_target_display):
        ax.annotate(point_labels[i], (x, y), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, alpha=0.7)

    # Artistas para la animación
    trace_line, = ax.plot([], [], 'g-', linewidth=2, label='Trayectoria dibujada')
    current_point, = ax.plot([], [], 'ro', markersize=8, label='Punto P actual')
    mechanism_line, = ax.plot([], [], 'k--', linewidth=1.5, alpha=0.8, label='Mecanismo')

    # Triángulo acoplador (Polygon) - inicial vacío
    coupler_triangle = Polygon([[0, 0], [0, 0], [0, 0]], closed=True, fc='cyan', ec='black', alpha=0.3)
    ax.add_patch(coupler_triangle)

    # Pivotes fijos
    ax.plot([A0[0], B0[0]], [A0[1], B0[1]], 'k^', markersize=10, label='Pivotes fijos')

    ax.set_title('Animación: trayectoria del acoplador y movimiento del mecanismo\n(6 puntos de precisión incluyendo contacto final)')
    ax.set_xlabel('Posición X (mm)')
    ax.set_ylabel('Posición Y (mm)')
    ax.grid(True)
    ax.axis('equal')
    ax.legend(loc='best')

    # Limites levemente por encima del recorrido
    all_pts = np.vstack((coupler_curve, P_target_display)) if coupler_curve.size > 0 else P_target_display
    margin = 120
    xmin, ymin = np.min(all_pts, axis=0) - margin
    xmax, ymax = np.max(all_pts, axis=0) + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Indices de frames válidos para trazar la trayectoria progresiva
    valid_frames = np.array(valid_idx)

    # Funciones de inicialización y actualización
    drawn_x = []
    drawn_y = []

    def init():
        trace_line.set_data([], [])
        current_point.set_data([], [])
        mechanism_line.set_data([], [])
        coupler_triangle.set_xy([[0, 0], [0, 0], [0, 0]])
        return trace_line, current_point, mechanism_line, coupler_triangle

    def update(frame):
        # Si frame no válido, mantener último estado
        if np.isnan(P_arr[frame, 0]):
            return trace_line, current_point, mechanism_line, coupler_triangle

        # Actualizar trayectoria dibujada: sólo los puntos válidos hasta este frame
        valid_up_to = ~np.isnan(P_arr[:frame + 1, 0])
        trace_line.set_data(P_arr[:frame + 1][valid_up_to, 0], P_arr[:frame + 1][valid_up_to, 1])

        # Punto actual (asegurar que sea una secuencia)
        current_point.set_data([P_arr[frame, 0]], [P_arr[frame, 1]])

        # Mecanismo: A0 -> A -> B -> B0 -> A0
        A_pt = A_arr[frame]
        B_pt = B_arr[frame]
        mech_x = [A0[0], A_pt[0], B_pt[0], B0[0], A0[0]]
        mech_y = [A0[1], A_pt[1], B_pt[1], B0[1], A0[1]]
        mechanism_line.set_data(mech_x, mech_y)

        # Triángulo acoplador (A, B, P)
        coupler_triangle.set_xy([A_pt.tolist(), B_pt.tolist(), P_arr[frame].tolist()])

        return trace_line, current_point, mechanism_line, coupler_triangle

    # Crear animación - CON CONFIGURACIÓN OPTIMIZADA PARA COLAB
    print("Generando animación...")
    
    # Configuración especial para Google Colab
    if IN_COLAB:
        # Aumentar el límite de tamaño para animaciones embebidas
        plt.rcParams['animation.embed_limit'] = 50.0  # 50 MB en lugar de 20 MB
        print("Configurando límite de animación a 50 MB para Colab")
        
        # Usar menos frames y compresión
        anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                            interval=50, blit=False, repeat=True,
                            cache_frame_data=False)  # No cachear frames para ahorrar memoria
    else:
        # Configuración normal para entorno local
        anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                            interval=30, blit=False, repeat=True)

    # Mostrar animación según el entorno
    if IN_COLAB:
        # En Colab, convertir a HTML y mostrar
        print("Generando animación HTML para Google Colab...")
        try:
            plt.close()  # Cerrar la figura para evitar mostrarla estáticamente
            html_content = anim.to_jshtml()
            print(f"Tamaño de la animación: {len(html_content)} bytes")
            return HTML(html_content)
        except Exception as e:
            print(f"Error al generar animación HTML: {e}")
            print("Mostrando gráfico estático como alternativa...")
            plt.show()
            return None
    else:
        # En local, mostrar ventana normal
        plt.show()
        return anim


# --- 6. Ejecución Principal ---
if _name_ == "_main_":
    optimal_parameters = run_synthesis()
    if optimal_parameters is not None:
        result = plot_results(optimal_parameters)
        # En Colab, mostrar el resultado HTML
        if IN_COLAB and result is not None:
            display(result)
