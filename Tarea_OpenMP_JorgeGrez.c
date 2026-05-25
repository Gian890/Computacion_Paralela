#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Función que aproxima Zeta de Riemann dependiente de s y k
double Riemann_Zeta(double s, int k) {
    if (k == 0) return 0.0;
    
    double result = 0.0;
    
    for (int i = 1; i <= k; i++) {
        for (int j = 1; j <= k; j++) {
            // Se genera el signo alternante
            result += (2 * (i % 2) - 1) / pow(i + j, s);
        }
    }
    return result * pow(2.0, s);
}

int main() {
    int N = 1000;         // Tamaño del vector y límite superior
    double s = 2.0;       // Parámetro de entrada (s=2 converge a pi^2/6)
    
    // Reserva de memoria para el arreglo X
    double *X = (double *)malloc(N * sizeof(double));
    if (X == NULL) {
        printf("Error: No se pudo asignar memoria.\n");
        return 1;
    }

    double start_time = omp_get_wtime();

    // Paralelización del bucle 
    // tiempo schedule(static): 2.62 seg promedio
    // tiempo schedule(dynamic): 1.07 seg promedio
    // tiempo schedule(guided): 1.06 seg promedio
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < N; k++) {
        X[k] = Riemann_Zeta(s, k);
    }

    double end_time = omp_get_wtime();

    printf("Tiempo de ejecucion: %f segundos\n", end_time - start_time);
    printf("Aproximacion de Zeta(%f) para k = %d: %f\n", s, N-1, X[N-1]);

    free(X);
    return 0;
}