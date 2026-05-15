#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define J 1.0
#define SWEEPS 2000 

// Estructura para almacenar las distintas configuraciones a simular.
typedef struct {
    int L;
    double T;
} SimConfig;

double calc_delta_E(int *grid, int L, int x, int y) {
    int up    = grid[((x - 1 + L) % L) * L + y];
    int down  = grid[((x + 1) % L) * L + y];
    int left  = grid[x * L + (y - 1 + L) % L];
    int right = grid[x * L + (y + 1) % L];
    
    return 2.0 * J * grid[x * L + y] * (up + down + left + right);
}

// Función para calcular la energía y magnetización media por sitio en un instante dado.
void calc_observables(int *grid, int L, double *E_site, double *M_site) {
    double E = 0;
    double M = 0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            M += grid[i * L + j];
            int down  = grid[((i + 1) % L) * L + j];
            int right = grid[i * L + ((j + 1) % L)];
            E += -J * grid[i * L + j] * (down + right);
        }
    }
    *E_site = E / (L * L);
    *M_site = fabs(M) / (L * L);
}

int main() {
    //4 configuraciones a probar simultáneamente
    int num_configs = 4;
    SimConfig configs[] = {
        {16, 2.0},   
        {32, 2.0},   
        {16, 3.0},   
        {32, 3.0}
    };

    FILE *fp = fopen("resultados_equilibracion.csv", "w");
    if (fp == NULL) {
        printf("Error al crear el archivo.\n");
        return 1;
    }
    
    fprintf(fp, "L,T,Sweep,Iteracion_Individual,E_por_sitio,M_por_sitio\n");

    #pragma omp parallel for default(none) shared(configs, fp, num_configs)
    for (int idx = 0; idx < num_configs; idx++) {
        
        int L = configs[idx].L;
        double T = configs[idx].T;
        unsigned int seed = 777 + omp_get_thread_num() + idx;

        // Reservar memoria dinámicamente para la grilla particular de este hilo
        int *grid = (int *)malloc(L * L * sizeof(int));
        
        // Reservar memoria para guardar el historial de la serie temporal en este hilo.
        // Hacemos esto para no colapsar la escritura en archivo durante la simulación.
        double *hist_E = (double *)malloc(SWEEPS * sizeof(double));
        double *hist_M = (double *)malloc(SWEEPS * sizeof(double));

        // Condición inicial aleatoria
        for (int i = 0; i < L * L; i++) {
            grid[i] = (rand_r(&seed) % 2 == 0) ? 1 : -1;
        }

        // Bucle de equilibración
        for (int sweep = 0; sweep < SWEEPS; sweep++) {
            for (int step = 0; step < L * L; step++) {
                int x = rand_r(&seed) % L;
                int y = rand_r(&seed) % L;
                double dE = calc_delta_E(grid, L, x, y);

                if (dE <= 0 || (double)rand_r(&seed) / RAND_MAX < exp(-dE / T)) {
                    grid[x * L + y] *= -1;
                }
            }

            calc_observables(grid, L, &hist_E[sweep], &hist_M[sweep]);
        }

        #pragma omp critical
        {
            for (int sweep = 0; sweep < SWEEPS; sweep++) {
                int iter_individual = sweep * L * L;
                fprintf(fp, "%d,%.2f,%d,%d,%e,%e\n", 
                        L, T, sweep, iter_individual, hist_E[sweep], hist_M[sweep]);
            }
            printf("Hilo %d finalizo configuracion L=%d, T=%.2f\n", omp_get_thread_num(), L, T);
        }

        free(grid);
        free(hist_E);
        free(hist_M);
    }

    fclose(fp);
    printf("Simulaciones finalizadas. Datos guardados en resultados_equilibracion.csv\n");

    return 0;
}
