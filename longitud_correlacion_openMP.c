#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> 
#define L 32           // Tamaño de la grilla (32x32). 
#define J 1.0          
#define EQ_STEPS 50000 // Pasos de termalización.
#define MC_STEPS 50000 // Pasos de muestreo (Montecarlo) donde tomaremos las mediciones estadísticas.
#define MAX_R (L/2)    // Distancia máxima a medir. 

double calc_delta_E(int grid[L][L], int x, int y) {
    int up    = grid[(x - 1 + L) % L][y]; // Vecino superior
    int down  = grid[(x + 1) % L][y];     // Vecino inferior
    int left  = grid[x][(y - 1 + L) % L]; // Vecino izquierdo
    int right = grid[x][(y + 1) % L];     // Vecino derecho
    return 2.0 * J * grid[x][y] * (up + down + left + right);
}

int main() {
    int num_T = 66;       // Evaluaremos 66 puntos de temperatura 
    double T_min = 1.4;   // Temperatura inicial
    double step_T = 0.025; // Incrementos de temperatura 

    //Archivos de salida
    FILE *fp_xi = fopen("resultados_xi_ejercicio_2.csv", "w");
    if (fp_xi == NULL) {
        printf("Error al crear el archivo de Xi.\n");
        return 1;
    }
    fprintf(fp_xi, "T,Xi\n"); // Cabecera

    // Archivo 2: Guarda Temperatura, Distancia R y Función de Correlación 
    FILE *fp_Gr = fopen("resultados_Gr_ejercicio_2.csv", "w");
    if (fp_Gr == NULL) {
        printf("Error al crear el archivo de G(R).\n");
        fclose(fp_xi); 
        return 1;
    }
    fprintf(fp_Gr, "T,R,G_R\n"); 

    #pragma omp parallel for default(none) \
            shared(num_T, T_min, step_T, fp_xi, fp_Gr)
    for (int idx = 0; idx < num_T; idx++) {
        double T = T_min + idx * step_T; 
        int grid[L][L];                  
        
        // Generación de una semilla única para el hilo actual, basada en su ID y la iteración.
        // Esto garantiza que cada simulación tenga una secuencia aleatoria totalmente distinta
        unsigned int seed = 8888 + omp_get_thread_num() + idx;

        // Se rellena la grilla con espines aleatorios.
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                grid[i][j] = (rand_r(&seed) % 2 == 0) ? 1 : -1;
            }
        }

        for (int step = 0; step < EQ_STEPS; step++) {
            for (int n = 0; n < L * L; n++) {
                // Seleccionamos un espín al azar
                int i = rand_r(&seed) % L;
                int j = rand_r(&seed) % L;
                double dE = calc_delta_E(grid, i, j); // Calculamos su diferencia de energía

                // criterio de metropolis
                if (dE <= 0 || (double)rand_r(&seed) / RAND_MAX < exp(-dE / T)) {
                    grid[i][j] *= -1; // Inversión del espín
                }
            }
        }

        // Arreglos para acumular las mediciones estadísticas.
        double sum_corr[MAX_R] = {0}; 
        double sum_m = 0;             

        for (int step = 0; step < MC_STEPS; step++) {
            for (int n = 0; n < L * L; n++) {
                int i = rand_r(&seed) % L;
                int j = rand_r(&seed) % L;
                double dE = calc_delta_E(grid, i, j);

                if (dE <= 0 || (double)rand_r(&seed) / RAND_MAX < exp(-dE / T)) {
                    grid[i][j] *= -1;
                }
            }

            double m_actual = 0;
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < L; j++) {
                    m_actual += grid[i][j];
                    for (int r = 0; r < MAX_R; r++) {
                        sum_corr[r] += grid[i][j] * grid[i][(j + r) % L];
                    }
                }
            }
            sum_m += (m_actual / (L * L)); // magnetización media por sitio
        }

        double avg_m = sum_m / MC_STEPS;
        
        double Gr_array[MAX_R];

        for (int r = 0; r < MAX_R; r++) {
            double avg_corr = sum_corr[r] / (MC_STEPS * L * L);
            Gr_array[r] = avg_corr - (avg_m * avg_m);
        }

        //regresion lineal para estimar la longitud de correlacion 
        int count = 0;
        double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
        
        for (int r = 1; r < MAX_R; r++) {
            if (Gr_array[r] > 0) {
                double x = (double)r;          // Eje X: Distancia R
                double y = log(Gr_array[r]);   // Eje Y: Logaritmo de G
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
                count++;
            } else {
                break; 
            }
        }

        double xi = 0.0;
        // Solo podemos trazar una recta si sobrevivieron al menos 2 puntos válidos
        if (count > 1) {
            double denominador = (count * sum_xx - sum_x * sum_x);
            if (denominador != 0) {
                // calculo pendiente
                double m = (count * sum_xy - sum_x * sum_y) / denominador;
                if (m < 0) {
                    xi = -1.0 / m; 
                }
            }
        }

        #pragma omp critical
        {
            fprintf(fp_xi, "%.2f,%f\n", T, xi);

            for (int r = 0; r < MAX_R; r++) {
                fprintf(fp_Gr, "%.2f,%d,%e\n", T, r, Gr_array[r]);
            }
        }
    } //fin del hilo paralelo. El hilo queda libre para tomar la siguiente temperatura.

    fclose(fp_xi);
    fclose(fp_Gr);
    
    printf("Simulaciones completadas.\n");
    printf(" - Creado: resultados_xi_ejercicio_2.csv\n");
    printf(" - Creado: resultados_Gr_ejercicio_2.csv\n");

    return 0;
}
