#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Mismos parámetros que la versión 2D para garantizar una comparación científica justa
#define GLOBAL_SIZE 1500    
#define MAX_ITERS 5000      
#define TIME_FILE_1D "tiempo_1d.csv"

// Función Gaussiana (Término fuente - idéntica a la versión 2D)
double get_source_term(int x, int y, int global_size) {
    double center = global_size / 2.0;
    double sigma = global_size / 10.0; 
    double amplitude = 100.0;
    double dx = x - center;
    double dy = y - center;
    return amplitude * exp(-(dx*dx + dy*dy) / (2.0 * sigma * sigma));
}

int main(int argc, char** argv) {
    int rank, numprocs;
    
    // 1. Inicialización de MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // 2. Partición 1D: División estricta de filas
    // Cada proceso recibe una franja horizontal completa de la matriz
    int local_rows = GLOBAL_SIZE / numprocs;
    int local_cols = GLOBAL_SIZE; // Se conserva el ancho total de la matriz
    
    // Coordenada global Y donde inicia esta franja
    int global_start_row = rank * local_rows;

    // 3. Asignación de memoria (Halos solo necesarios arriba y abajo)
    int halo_rows = local_rows + 2;
    
    double *data     = (double*)calloc(halo_rows * local_cols, sizeof(double));
    double *data_new = (double*)calloc(halo_rows * local_cols, sizeof(double));
    double *rho      = (double*)calloc(halo_rows * local_cols, sizeof(double));

    // Llenado de la matriz fuente 'rho'
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 0; j < local_cols; j++) {
            int global_y = global_start_row + (i - 1);
            rho[i * local_cols + j] = get_source_term(j, global_y, GLOBAL_SIZE);
        }
    }

    // 4. Identificar vecinos lineales (Solo Norte y Sur)
    // MPI_PROC_NULL asegura que los procesos de los extremos (0 y numprocs-1) 
    // no intenten comunicarse fuera de los límites, ignorando la operación silenciosamente.
    int nbr_up   = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int nbr_down = (rank == numprocs - 1) ? MPI_PROC_NULL : rank + 1;

    // --- INICIO DE MEDICIÓN DE TIEMPO ---
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // 5. Bucle Principal de Iteración (Jacobi)
    for (int iter = 0; iter < MAX_ITERS; iter++) {
        MPI_Request reqs[4];
        int req_cnt = 0;

        // -- INTERCAMBIO DE HALOS 1D --
        // Al tener el ancho completo de las filas, toda la memoria intercambiada es contigua.
        // No hay comunicación horizontal ni uso de MPI_Type_vector.

        // Enviar al SUR (última fila calculada), Recibir del NORTE (halo superior)
        MPI_Isend(&data[local_rows * local_cols], local_cols, MPI_DOUBLE, nbr_down, 0, MPI_COMM_WORLD, &reqs[req_cnt++]);
        MPI_Irecv(&data[0],                       local_cols, MPI_DOUBLE, nbr_up,   0, MPI_COMM_WORLD, &reqs[req_cnt++]);
        
        // Enviar al NORTE (primera fila calculada), Recibir del SUR (halo inferior)
        MPI_Isend(&data[1 * local_cols],                 local_cols, MPI_DOUBLE, nbr_up,   1, MPI_COMM_WORLD, &reqs[req_cnt++]);
        MPI_Irecv(&data[(local_rows + 1) * local_cols],  local_cols, MPI_DOUBLE, nbr_down, 1, MPI_COMM_WORLD, &reqs[req_cnt++]);

        // Esperar a que los halos superior e inferior estén listos
        MPI_Waitall(req_cnt, reqs, MPI_STATUSES_IGNORE);

        // -- CÁLCULO DE LA REGLA DE ACTUALIZACIÓN --
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < local_cols; j++) {
                int global_y = global_start_row + (i - 1);
                
                // Condiciones de frontera absolutas (los 4 bordes del dominio global)
                if (j == 0 || j == GLOBAL_SIZE - 1 || global_y == 0 || global_y == GLOBAL_SIZE - 1) {
                    data_new[i * local_cols + j] = 0.0;
                } else {
                    data_new[i * local_cols + j] = (data[(i+1) * local_cols + j] + 
                                                   data[(i-1) * local_cols + j] + 
                                                   data[i * local_cols + (j+1)] + 
                                                   data[i * local_cols + (j-1)] + 
                                                   rho[i * local_cols + j]) / 4.0;
                }
            }
        }

        // Intercambio de punteros
        double *temp = data; 
        data = data_new; 
        data_new = temp;
    }

    // --- FIN DE MEDICIÓN DE TIEMPO ---
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    double max_time;
    
    // Calculamos el tiempo máximo tomado por cualquier proceso
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 6. Guardado de métricas en archivo (Solo el rango 0)
    if (rank == 0) {
        // Modo "append" para acumular el historial de pruebas
        FILE *ft = fopen(TIME_FILE_1D, "a");
        if (ft != NULL) {
            fprintf(ft, "%d, %f\n", numprocs, max_time);
            fclose(ft);
            printf("[1D] Simulación con %d procesos. Tiempo: %f seg. Guardado en %s\n", numprocs, max_time, TIME_FILE_1D);
        }
    }

    // Limpieza de memoria (Nota: no guardamos los datos físicos aquí para evitar duplicar el archivo grande de la v2D)
    free(data); 
    free(data_new); 
    free(rho);
    
    MPI_Finalize();
    return 0;
}
