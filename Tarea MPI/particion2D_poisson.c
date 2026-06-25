#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Parámetros globales de la simulación
#define GLOBAL_SIZE 1500    // Tamaño de la malla global (1000x1000)
#define MAX_ITERS 5000      // Número de iteraciones de Jacobi
#define OUTPUT_FILE "poisson_data.csv"
#define TIME_FILE "tiempo_2d.csv"  // Archivo exclusivo para guardar el rendimiento

// Función para calcular el término heterogéneo f(x,y) - Distribución Gaussiana
double get_source_term(int x, int y, int global_size) {
    // Centramos la gaussiana en el medio de la malla global
    double center = global_size / 2.0;
    double sigma = global_size / 10.0; 
    double amplitude = 100.0;
    
    double dx = x - center;
    double dy = y - center;
    
    // Ecuación de la campana de Gauss
    return amplitude * exp(-(dx*dx + dy*dy) / (2.0 * sigma * sigma));
}

int main(int argc, char** argv) {
    int rank, numprocs;
    
    // 1. Inicialización de MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // 2. Creación de la Topología Cartesiana 2D
    int dims[2] = {0, 0}; // Dejamos que MPI calcule la mejor distribución (ej. 4 procesos -> 2x2)
    MPI_Dims_create(numprocs, 2, dims);
    
    int periods[2] = {0, 0}; // Bordes fijos, sin condiciones periódicas (no es un toroide)
    MPI_Comm cart_comm;
    // Creamos el nuevo comunicador con topología cartesiana
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    
    // Obtenemos las coordenadas (X,Y) de este proceso específico dentro de la cuadrícula
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Encontramos los rangos de los procesos vecinos (Norte, Sur, Este, Oeste)
    int nbr_up, nbr_down, nbr_left, nbr_right;
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_up, &nbr_down);    // Desplazamiento vertical (eje 0)
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_left, &nbr_right); // Desplazamiento horizontal (eje 1)

    // 3. Cálculo de los tamaños locales del bloque (Tile)
    int local_rows = GLOBAL_SIZE / dims[0];
    int local_cols = GLOBAL_SIZE / dims[1];

    // Calculamos las coordenadas globales donde inicia el bloque de este proceso
    int global_start_row = coords[0] * local_rows;
    int global_start_col = coords[1] * local_cols;

    // 4. Asignación de memoria (incluyendo celdas fantasma o 'halos' -> +2)
    int halo_rows = local_rows + 2;
    int halo_cols = local_cols + 2;
    
    double *data     = (double*)calloc(halo_rows * halo_cols, sizeof(double));
    double *data_new = (double*)calloc(halo_rows * halo_cols, sizeof(double));
    double *rho      = (double*)calloc(halo_rows * halo_cols, sizeof(double));

    // Llenamos la matriz fuente 'rho' con la distribución Gaussiana
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= local_cols; j++) {
            int global_y = global_start_row + (i - 1);
            int global_x = global_start_col + (j - 1);
            rho[i * halo_cols + j] = get_source_term(global_x, global_y, GLOBAL_SIZE);
        }
    }

    // 5. Creación de Tipo de Dato Derivado para las Columnas
    // Necesario porque en C la memoria es "row-major". Las filas son contiguas, las columnas no.
    MPI_Datatype column_type;
    // Parámetros: bloques(local_rows), tamaño de bloque(1), salto entre bloques(halo_cols)
    MPI_Type_vector(local_rows, 1, halo_cols, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    // --- INICIO DE MEDICIÓN DE TIEMPO ---
    // Barrera para que ningún proceso inicie el reloj antes que los demás
    MPI_Barrier(MPI_COMM_WORLD); 
    double start_time = MPI_Wtime();

    // 6. Bucle principal de Iteración de Jacobi
    for (int iter = 0; iter < MAX_ITERS; iter++) {
        MPI_Request reqs[8];
        int req_cnt = 0;

        // -- INTERCAMBIO DE HALOS (Comunicaciones no bloqueantes) --
        
        // Eje Vertical (Filas contiguas)
        MPI_Isend(&data[local_rows * halo_cols + 1], local_cols, MPI_DOUBLE, nbr_down, 0, cart_comm, &reqs[req_cnt++]);
        MPI_Irecv(&data[0 * halo_cols + 1],          local_cols, MPI_DOUBLE, nbr_up,   0, cart_comm, &reqs[req_cnt++]);
        MPI_Isend(&data[1 * halo_cols + 1],            local_cols, MPI_DOUBLE, nbr_up,   1, cart_comm, &reqs[req_cnt++]);
        MPI_Irecv(&data[(local_rows + 1) * halo_cols + 1], local_cols, MPI_DOUBLE, nbr_down, 1, cart_comm, &reqs[req_cnt++]);
        
        // Eje Horizontal (Columnas no contiguas -> Usamos column_type)
        MPI_Isend(&data[1 * halo_cols + local_cols], 1, column_type, nbr_right, 2, cart_comm, &reqs[req_cnt++]);
        MPI_Irecv(&data[1 * halo_cols + 0],          1, column_type, nbr_left,  2, cart_comm, &reqs[req_cnt++]);
        MPI_Isend(&data[1 * halo_cols + 1],              1, column_type, nbr_left,  3, cart_comm, &reqs[req_cnt++]);
        MPI_Irecv(&data[1 * halo_cols + (local_cols + 1)], 1, column_type, nbr_right, 3, cart_comm, &reqs[req_cnt++]);

        // Esperamos a que todas las transferencias de bordes finalicen
        MPI_Waitall(req_cnt, reqs, MPI_STATUSES_IGNORE);

        // -- CÁLCULO DE LA REGLA DE ACTUALIZACIÓN (Ecuación de Poisson discretizada) --
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= local_cols; j++) {
                int global_y = global_start_row + (i - 1);
                int global_x = global_start_col + (j - 1);
                
                // Condición de frontera de Dirichlet: los bordes absolutos de la matriz global son 0
                if (global_x == 0 || global_x == GLOBAL_SIZE - 1 || 
                    global_y == 0 || global_y == GLOBAL_SIZE - 1) {
                    data_new[i * halo_cols + j] = 0.0;
                } else {
                    // Promedio de los 4 vecinos más el término fuente
                    data_new[i * halo_cols + j] = (data[(i+1) * halo_cols + j] + 
                                                   data[(i-1) * halo_cols + j] + 
                                                   data[i * halo_cols + (j+1)] + 
                                                   data[i * halo_cols + (j-1)] + 
                                                   rho[i * halo_cols + j]) / 4.0;
                }
            }
        }

        // Intercambio de punteros para la siguiente iteración
        double *temp = data;
        data = data_new;
        data_new = temp;
    }

    // --- FIN DE MEDICIÓN DE TIEMPO ---
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    double max_time;
    
    // Obtenemos el tiempo del proceso que más tardó (el cuello de botella define el tiempo total)
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 7. Guardado de métricas de rendimiento (Solo el rango 0)
    if (rank == 0) {
        // "a" (append) permite añadir líneas sin borrar ejecuciones previas
        FILE *ft = fopen(TIME_FILE, "a"); 
        if (ft != NULL) {
            fprintf(ft, "%d, %f\n", numprocs, max_time); // Formato: Procesadores, Tiempo
            fclose(ft);
            printf("[2D] Simulación con %d procesos. Tiempo: %f seg. Guardado en %s\n", numprocs, max_time, TIME_FILE);
        }
    }

    // 8. Guardado de datos físicos secuencialmente para evitar corrupción de archivos
    if (rank == 0) {
        FILE *fp = fopen(OUTPUT_FILE, "w"); // "w" sobreescribe el archivo de datos físicos
        fprintf(fp, "X,Y,Value\n");
        fclose(fp);
    }
    
    // Sincronización: cada proceso escribe por turnos
    for (int p = 0; p < numprocs; p++) {
        MPI_Barrier(MPI_COMM_WORLD); 
        if (rank == p) {
            FILE *fp = fopen(OUTPUT_FILE, "a"); 
            for (int i = 1; i <= local_rows; i++) {
                for (int j = 1; j <= local_cols; j++) {
                    int global_y = global_start_row + (i - 1);
                    int global_x = global_start_col + (j - 1);
                    fprintf(fp, "%d,%d,%f\n", global_x, global_y, data[i * halo_cols + j]);
                }
            }
            fclose(fp);
        }
    }

    // Liberación de recursos
    MPI_Type_free(&column_type);
    free(data); 
    free(data_new); 
    free(rho);
    
    MPI_Finalize();
    return 0;
}